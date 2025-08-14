from abc import abstractmethod
from azure.search.documents import SearchClient
from openai import AzureOpenAI
from ..base_generator import BaseGenerator
import diversity.diversity_generator as DiversityGenerator
from tracing.telemetry import traced
from models import DiversityInjection, QATagged, QA, QAVerificationResult
from search.search_service import SearchService

class BaseSingleDocumentQAGenerator(BaseGenerator):

    def __init__(
            self, 
            diversity_generator: DiversityGenerator, 
            search_client: SearchClient,
            llm_client: AzureOpenAI,
    ):
        super().__init__(
            diversity_generator = diversity_generator, 
            search_client = search_client, 
            llm_client = llm_client,
        )

    @abstractmethod
    @traced("generator.verify_qa")
    def _verify_qa(self, qa: QA) -> QAVerificationResult:
        """
        Verify the generated Q&A for quality and accuracy.
        
        This should be implemented by subclasses to perform specific verification
        checks on the generated Q&A.
        
        Args:
            qa (QA): The generated Q&A object.
        
        Returns:
            bool: True if the Q&A is valid, False otherwise.
        """
        pass

    @abstractmethod
    def _get_base_prompt_generator(self) -> str:
        """
        Get the base prompt for this generator.
        
        This should be implemented by subclasses to return the specific base prompt
        used for generating questions and answers.
        
        Returns:
            str: The base prompt for this generator.
        """
        pass

    @abstractmethod
    def _get_tags(self) -> dict:
        """
        Get the tags for the generated Q&A.
        
        This should be implemented by subclasses to return the specific tags
        used for categorizing the generated questions and answers.
        
        Returns:
            dict: A dictionary of tags for the generated Q&A.
        """
        pass

    @traced("generator.generate")
    def generate(self) -> QATagged:
        """
        Generate a question-answer pair from a single document using all available chunks.
        
        This method implements a comprehensive Q&A generation pipeline that selects a random
        document chunk as a starting point, retrieves all chunks from the same document,
        and uses them as context to generate a high-quality Q&A pair using an LLM with
        structured output parsing and automatic retry mechanisms.
        
        The generation process follows these steps:
        1. Select a random chunk from the search service as the seed
        2. Retrieve all chunks belonging to the same source document
        3. Sort chunks by part number to maintain document structure
        4. Build comprehensive context string from all document chunks
        5. Generate diversity injection guidelines for varied question types
        6. Create system prompt combining base prompt, diversity guidelines, and context
        7. Call LLM with structured output parsing (QA model) and retry tool
        8. Handle LLM-initiated retries if generation is deemed impossible
        9. Verify the generated Q&A meets quality criteria via subclass implementation
        10. Package the result with complete metadata and chunk content
        
        Returns:
            QATagged: A comprehensive Q&A object containing:
                - question (str): The generated question
                - answer (str): The generated answer
                - tags (dict): Question categorization tags (implementation-specific)
                - diversity_injection (DiversityInjection): Applied diversity guidelines
                - chunk_ids (list[str]): IDs of chunks referenced in the answer
                - chunk_content (list[str]): Full text content of referenced chunks
        
        Raises:
            Exception: Propagated from context building or search service operations
        
        Note:
            - Uses OpenTelemetry tracing for observability and debugging
            - Implements automatic retry logic via LLM tool calls when generation fails
            - Subclasses must implement _verify_qa(), _get_base_prompt(), and _get_tags()
            - Uses temperature=1.0 for diverse question generation
            - Validates answers against specified chunk content through subclass verification
            - Retry mechanism ensures robust generation even with challenging context
        """

        random_chunk = self.search_service.get_random_chunk()

        all_chunks_for_document = self.search_service.get_all_chunks_of_document(random_chunk)

        all_chunks_for_document_sorted = self.search_service.sort_chunks_by_part_number(all_chunks_for_document)

        try:
            context = self._build_context(all_chunks_for_document_sorted)
        except Exception as e:
            print(f"Error building context: {e}")
            return self._retry_logic()

        diversity_injection = self.diversity_generator.get_diversity_injection()

        system_prompt = self._build_system_prompt(context=context, diversity_injection=diversity_injection)

        # Make the OpenAI call with manual tracing
        with self.tracer.start_as_current_span("generator.openai.chat") as span:
            span.set_attribute("gen_ai.prompt.0.content", system_prompt)
            
            qa_completion = self.llm_client.beta.chat.completions.parse(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                ],
                response_format=QA,
                max_completion_tokens=800,
                temperature=1.0,
                #top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                tool_choice="auto",
                tools=self._get_tools()
            )

            # Execute if tool call returned
            if qa_completion.choices[0].message.tool_calls:
                tool_call = qa_completion.choices[0].message.tool_calls[0]
                if tool_call.function.name == "_retry_logic":
                    print("Function call - Not possible to generate a valid Q&A according to the instructions, retrying...")
                    return self._retry_logic()

            qa = qa_completion.choices[0].message.parsed

            # Telemetry: Log the generated Q&A
            span.set_attribute("gen_ai.qa.question", qa.question)
            span.set_attribute("gen_ai.qa.ground_truth_answer", qa.ground_truth_answer)

        # Verify the quality of the generated Q&A
        verify_qa_result = self._verify_qa(qa)

        # Retry logic if the Q&A is not valid
        if not verify_qa_result.is_correct:
            return self._retry_logic()

        qa_tagged = QATagged(
            question=qa.question,
            ground_truth_answer=qa.ground_truth_answer,
            tags=self._get_tags(),
            diversity_injection=diversity_injection,
            chunk_ids=qa.chunk_ids,
            chunk_content=self._get_chunk_content_by_ids(qa.chunk_ids)
        )
            
        return qa_tagged
    
    @traced("generator.build_system_prompt")
    def _build_system_prompt(self, context: str, diversity_injection: DiversityInjection) -> str:
        """
        Build the system prompt for Q&A generation by combining base prompt, diversity guidelines, and context.
        
        Args:
            context (str): The document chunks context to include in the prompt.
            
        Returns:
            tuple[str, dict]: A tuple containing the complete system prompt and the diversity injection data.
        """
        base_prompt = self._get_base_prompt_generator()
        
        # Format the diversity injection nicely
        formatted_diversity = self.diversity_generator.get_injection_as_string(diversity_injection)

        prompt: str = (
            base_prompt
            + "\n\n# Additional guidelines for the question generation:\n"
            + "Note this instructions does not apply when generating the answer.\n"
            + formatted_diversity
            + "\n\n# Context:\n"
            + context
        )
        return prompt
    
    def _get_chunk_content_by_ids(self, chunk_ids: list[str]) -> list[str]:
        """
        Get the content of chunks by their IDs.
        
        Args:
            chunk_ids (list[str]): List of chunk IDs to retrieve content for.

        Returns:
            list[str]: List of content strings for the specified chunk IDs.
        """
        return [
            SearchService.get_search_record_content(
                chunk
            ) for chunk in self.search_service.get_search_records_by_ids(chunk_ids)
        ]
    
    def _get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "_retry_logic",
                    "description": """
                        If it is not possible to generate a valid Q&A according to the instructions, this tool will retry the generation process by retrieving new context.
                        Use this tool if the context provided is not sufficient to generate a Q&A that fully meets the requirements specified in the prompt.""",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        ]