import sys
from abc import ABC, abstractmethod
from azure.search.documents import SearchClient
from openai import AzureOpenAI
import diversity.diversity_generator as DiversityGenerator
from search.search_service import SearchService
from models import QA, QAVerificationResult, QATagged
from tracing.telemetry import get_tracer, traced
from typing import Iterable


class BaseGenerator(ABC):

    MAX_RETRIES = 10

    def __init__(
            self, 
            diversity_generator: DiversityGenerator, 
            search_client: SearchClient, 
            llm_client: AzureOpenAI,
    ):
        self.diversity_generator = diversity_generator
        self.search_service = SearchService(search_client, llm_client)
        self.llm_client = llm_client
        self.retry_count = 0  # Initialize retry count
        self.tracer = get_tracer(f"generator-{self.__class__.__name__}")

    @abstractmethod
    def generate(self) -> QATagged:
        pass

    @abstractmethod
    def _verify_qa(self, qa: QA) -> QAVerificationResult:
        """
        Verify the generated Q&A for quality and accuracy.
        
        This should be implemented by subclasses to perform specific verification
        checks on the generated Q&A.
        
        Args:
            qa (QA): The generated Q&A object.
        
        Returns:
            QAVerificationResult: True if the Q&A is valid, False otherwise.
        """
        pass

    @abstractmethod
    def _build_context(self, results: Iterable[dict]) -> str:
        """
        Build a context string from the search results.
        
        This should be implemented by subclasses to format the context appropriately.
        
        Args:
            results (Iterable[dict]): The search results to build context from.
        
        Returns:
            str: The formatted context string.
        """
        pass


    @traced("generator.retry_logic")
    def _retry_logic(self) -> QATagged:
        # Check number of retries
        if self.retry_count < self.MAX_RETRIES:
            print("The Q&A generation failed, retrying...")
            self.retry_count += 1
            print(f"Retrying... Attempt {self.retry_count}")
            # Retry generating the Q&A
            return self.generate()
        else:
            print("Max retries reached. Giving up. Stop program")
            sys.exit(1)

    
    @traced("generator.verify_answer_uniqueness")
    def _verify_qa_answer_not_in_other_chunks(self, qa: QA) -> QAVerificationResult:
        """
        Verify that the answer to the question is not found in any other chunks.
        
        Args:
            qa (QA): The generated Q&A object.
        
        Returns:
            bool: True if the answer is not found in other chunks, False otherwise.
        """
        results = self.search_service.qa_search(qa)

        # If results is empty, return a verification result indicating no relevant chunks found
        if not results:
            return QAVerificationResult(is_correct=True, reason="No other relevant chunks found with similar content")

        try:
            context = self._build_context(results)
        except Exception as e:
            print(f"Error building context: {e}")
            return QAVerificationResult(is_correct=False, reason="Error building context")

        system_prompt = f"""
        You are a helpful assistant. Your task is to verify that the answer to the question can NOT be found in the context provided.
        If the answer can NOT be found in the context, return True, otherwise return False.

        is_correct: True if the answer can NOT be found in the context, otherwise False.

        Provide a reason for your answer. The reason should be less then 30 words.

        Question: {qa.question}

        # Context:
        {context}

        """
        
        # Make the OpenAI call with manual tracing
        with self.tracer.start_as_current_span("generator.openai.chat") as span:
            span.set_attribute("gen_ai.prompt.0.content", system_prompt)

            qa_validation_completion = self.llm_client.beta.chat.completions.parse(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                ],
                response_format=QAVerificationResult,
                max_completion_tokens=800,
                temperature=0.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )

            qa_validation = qa_validation_completion.choices[0].message.parsed

            span.set_attribute("gen_ai.qa.is_correct", qa_validation.is_correct)
            span.set_attribute("gen_ai.qa.reason", qa_validation.reason)

        # Console print
        if not qa_validation.is_correct:
            print(f"Q&A validation failed: {qa_validation.reason}")

        return qa_validation

    @traced("generator.verify_chunk_connection")
    def _verify_qa_chunk_connection(self, qa: QA) -> QAVerificationResult:
        """
        Verify that the generated Q&A is valid by checking if the answer can be found in the context of the chunks.
        Args:
            qa (QA): The generated Q&A object.
        
        Returns:
            bool: True if valid, False otherwise.
        """
        results = self.search_service.get_search_records_by_ids(qa.chunk_ids)
        
        if not results:
            return QAVerificationResult(
                is_correct=False, 
                reason="No chunks found for the given chunk IDs"
            )
        
        try:
            context = self._build_context(results)
        except Exception as e:
            print(f"Error building context: {e}")
            return QAVerificationResult(is_correct=False, reason="Error building context")

        system_prompt = f"""
        You are a helpful assistant. Your task is to verify that a comprehensive answer to a question exists in the context provided.
        
        If the answer to the question can be found in the context answer True otherwise return False.
        Provide a reason for your answer. The reason should be less then 30 words.

        Question: {qa.question}

        # Context:
        {context}
        """
        
        # Make the OpenAI call with manual tracing
        with self.tracer.start_as_current_span("generator.openai.chat") as span:
            span.set_attribute("gen_ai.prompt.0.content", system_prompt)

            qa_validation_completion = self.llm_client.beta.chat.completions.parse(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                ],
                response_format=QAVerificationResult,
                max_completion_tokens=800,
                temperature=0.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )

            qa_validation = qa_validation_completion.choices[0].message.parsed

            span.set_attribute("gen_ai.qa.is_correct", qa_validation.is_correct)
            span.set_attribute("gen_ai.qa.reason", qa_validation.reason)

        # Console print
        if not qa_validation.is_correct:
            print(f"Q&A validation failed: {qa_validation.reason}")

        return qa_validation
    
    @traced("generator._build_context")
    def _build_context(self, search_results) -> str:
        """
        Build a context string from search results.
        
        Args:
            search_results: Iterable of search results from Azure AI Search containing chunk data.
        
        Returns:
            str: Formatted context string with titles and content.
        """
        context_parts = []
        previous_title = None
        has_results = False
        
        # Process results in a single iteration
        for result in search_results:
            has_results = True
            
            # Get title from current result
            current_title = self.search_service.get_search_record_title(result)
            
            # Add title if it's different from the previous one
            if current_title != previous_title:
                if current_title:
                    context_parts.append("Document Title: " + current_title)
                previous_title = current_title
            
            # Build chunk content and add immediately after title
            chunk_id = self.search_service.get_search_record_id(result)
            content = self.search_service.get_search_record_content(result)
            context_parts.append(f"CHUNK_ID: {chunk_id}\nContent: {content}")
        
        # Handle empty results
        if not has_results:
            raise ValueError("No search results found. Cannot build context.")
        
        return "\n\n".join(context_parts)