from azure.search.documents import SearchClient
from openai import AzureOpenAI
import diversity.diversity_generator as DiversityGenerator
from .base_single_document import BaseSingleDocumentQAGenerator
from models import QAVerificationResult, QA
from tracing.telemetry import traced

generator_prompt ="""
You are a Q&A‐generation assistant. You receive a single document divided into chunks (thousands of other documents exist but are not visible). 
Your task: generate exactly one question and its answer, drawn from a single chunk you’ve been given. 
Follow these rules:

# Dual-Chunk Sourcing Task
**Objective:** Craft a question that requires synthesizing information from exactly two distinct chunks of the provided document.

**Guidelines:**

- The question must necessitate the combination of both chunks for a complete and accurate answer.

- It should be impossible to answer the question correctly using only one chunk in isolation.

- Avoid surface-level or fact-recall questions that could be answered by a single chunk.

- Design the question to draw connections, highlight contrasts, or require reasoning across the two chunks.

**Common feedback from reviewers (Avoid these pitfalls):**
- The answer to any question about backcharges for deficient work can be answered from either chunk, as both contain this backcharge provision verbatim.


# Don’t assume the reader knows which chunk you used.

- If the question refers to details unique to this document (e.g. a particular agreement or project), name it explicitly in the question.

- Avoid “according to this document,” “in this contract,” etc. instead embed the title or identifier:

    ✔ Good: “What is the minimum penalty fee for breach of contract when doing a joint venture?”

    ✔ Good: “Who is responsible for assembling and submitting the Engineering Turnover Packages (ETOP) for each trade in the Schering-Plough Cafeteria Project?”

    ✘ Bad: “What is the penalty fee for breach of contract in this document?”

# Corpus-wide uniqueness

- Avoid using sections or clauses when referencing the document.

- Avoid using the title of the document word for word in you question.

# Clarity & self-containment

- A reader seeing only the question must know exactly where to look.

- The reader does not know which document the question is based on

Don’t use pronouns or references like “it” or “that clause” without explicitly naming the clause, section, or title.
"""

class MultiHopOneDocGenerator(BaseSingleDocumentQAGenerator):
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


    @traced("generator.verify_qa")
    def _verify_qa(self, qa: QA) -> QAVerificationResult:
        """
        Verify that the generated Q&A is valid.
        
        This method performs three validation checks:
        1. Answer uniqueness: The answer should not be found in other document chunks
        2. Chunk connection: The answer can be found in the specified chunks
        3. Multi-hop reasoning: The question requires information from multiple chunks (Multi-hop in this context means the answer cannot be found in a single chunk)

        Args:
            qa (QA): The generated Q&A object to validate
        
        Returns:
            bool: True if all validations pass, False otherwise
        """

        # Check 1: Verify answer is unique to the specified chunks (not found elsewhere)
        """
        is_answer_unique = self._verify_qa_answer_not_in_other_chunks(qa)

        if not is_answer_unique.is_correct:
            return QAVerificationResult(
                is_correct=False,
                reason=f"Answer '{qa.ground_truth_answer}' found in other chunks. Detailed reason: {is_answer_unique.reason}"
            )
        """
            
        # Check 2: Verify the answer can actually be found in the specified chunks
        answer_has_chunk_connection = self._verify_qa_chunk_connection(qa)

        if not answer_has_chunk_connection.is_correct:
            return QAVerificationResult(
                is_correct=False,
                reason=f"Answer '{qa.ground_truth_answer}' not found in specified chunks. Detailed reason: {answer_has_chunk_connection.reason}"
            )

        # Check 3: Verify that the question requires multi-hop reasoning across multiple chunks
        question_requires_multi_hop = self._verify_qa_question_requires_multi_hop(qa)

        if not question_requires_multi_hop.is_correct:
            return QAVerificationResult(
                is_correct=False,
                reason=f"Question '{qa.question}' does not require multi-hop reasoning. Detailed reason: {question_requires_multi_hop.reason}"
            )

        return QAVerificationResult(
            is_correct=True,
            reason="All validation checks passed."
        )
    
    @traced("generator.verify_qa_question_requires_multi_hop")
    def _verify_qa_question_requires_multi_hop(self, qa: QA) -> QAVerificationResult:
        """
        Verify that the question requires multi-hop reasoning across multiple chunks.
        
        Args:
            qa (QA): The generated Q&A object.
        
        Returns:
            QAVerificationResult: True if the question requires multi-hop reasoning, False otherwise.
        """

        results = self.search_service.get_search_records_by_ids(qa.chunk_ids)
        
        try:
            context = self._build_context(results)
        except Exception as e:
            print(f"Error building context: {e}")
            return QAVerificationResult(is_correct=False, reason="Error building context")

        system_prompt = f"""
        Your role as a helpful assistant is to verify that answering the question requires drawing on information from BOTH chunks in the provided context.
        If the answer to the question REQUIRES information from BOTH chunks, return True, 
        If the answer to the question can be answered by ONE of the chunks alone, return False.

        is_correct: True if the question can only be answered by combining information from the two chunks, otherwise False.

        Provide a reason for your answer. The reason should be less then 30 words.

        # Question:
        {qa.question}

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

        if not qa_validation.is_correct:
            print(f"Q&A verification failed: {qa_validation.reason}")

        return qa_validation
    
    def _get_base_prompt_generator(self) -> str:
        """
        Get the base prompt for this generator.
        
        This should be implemented by subclasses to return the specific base prompt
        used for generating questions and answers.
        
        Returns:
            str: The base prompt for this generator.
        """
        return generator_prompt
    
    def _get_tags(self) -> dict:
        """
        Get the tags for the generated Q&A.
        
        This should be implemented by subclasses to return the specific tags
        used for categorizing the generated questions and answers.
        
        Returns:
            dict: A dictionary of tags for the generated Q&A.
        """
        return {
            "question_type": "multi_hop_same_doc",
        }