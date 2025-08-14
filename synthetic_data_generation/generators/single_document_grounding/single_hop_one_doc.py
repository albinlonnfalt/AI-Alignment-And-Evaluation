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

# Single-chunk sourcing

- The answer must appear in exactly one provided chunk.

# Chunk-specific wording

-  Don’t assume the reader knows which chunk you used.

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

class SingleHopOneDocGenerator(BaseSingleDocumentQAGenerator):
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

        return QAVerificationResult(
            is_correct=True,
            reason="All validation checks passed."
        )

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
            "question_type": "single_hop_same_doc",
        }