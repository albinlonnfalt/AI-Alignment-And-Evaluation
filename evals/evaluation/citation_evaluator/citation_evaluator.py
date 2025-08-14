from openai import AzureOpenAI
from pydantic import BaseModel

citation_evaluator_prompt = """
You are evaluating whether a chatbot response meets the citation requirements.

The chatbot was built to assist internal employees and uses a Retrieval-Augmented Generation (RAG) architecture. It generates answers using a large language model (LLM) based on documents retrieved from an internal knowledge base.

Users have observed that the chatbot sometimes provides answers that include internal, organization-specific information but fails to include citations for the internal sources used.

The chatbot should follow this rule:

- If the answer includes organization-specific (internal) information, it must include one or more citations pointing to the source(s) of that information.
- If the answer is based only on general knowledge, then citations are not required, but they are allowed.

## Citation format:
Citations are embedded inline in the text using this format:
[citationIndex-#] — for example:

"The internal process requires a three-step approval [citationIndex-2]."

## Your task:
Evaluate whether the chatbot correctly followed the citation rule in its response.

Return your evaluation in the following structured format:

class CitationEvaluationResponse(BaseModel):
    is_valid: bool  # True if the citation behavior is correct, False otherwise
    reason: str     # A brief explanation of your reasoning

When generating your response:

Set is_valid = True if the chatbot made the correct decision about whether to include a citation (based on whether the content is organization-specific/internal or general knowledge).

Set is_valid = False if the chatbot failed to include a citation when it should have, or included one unnecessarily.

In the reason, explain whether the content is internal/organization-specific or general knowledge, and justify whether the inclusion or absence of citations is appropriate.

Note: Your task is only to evaluate whether a citation is present when required. You do not need to verify whether the citation itself is correct or points to the right content.
"""


class CitationEvaluationResponse(BaseModel):
    is_valid: bool
    reason: str


class CitationEvaluator:
    def __init__(
            self,
            model_config_dict: dict
        ):
        # Initialize the citation evaluator
        self.model_config = model_config_dict

    def __call__(
            self, 
            question: str,
            chatbot_answer: str,
        ):


        # ---- Remove to get a more nuanced evaluation ----
        # If the answer include any citations, it passes for now
        """
    if "[citationIndex-" in chatbot_answer:
            return {"is_valid": True, "reason": "The answer includes citations"}
        """

        llm_client = AzureOpenAI(
            api_version=self.model_config["api_version"],
            azure_endpoint=self.model_config["azure_endpoint"],
            api_key=self.model_config["api_key"]
        )

        result = llm_client.beta.chat.completions.parse(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": citation_evaluator_prompt},
                {"role": "user", "content": f"Question: {question} Answer: {chatbot_answer}"}
            ],
            response_format=CitationEvaluationResponse,
            max_completion_tokens=200,
            temperature=0.0,
            #top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        citation_result = result.choices[0].message.parsed

        return citation_result.model_dump()