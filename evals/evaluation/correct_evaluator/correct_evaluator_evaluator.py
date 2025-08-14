from openai import AzureOpenAI
from pydantic import BaseModel

correct_evaluator_prompt = """
Your task is to evaluate a response based on whether it is correct or not. 

You will receive a question and an answer. You will also receive the ground truth answer for comparison.
"""


class CorrectEvaluationResponse(BaseModel):
    is_valid: bool
    reason: str


class CorrectEvaluator:
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
            ground_truth_answer: str
        ):

        llm_client = AzureOpenAI(
            api_version=self.model_config["api_version"],
            azure_endpoint=self.model_config["azure_endpoint"],
            api_key=self.model_config["api_key"]
        )

        result = llm_client.beta.chat.completions.parse(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": correct_evaluator_prompt},
                {"role": "user", "content": f"Question: {question} Answer: {chatbot_answer} Ground Truth: {ground_truth_answer}"}
            ],
            response_format=CorrectEvaluationResponse,
            max_completion_tokens=200,
            temperature=0.0,
            #top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )

        correct_result = result.choices[0].message.parsed

        return correct_result.model_dump()