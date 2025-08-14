import os
import warnings
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import urllib3
# Suppress urllib3 InsecureRequestWarning for localhost HTTPS requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from openai import AzureOpenAI
from azure.ai.evaluation import AzureOpenAIModelConfiguration
from answer_generation import AuthService, AnswerFactory, AnswerGenerator
from evaluation.evaluation_factory import EvaluationFactory
from visualizer.visualizer import Visualizer

# The path to question & answers used as input to the evaluation
input_file_path = "data/q-a/qa_generated_20250711_141152.json"

def main():
    auth_service = AuthService()

    chatbot_answer_generator = AnswerGenerator(
        base_url=os.getenv("CHATBOT_BACKEND_ENDPOINT"),
        auth_service=auth_service,
        expert_identifier=os.getenv("CHATBOT_EXPERT_IDENTIFIER")
    )

    answer_factory = AnswerFactory(
        answer_generator=chatbot_answer_generator,
        input_file_path=input_file_path,
        output_folder_path="data/chatbot-answers/"
    )

    answer_factory.run()

    model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_MODEL_EVAL"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    evaluation_factory = EvaluationFactory(
        model_config=model_config,
    input_file=answer_factory.output_file_path,
        output_folder_base="data/eval_results/",
    )

    eval_result, df_rows, df_kpis = evaluation_factory.run_evaluation()

    visualizer = Visualizer(
        #eval_result=eval_result,
        df_rows=df_rows,
        #df_kpis=df_kpis,
        output_folder=evaluation_factory.output_folder_full
    )

    visualizer.visualize()

if __name__ == "__main__":
    main()

