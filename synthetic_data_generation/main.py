from azure.search.documents import SearchClient
from openai import AzureOpenAI
import dotenv
dotenv.load_dotenv()
import os
from azure.core.credentials import AzureKeyCredential
from generators import SingleHopOneDocGenerator, MultiHopOneDocGenerator
from diversity.diversity_generator import DiversityGenerator
from models import LengthDistributionConfig
from qa_factory.qa_factory import QAFactory
from tracing.telemetry import setup_tracing

if __name__ == "__main__":

    # Initialize OpenTelemetry tracing
    setup_tracing(enable_console=False, enable_app_insights=True)

    index_name = "usa-expert-chatbots"
    search_client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),
    index_name=os.getenv("INDEX_NAME_OF_EXPERT_CHATBOT"),
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_SERVICE_KEY"))
    )

    llm_client = AzureOpenAI(
        api_version="2024-10-21",
        azure_endpoint =os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY")
    )

    diversity_generator = DiversityGenerator(
        LengthDistributionConfig(
            mean=2.0,
            sigma=0.8,
            shift=3
        )
    )

    generator_single_hop = SingleHopOneDocGenerator(
        diversity_generator=diversity_generator,
        search_client=search_client,
        llm_client=llm_client
    )

    generator_multi_hop = MultiHopOneDocGenerator(
        diversity_generator=diversity_generator,
        search_client=search_client,
        llm_client=llm_client
    )

    factory = QAFactory(
        generators=[
            (generator_single_hop, 0.5),
            (generator_multi_hop, 0.5)
        ],
        output_folder = "data/q-a"
    )

    qa, path = factory.generate(6)