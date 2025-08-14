from pydantic import BaseModel
from .diversity_models import DiversityInjection

class QA(BaseModel):
    question: str
    ground_truth_answer: str
    chunk_ids: list[str]  # List of chunk IDs that the answer is based on

class QATagged(BaseModel):
    question: str
    ground_truth_answer: str
    tags: dict[str, str]  # Tags for the question, e.g., {"question_type": "factoid"}
    diversity_injection: DiversityInjection
    chunk_ids: list[str]  # List of chunk IDs that the answer is based on
    chunk_content: list[str]