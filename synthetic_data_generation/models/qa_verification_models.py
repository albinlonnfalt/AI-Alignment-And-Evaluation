from pydantic import BaseModel

class QAVerificationResult(BaseModel):
    is_correct: bool
    reason: str