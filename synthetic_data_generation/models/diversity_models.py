from pydantic import BaseModel

class LengthDistributionConfig(BaseModel):
    mean: float
    sigma: float
    shift: float

class DiversityInjection(BaseModel):
    response_length: int
    tone_injection: str
    disruptive_injection: str
    language_injection: str