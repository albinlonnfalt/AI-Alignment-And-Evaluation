# Evaluation System Documentation

## Overview

This system evaluates chatbot responses using Azure AI Evaluation framework and custom evaluators.

**Components:**
- **EvaluationFactory**: Main orchestrator
- **Built-in Evaluators**: Relevance, Groundedness (Azure AI)
- **Custom Evaluators**: Citation validation
- **Alignment System**: Quality assurance for evaluators

**Current Evaluators:**
1. **Relevance**: Response relevance to question
2. **Groundedness**: Response grounding in provided context
3. **Citation**: Validates that organization-specific information includes citations `[citationIndex-#]`

## Usage

```python
from evaluation.evaluation_factory import EvaluationFactory
from azure.ai.evaluation import AzureOpenAIModelConfiguration

model_config = AzureOpenAIModelConfiguration(
    azure_endpoint="your-endpoint", 
    api_key="your-key",
    azure_deployment="your-deployment", 
    api_version="your-version"
)

evaluator = EvaluationFactory(
    model_config=model_config, 
    output_folder_base="./results",
    input_file="path/to/your/data.jsonl"
)
results = evaluator.run_evaluation()
```

**Input Format (JSONL):**
```json
{
    "question": "User's question",
    "chatbot_answer": "Chatbot's response", 
    "context": "Retrieved context",
    "ground_truth_answer": "Expected answer (optional)"
}
```

## Adding New Evaluators

### 1. Create Evaluator Class
```python
from pydantic import BaseModel

class YourEvaluationResponse(BaseModel):
    score: float  # or bool
    reason: str

class YourEvaluator:
    def __init__(self, model_config_dict: dict):
        self.model_config = model_config_dict
    
    def __call__(self, question: str, chatbot_answer: str, **kwargs):
        # Your evaluation logic
        return {"score": your_score, "reason": your_reasoning}
```

### 2. Update EvaluationFactory
```python
# Add to evaluators dict
"your_metric": YourEvaluator(model_config_dict={...})

# Add column mapping
"your_metric": {
    "column_mapping": {
        "question": "${data.question}",
    "chatbot_answer": "${data.chatbot_answer}"
    }
}
```

### 3. Create Alignment Data
Follow pattern in `citation_evaluator/alignment_data_citation.jsonl`

## Evaluator Alignment & Quality Assurance

**Why Important:** Ensures evaluators align with human judgment for reliability and trust.

### Process

**1. Create Alignment Dataset**
```json
{"human_label": true, "question": "...", "chatbot_answer": "..."}
```
Include diverse examples with edge cases.

**2. Run Alignment**
See an example `citation_evaluator/alignment.py`

**3. Analyze Results**
- **Cohen's Kappa**: Agreement measure
  - κ ≥ 0.60: Minimum acceptable
  - κ ≥ 0.80: Production target
- **Confusion Matrix**: Visual agreement analysis
- **Misaligned Cases**: Examples for improvement

**4. Iterate**
Analyze misaligned cases → Refine prompts → Add examples → Re-evaluate

### Implementation
See `citation_evaluator/alignment.py` for complete example.

## Configuration

**Environment Variables:**
```bash
AZURE_OPENAI_API_VERSION=your_version
AZURE_OPENAI_ENDPOINT=your_endpoint  
AZURE_OPENAI_API_KEY=your_key
```

**Dependencies:**
`azure-ai-evaluation`, `openai`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

**Output:**
Results saved in timestamped folders with individual scores, aggregate metrics, and detailed logs.