# Evaluation

This evaluation system is designed to assess the performance of the an AI assistant by generating answers to questions and evaluating their quality using Azure AI evaluation tools.

## Quick Start

To run the complete evaluation pipeline:

```bash
python evaluation_pipeline.py
```

## System Components

The evaluation system consists of four main components:

### 1. Answer Generation (`answer_generation/`)

This module is responsible for generating answers from the assistant.

- **`auth_service.py`** - Handles authentication for accessing the chatbot API
- **`chat_session_initializer.py`** - Sets up and manages chat sessions
- **`answer_generator.py`** - Core component that queries the chatbot API with questions
- **`answer_factory.py`** - Factory class that orchestrates the answer generation process

**Purpose**: Takes question-answer pairs from the input data and generates responses from the chatbot for evaluation.

### 2. Evaluation (`evaluation/`)

This module evaluates the quality of generated answers using Azure AI evaluation metrics.

- **`evaluation_factory.py`** - Main factory class that runs various evaluation metrics
- **`citation_evaluator/`** - Specialized evaluators for assessing citation quality and accuracy

**Purpose**: Analyzes the generated answers using metrics like:
- Relevance
- Coherence
- Groundedness
- Citation accuracy
- Response quality

### 3. Visualization (`visualizer/`)

This module creates visual reports and analytics from the evaluation results.

- **`visualizer.py`** - Generates charts, graphs, and reports from evaluation data

**Purpose**: Transforms evaluation metrics into visual insights including:
- Performance dashboards
- Metric comparisons
- Trend analysis
- Quality score distributions

### 4. Main Pipeline (`evaluation_pipeline.py`)

The main orchestrator that ties all components together in a complete evaluation workflow:

1. **Answer Generation**: Generates chatbot responses for input questions
2. **Evaluation**: Runs quality assessments on the generated answers
3. **Visualization**: Creates visual reports of the results

## Data Flow

1. **Input**: Question-answer pairs from `data/q-a/` directory
2. **Answer Generation**: Chatbots responses saved to `data/chatbot-answers/`
3. **Evaluation**: Results stored in `data/eval_results/`
4. **Visualization**: Charts and reports generated in the evaluation results folder

## Prerequisites

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env` file:
```
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_CHAT_MODEL_EVAL=your_model_deployment
AZURE_OPENAI_API_VERSION=your_api_version
```

3. Ensure the chatbot API is running at `https://localhost:40443`

## Configuration

The evaluation pipeline can be customized by modifying:
- Input file path in `evaluation_pipeline.py`
- Output folder paths
- Evaluation metrics in the evaluation factory

## Output

The system generates:
- **Answer files**: JSONL format with chatbot responses
- **Evaluation results**: Detailed metrics and scores
- **Visualizations**: Charts and dashboards showing performance insights

## Usage Examples

### Running Individual Components

```python
# Generate answers only
from answer_generation import AnswerFactory
factory = AnswerFactory(...)
factory.run()

# Evaluate existing answers
from evaluation.evaluation_factory import EvaluationFactory
evaluator = EvaluationFactory(...)
results = evaluator.run_evaluation()

# Create visualizations
from visualizer.visualizer import Visualizer
viz = Visualizer(...)
viz.visualize()
```

## Troubleshooting

- Ensure all environment variables are properly set
- Verify the chatbot API is accessible
- Check that input data files exist in the expected format
- Review log outputs for detailed error information