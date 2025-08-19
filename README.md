# AI Assistant Alignment & Evaluation

Welcome to the AI Assistant Alignment and Evaluation project! This repository is designed to accelerate your evaluation and alignment efforts for AI Assistants. It includes best practices and a user-friendly framework that you can tailor to your specific needs. **This methodology has been successfully used to align AI Assistants deployed to millions of consumers**. Please note that the content in this repository is entirely fictional.

Note: This repository is designed to be 'bring your own assistant'. The methodology provided here serves as a wrapper around your existing assistant implementation.

## Solution Overview
The methodology is built on two crucial pillars: synthetic data generation and automatic evaluators. Our field experience indicates that both synthetic data and automatic evaluators are often overlooked and underutilized. This is primarily due to a lack of understanding of the underlying mathematical properties of generative AI models and the challenges associated with effectively implementing these components. This repository aims to accelerate and empower developers to harness the potential of these components by compiling key learnings from our work with companies that have launched AI solutions to millions of users.

![Solution Overview](media/img/solution_overview.png)

## 🧰 Key Technologies

- Azure OpenAI (chat, embeddings, optional evaluation models) – can be substituted with OpenAI / other providers.
- Azure AI Evaluation SDK (orchestrates multi-metric runs; local or scaled execution).
- Retrieval / Search (Azure AI Search or custom) – pluggable via `search_service.py`.

## 🧪 Importance of Synthetic Data

Our field experience shows that synthetic data plays a crucial role in successfully aligning AI systems. Despite its importance, synthetic data is rarely used for AI evaluation and alignment. This underutilization is often due to a lack of understanding of its benefits and the challenges associated with its implementation.

### Importance of Sufficient Data Quantity

Having a sufficient quantity of data from a representative distribution is crucial for the effective alignment and evaluation of AI systems. Current AI models are unpredictable and sensitive to small changes in prompts. To align and evaluate reliably, it is important to sample extensively from the distribution of possible questions to be able make statistically significant conclusions. Additionally, many AI systems may have an asymmetric risk profile, meaning that misalignment can lead to significant consequences. For such solutions, it is crucial to scale the annotated data to achieve statistically significant results with an acceptable risk profile.

### Out of Distribution Behaviour

A common pattern we have observed is that the sampled questions used for alignment and evaluation do not represent the entire distribution of questions that will be asked to the AI assistant. Often, the dataset exclusively contains the most common topics, lengths, and phrasings. The image below illustrates the sampling cut-off from the tails of the distribution. (Please note that the image is purely illustrative and not representative of a real distribution.)

![Sampling Cutoff Illustration](media/img/sampling_from_distribution.png)

The behavior of AI systems for questions that fall outside the sampling distribution used for alignment and evaluation often results in unpredictable outcomes. Therefore, it is crucial to obtain samples from as large a portion of the distribution as possible. Synthetic data is an excellent tool for achieving a dataset that covers a broader range of possible questions, as it can be scaled indefinitely and eliminates human biases.

![Bad vs Good Sampling](media/img/bad_vs_good_data_sampling.png)

## 🤖 Importance of Automatic Evaluators

### Iterative Alignment

To rapidly iterate on prompts, configurations, or fine-tuned models, it is crucial to have a swift evaluation method to determine if the system has improved. Automatic evaluators can provide near-instant feedback, eliminating the need to rely on human domain experts for daily development tasks.

### Scale the Evaluation

To effectively cover a significant dataset, it is important to scale the evaluations. Scaling evaluations to a sufficient number of data points often requires automated evaluators that can either operate independently or assist in reducing the number of data points that need to be reviewed by a human.

## 🚀 Get Started

### Prerequisites

- Python 3.8+
- Azure OpenAI access
- Chatbot API access

### 0. Implement Pluggable Components (REQUIRED)

Before running any of the commands below you MUST wire up your own search + answer generation logic.

Implement these two files first:

1. `synthetic_data_generation/search/search_service.py`
   - Add your retrieval / search call (Azure AI Search, custom, etc.).
   - Map your index field names and return a normalized list of docs like `{ "id": str, "content": str, "source": str, "metadata": {...} }`.
2. `evals/answer_generation/answer_generator.py`
   - Call your chatbot / model endpoint for each question.
   - Return at least an `answer` string (optionally citations, raw payload, latency, etc.).

Quick smoke test before proceeding:
 - `python -c "from synthetic_data_generation.search.search_service import SearchService; print(SearchService().search('test'))"` returns docs
 - `python -c "from evals.answer_generation.answer_generator import AnswerGenerator; print(AnswerGenerator().generate_answer('Test question')['answer'])"` returns a non-empty answer

If those work, continue with Step 1.

### 1. Generate Synthetic Data

```bash
cd synthetic_data_generation
pip install -r requirements.txt
python main.py
```

### 2. Manual Vetting (Recommended)

Launch the annotation tool to manually review generated Q&A pairs:

```bash
cd synthetic_data_generation
streamlit run qa_annotator_app.py
```

### 3. Run Evaluation Pipeline

```bash
cd evals
pip install -r requirements.txt
python evaluation_pipeline.py
```

## 🔧 Components

### Synthetic Data Generation

The synthetic data generation system creates diverse, question-answer pairs for evaluation purposes.

**Components:**
- `generators/` - Question generation algorithms
- `diversity/` - Diversity optimization
- `qa_factory/` - Orchestration and quality control
- `search/` - Integration to the knowlege base of the chatbot
- `qa_annotator_app.py` - Manual vetting interface

### Evaluation Pipeline

The evaluation system assesses chatbot responses using multiple Azure AI metrics and custom evaluators.

**Key Features:**
- **Multi-metric evaluation**: Evaluate the answers based on several metrics
- **Automated pipeline**: End-to-end evaluation workflow
- **Visual reporting**: Charts and dashboards for result analysis

**Components:**
- `answer_generation/` - Chatbot response collection
- `evaluation/` - Azure AI evaluation metrics
- `visualizer/` - Result visualization and reporting

## 📊 Workflow

### Primary Evaluation Workflow

The intended workflow follows this sequence:

```
Synthetic Data Generation → Vetting → Run Evals → Iterate → Run Evals → ...
```

**Step-by-Step Process:**

1. **Generate Synthetic Data**
   ```bash
   python synthetic_data_generation/main.py
   ```
   Creates initial Q&A pairs for evaluation

2. **Manual Vetting** (Recommended for Quality)
   ```bash
   streamlit run qa_annotator_app.py
   ```
   - Review generated Q&A pairs
   - Approve high-quality examples
   - Reject or modify problematic cases
   - Build a "gold standard" dataset
   
   > **Note**: This step can be skipped if testing scale is critical and you need to run evaluations quickly. However, vetting significantly improves evaluation quality.

3. **Run Evaluation**
   ```bash
   python evals/evaluation_pipeline.py
   ```
   Execute evaluation pipeline using vetted (or unvetted) data

4. **Iterate & Improve**
   - Analyze results in `data/eval_results/`
   - Make improvements to the chatbot
   - Generate additional test cases if needed

5. **Re-run Evaluation**
   - Repeat evaluation with the same dataset to measure improvements
   - Continue the iterate → evaluate cycle

### Alternative: Scale Testing Workflow

For scenarios where testing scale is more critical than precision:

```
Synthetic Data Generation → Run Evals (Skip Vetting) → Iterate → Run Evals → ...
```

- Skip the vetting step to quickly test with large datasets
- Use for trend analysis and performance patterns
- Accept moderate quality trade-off for speed and scale

**Vetted Data (High Quality, Small Scale)**
- Manual review by SMEs
- High confidence in correctness
- Used for critical performance metrics
- Stored in `data/q-a-vetted/`

**Unvetted Data (Moderate Quality, Large Scale)**
- Generated automatically
- Good for scale testing and trend analysis
- Helps identify performance patterns
- Stored in `data/q-a/`

## 🛠️ Configuration

### Required Custom Implementations

You MUST adapt two pluggable components to your own environment before meaningful evaluations will work (these differ from any similarly named components you may have seen elsewhere):

1. Search Service Integration
   - File: `synthetic_data_generation/search/search_service.py`
   - Implement the query logic against YOUR Azure AI Search (or other) index.
   - Map field names to your index schema (e.g. content/body field, title, metadata tags).
   - Normalize returned documents to a structure like: `{ "id": str, "content": str, "source": str, "metadata": {...} }`.
   - Hide hybrid / vector specifics inside this layer so the rest of the pipeline stays unchanged.

2. Answer Generation
   - File: `evals/answer_generation/answer_generator.py`
   - Implement the function/class that calls your chatbot or model endpoint to produce an answer for each question.
   - Return at minimum an `answer` string; optionally include `citations`, `raw_response`, latency metrics, etc.
   - Add retries / timeout handling so single failures don't abort the batch.

Quick Verification Checklist:
   - [ ] `search_service.py` returns results for a sample query.
   - [ ] `answer_generator.py` produces an answer dict for one question.
   - [ ] `python evals/evaluation_pipeline.py` runs without key/schema errors.
   - [ ] Outputs appear under `data/chatbot-answers/`.

> Skipping these steps will lead to empty context retrieval or missing answers and the evaluators will yield low or meaningless scores.

### Environment Variables

Create a `.env` file in each component directory:

```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_key

# Azure AI Search
AZURE_SEARCH_SERVICE_ENDPOINT=your_search_endpoint
AZURE_SEARCH_SERVICE_KEY=your_search_key
INDEX_NAME_OF_EXPERT_CHATBOT=your_index_name

# Chatbot API
CHATBOT_API_ENDPOINT=your_chatbot_endpoint
CHATBOT_API_KEY=your_chatbot_key
```

## 📈 Output and Results

### Generated Data
- **Q&A Pairs**: JSON files with questions, expected answers, and metadata
- **Chatbot Responses**: JSONL files with chatbot responses to questions
- **Evaluation Results**: Comprehensive metrics and scores

---

**Note**: The quality of your evaluation is directly proportional to the quality of your synthetic data. It is recommended to perform manual vetting for critical evaluation datasets, and use unvetted data thoughtfully for scale testing.
