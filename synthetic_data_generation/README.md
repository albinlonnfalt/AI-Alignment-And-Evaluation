# Synthetic Data Generation

This module generates synthetic question-answer (Q&A) pairs for evaluating an internal chatbot. It creates diverse, realistic questions based on documents in Azure AI Search and generates corresponding ground-truth answers for use in the evaluation pipeline.

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables** in `.env`:
   ```
   AZURE_OPENAI_ENDPOINT=your_endpoint
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_SEARCH_SERVICE_ENDPOINT=your_search_endpoint
   AZURE_SEARCH_SERVICE_KEY=your_search_key
   INDEX_NAME_OF_EXPERT_CHATBOT=your_index_name
   ```

3. **Generate Q&A pairs**:
   ```bash
   python synthetic_data_generation/main.py
   ```

4. **Review generated data** (recommended):
   ```bash
   streamlit run synthetic_data_generation/qa_annotator_app.py
   ```

## 📁 Module Structure

- **`main.py`** - Main entry point for Q&A generation
- **`qa_annotator_app.py`** - Streamlit web app for manual Q&A review and vetting
- **`generators/`** - Core Q&A generation algorithms
- **`diversity/`** - Tools for ensuring question diversity and length variation
- **`qa_factory/`** - Factory classes for orchestrating Q&A generation
- **`models/`** - Data models and configuration classes
- **`search/`** - Azure AI Search integration utilities
- **`tracing/`** - OpenTelemetry telemetry and monitoring

## 🔧 Key Features

### Question Generation Types
- **Single-hop**: Direct questions answerable from a single chunk
- **Multi-hop**: Questions where information for multiple chumks need to be combined to generate an answer

### Diversity & Quality
- **Length variation**: Configurable question length distributions
- **Content diversity**: Ensures broad coverage of different document types and topics
- **Quality control**: Built-in validation and filtering mechanisms

### Manual Review Tool
- **Web interface**: Streamlit-based annotation tool for reviewing generated Q&A pairs
- **Annotation workflow**: Pass/fail/skip options with comments
- **Export functionality**: Outputs vetted data to `data/q-a-vetted/` for evaluation

## 🔍 Why Vet Synthetic Data?

**Quality Assurance**: While synthetic generation creates diverse Q&A pairs at scale, manual vetting ensures:
- **Accuracy**: Verifies that generated answers are factually correct and relevant
- **Clarity**: Ensures questions are well-formed and unambiguous
- **Relevance**: Confirms questions align with actual user scenarios
- **Coverage**: Identifies gaps or biases in generated content

**Impact on Evaluation**:
- **Higher Confidence**: Vetted data provides more reliable evaluation metrics
- **Better Baselines**: Creates gold-standard datasets for benchmarking
- **Reduced Noise**: Eliminates poor-quality examples that could skew results
- **SME Validation**: Leverages subject matter expertise to improve data quality

**When to Vet**:
- ✅ **Small to medium datasets** (< 500 Q&A pairs)
- ✅ **Frequently reused dataset** If the dataset will be used frequently to for example apply metric driven development and iterative improvement 
- ❌ **Large-scale analysis** When evaluations need to be scaled to a large number of Q&A pairs, the cost of vetting synthetic data can be high. For these cases, a non-vetted dataset may be appropriate. Ensure you are confident that the quality of the synthetic data generator is sufficiently high.

**Vetting Workflow**: The annotation tool saves vetted Q&A pairs to `../data/q-a-vetted/` with approval status, reviewer comments, and timestamps for full traceability.

## 📊 Output

Generated Q&A pairs are saved to:
- **`../data/q-a/`** - Raw generated Q&A pairs
- **`../data/q-a-vetted/`** - Manually reviewed and approved pairs (after annotation)

Each Q&A pair includes:
- Question text
- Ground-truth answer
- Source document references
- Metadata (generation method, confidence scores, etc.)

## 🎯 Usage in Evaluation Pipeline

The synthetic data serves as input for the evaluation system in `../evals/`, providing:
1. **Test questions** for the chatbot
2. **Ground-truth answers** for quality assessment
3. **Diverse scenarios** to comprehensively test chatbot capabilities

## 🔍 Monitoring & Telemetry

The system includes OpenTelemetry integration for:
- Performance monitoring
- Generation metrics tracking
- Azure Application Insights integration
- Debugging and optimization insights