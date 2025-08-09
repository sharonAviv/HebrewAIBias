# NLP Final Project: Refusal Probability Analysis in Multiple Choice Questions

This project investigates refusal behavior in language models when presented with multiple choice questions across different languages (English and Hebrew). The study measures how likely models are to refuse answering questions and analyzes the probability distributions when a refusal option is available.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [Supported Models](#supported-models)
- [Datasets](#datasets)
- [Experiment Types](#experiment-types)
- [Results](#results)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

This research explores two main questions:
1. How do language models behave when given standard multiple choice questions?
2. How does the presence of a "refusal" option affect model responses and probability distributions?

The project implements a comprehensive evaluation framework that:
- Loads various language models (OpenAI API and local Hugging Face models)
- Processes multilingual datasets (English and Hebrew)
- Measures choice probabilities using logprobs
- Calculates refusal probabilities when a refusal option is present
- Provides detailed statistical analysis and logging

## 📁 Project Structure

```
NLP_final_project/
├── 1_data/                         # Data processing and datasets
│   ├── converted_questions.json    # English dataset (processed)
│   ├── translated_questions.json   # Hebrew dataset
│   ├── fix_json_format.py          # Survey data converter
│   ├── group_questions.py          # Semantic grouping
│   ├── populate_distributions.py   # Distribution filling
│   ├── translate_questions.py      # Hebrew translation
│   ├── wd_compare.py              # Wasserstein distance analysis
│   └── [other data processing files]
├── 2_train_eval/                   # Model evaluation and experiments
│   ├── run_refusal_experiment.py   # Main experiment runner
│   ├── config_refusal.yaml         # Configuration file
│   ├── utils.py                    # Utility functions
│   ├── models.py                   # Model loading and management
│   ├── refusal_utils.py            # Refusal-specific utilities
│   └── keys.yaml.example           # API keys template
└── results/                        # Experiment outputs
```

## ✨ Features

### Multi-Language Support
- **English**: Standard refusal text "I prefer not to answer"
- **Hebrew**: Localized refusal text "אני מעדיף לא לענות"
- Automatic encoding detection for Hebrew datasets

### Robust Model Management
- Supports both OpenAI API models and local Hugging Face models
- Automatic model loading/unloading with GPU memory management
- Rate limiting for API models
- CUDA cache clearing for local models

### Comprehensive Experiment Framework
- Two experiment variants: `no_refusal` (baseline) and `with_refusal`
- Probability extraction using logprobs
- Statistical analysis and detailed logging
- Configurable sampling for debug mode

### Data Validation
- Robust handling of malformed data entries
- Encoding detection for multilingual datasets
- Graceful error handling and reporting

## 🚀 Setup

### Prerequisites
```bash
pip install pandas torch transformers langchain_core pyyaml numpy pyreadstat deep-translator sentence-transformers scipy
```

### API Keys Setup
1. Copy the example keys file:
```bash
cp 2_train_eval/keys.yaml.example 2_train_eval/keys.yaml
```

2. Add your API keys to `keys.yaml`:
```yaml
OPENAI_API_KEY: "your-openai-api-key-here"
# Add other API keys as needed
```

### Local Models Setup
For local models, ensure you have sufficient GPU memory and the required model repositories downloaded or accessible via Hugging Face.

## 📖 Usage

### Running Refusal Experiments

#### Basic Usage
Run experiments with default configuration (both English and Hebrew):
```bash
python 2_train_eval/run_refusal_experiment.py
```

#### Custom Configuration
```bash
python 2_train_eval/run_refusal_experiment.py --config_file path/to/your/config.yaml --keys_file path/to/your/keys.yaml
```

#### Hebrew-Only Experiments
Create a custom config file with only Hebrew dataset:
```yaml
datasets:
  - data_file: "1_data/translated_questions.json"
    language: "hebrew"
```

#### Command Line Options
- `--config_file`: Path to configuration YAML file (default: `./2_train_eval/config_refusal.yaml`)
- `--keys_file`: Path to API keys YAML file
- `--seed`: Random seed for reproducibility
- `--variant`: Experiment variant (`no_refusal`, `with_refusal`, or `both`)

## ⚙️ Configuration

### Main Configuration (`config_refusal.yaml`)
```yaml
debug: true                    # Enable debug mode with limited samples
prompt_type: "zero_shot"       # Prompting strategy
seed: 42                       # Random seed
logs_dir: "results"            # Output directory

# Multiple datasets
datasets:
  - data_file: "1_data/converted_questions.json"
    language: "english"
  - data_file: "1_data/translated_questions.json"
    language: "hebrew"

# Model configurations
models:
  - model: "gpt-4o-mini"
    temperature: 0.8
    use_rate_limiter: true
    requests_per_second: 0.8
    num_samples: 5
  - model: "mistral-7b"
    local: "mistralai/Mistral-7B-Instruct-v0.2"
    temperature: 0.7
    use_rate_limiter: false
    num_samples: 5
```

## 🤖 Supported Models

### OpenAI Models
- GPT-4o-mini
- GPT-4
- GPT-3.5-turbo
- Any OpenAI API compatible model

### Local Models (via Hugging Face)
- Mistral-7B-Instruct-v0.2
- Mistral-7B-Instruct-v0.3  
- Llama-2-7b-chat-hf
- Zephyr-7b-beta
- Any compatible Hugging Face model

## 📊 Datasets

### English Dataset (`converted_questions.json`)
- Multiple choice questions in English
- Format: `{"id": int, "question": str, "answers": {"1": str, "2": str, ...}, "distribution": {...}}`
- Processed from original survey data

### Hebrew Dataset (`translated_questions.json`)
- Translated multiple choice questions in Hebrew
- Same format as English dataset
- Automatic encoding detection handles various Hebrew text encodings

## 🧪 Experiment Types

### No Refusal Variant (`no_refusal`)
- Baseline experiment with only original answer choices
- Measures standard multiple choice performance
- Calculates probability distribution across available options

### With Refusal Variant (`with_refusal`)
- Adds language-appropriate refusal option to choices
- Calculates refusal probability: `P(refusal) = exp(logprob_refusal) / Σ(exp(logprob_i))`
- Analyzes impact of refusal option on choice distributions

## 📈 Results

Experiments generate the following outputs:

### Directory Structure
```
results/
└── refusal_{variant}_{language}_{model}_tmp{temp}_seed{seed}/
    ├── config.yaml           # Experiment configuration
    └── refusal_results.json   # Detailed results
```

### Metrics Tracked
- **Choice Probabilities**: Normalized probability for each option
- **Refusal Probability**: Likelihood of selecting refusal option
- **Selection Accuracy**: Most likely choice based on probabilities
- **Statistical Summaries**: Mean, median, max refusal probabilities

### Log Output
- Real-time processing updates
- Model loading/unloading status
- Memory management information
- Statistical summaries per experiment
- Top questions with highest refusal probabilities

## 🔄 Data Processing Pipeline

The project includes a comprehensive data processing pipeline in the `1_data/` directory:

### Core Data Files
- **converted_questions.json**: Master JSON with all questions, answers, and distributions
- **translated_questions.json**: Hebrew version of the dataset
- **question_groups.json**: Semantic groupings of similar questions
- **topic_to_question.json**: Topic-to-question ID mappings

### Processing Scripts
- **fix_json_format.py**: Converts raw survey JSONs to standardized question format
- **group_questions.py**: Groups questions by semantic similarity using embeddings
- **populate_distributions.py**: Fills missing distribution fields using .sav survey files
- **translate_questions.py**: Translates questions and answers to Hebrew
- **wd_compare.py**: Computes Wasserstein distance between question sets with refusal diagnostics

### Dependencies for Data Processing
- pandas, pyreadstat, deep-translator, sentence-transformers, torch, numpy, scipy

## 🔧 Troubleshooting

### Encoding Issues (Hebrew Dataset)
The system automatically tries multiple encodings:
- UTF-8, UTF-8-BOM
- Windows-1255, CP1255 (Hebrew)
- ISO-8859-8, CP862
- UTF-16 variants

### Memory Issues (Local Models)
- Models are automatically unloaded after each experiment
- CUDA cache is cleared between models
- Use debug mode (`debug: true`) for testing with smaller samples

### Data Validation Errors
The system automatically:
- Skips rows with malformed answer data
- Reports number of skipped rows
- Validates question and answer format

### Common Issues
1. **Missing API Keys**: Ensure `keys.yaml` is properly configured
2. **GPU Memory**: Reduce model size or enable debug mode
3. **Rate Limits**: Adjust `requests_per_second` for API models
4. **Data Format**: Ensure JSON files have proper structure
5. **Hebrew Text**: System handles encoding automatically

## 📝 Citation

```bibtex
@misc{nlp_refusal_analysis,
  title={Refusal Probability Analysis in Multiple Choice Questions},
  author={[Your Name]},
  year={2025},
  note={NLP Final Project}
}
```

## 🤝 Contributing

This is an academic research project. For questions or issues, please refer to the course materials or contact the project maintainers.

---

**Note**: This project is designed for research and educational purposes. Ensure compliance with API usage policies and ethical guidelines when working with language models.
