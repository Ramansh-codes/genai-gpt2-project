# Generative AI Course Project (UCS748)

**Name:** Ramansh


## Project Overview
This project implements a GPT-2 style Large Language Model from scratch using PyTorch. It covers the complete pipeline from data preprocessing and tokenization to model architecture definition, text generation, and fine-tuning. 

Additionally, it includes a demonstration of **Prompt Engineering** using the Chain-of-Thought (CoT) technique.

## Repository Structure
* `data/`: Contains the raw text corpus (`the-verdict.txt`) and instruction datasets.
* `src/gpt2_from_scratch/`: Core implementation of the Transformer architecture, attention mechanisms, and tokenizer.
* `scripts/`: Executable scripts for data processing, model generation, and fine-tuning.
* `prompts/`: Contains the Chain-of-Thought prompt engineering demonstration.
* `results/`: Output logs and generated text from our experiments.

## How to Run

### 1. Setup
Create a virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
