# Tiny Transformer Language Model

A small autoregressive Transformer language model built from scratch using Python and PyTorch. The project demonstrates how language models learn text patterns and generate new text through next-token prediction.

## Features

- Custom word-level tokenizer
- Token and positional embeddings
- Multi-head self-attention
- Causal attention masking
- Feed-forward neural network
- Model training with backpropagation
- Next-token text generation
- Temperature-based sampling
- Interactive Streamlit interface

## Model Architecture

| Component | Configuration |
|---|---|
| Vocabulary size | 256 tokens |
| Embedding dimension | 8 |
| Context length | 4 tokens |
| Transformer blocks | 1 |
| Attention heads | 2 |
| Feed-forward hidden size | 32 |
| Training steps | 2,000 |
| Dataset | Custom text dataset |

## Tech Stack

Python, PyTorch, Streamlit

## Project Structure

```text
tiny_llm/
├── app.py
├── model.py
├── transformer_block.py
├── tokenizer.py
├── dataset.py
├── dataset.txt
├── vocab.json
├── tiny_llm.pth
├── train_step.py
├── generate.py
├── requirements.txt
└── README.md
```

## Run Locally

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd tiny_llm
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Text Generation

Enter a starting prompt and adjust the temperature to control the randomness of generated text. The model predicts one token at a time using its learned parameters.

## Project Scope

This is an educational, small-scale language model designed to demonstrate the fundamentals of Transformer architecture and autoregressive text generation. It is not a large-scale or instruction-tuned LLM.