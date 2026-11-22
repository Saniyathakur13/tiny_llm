
# 🧠 Raw AI Orbit: From-Scratch Transformers & Vector Search

A dual-model AI application built entirely from scratch using PyTorch and Streamlit. No OpenAI APIs, no pre-trained language weights—just raw math, custom architecture, and custom training loops running on a CPU.

## 🚀 What's Inside?

This app features four distinct tabs, showcasing different areas of Machine Learning and NLP:

1. **⚡ Tiny LLM (2.5K Params):** A microscopic language model built to demonstrate the absolute fundamentals of Transformer architecture (embeddings, self-attention, causal masking).
2. **📄 Document Engine:** A search engine featuring *two* modes:
   * **AI Embeddings:** Uses a pre-trained `SentenceTransformer` (384 dimensions) and Cosine Similarity to find paragraphs by meaning.
   * **TF-IDF (From Scratch):** A pure Python implementation of Term Frequency-Inverse Document Frequency math to find exact keyword matches.
3. **📖 Storyteller (13.7M Params):** A highly capable language model trained from scratch on 10,000 TinyStories. It uses a custom BPE tokenizer and generates coherent, grammatical children's stories.
4. **📊 Model Stats:** A transparent look under the hood at the hyperparameters, parameter counts, and math behind all three engines.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch (CPU optimized)
* **Tokenization:** Tiktoken (GPT-2 BPE) & Custom Python BPE
* **NLP:** SentenceTransformers, Scikit-learn (math equivalents)
* **Frontend:** Streamlit, Custom CSS (Glassmorphism UI)

## 🧮 The "From Scratch" Journey

### 1. The 13.7M Parameter Storyteller
Unlike standard tutorials that just load a model, this project trains a Transformer from absolute zero.
* **Dataset:** 10,000 stories from the TinyStories dataset (~2.1 Million tokens).
* **Architecture:** 4 Transformer Blocks, 4 Attention Heads, 128 Embedding Dimensions, 128 Context Window.
* **Training:** Custom PyTorch training loop with AdamW optimizer, Cross-Entropy loss, and a custom repetition penalty during inference.
* **Result:** Loss dropped from `11.0` (random guessing) to `2.7` (coherent English).

### 2. The Dual-Search Engine
* **Vector Search:** Converts text chunks into 384-dimensional dense vectors and calculates the dot product/angle between them.
* **TF-IDF:** Built 100% from scratch. It tokenizes text, removes stop words, calculates logarithmic IDF weights, and computes cosine similarity on sparse vectors.

## 📁 Project Structure

```text
tiny_llm/
├── app.py                     # Main Streamlit UI (4 tabs)
├── style.css                  # Custom Glassmorphism CSS
├── requirements.txt           # Dependencies
│
├── model.py                   # 2.5K parameter Transformer
├── tiny_llm.pth               # Weights for tiny model
│
├── local_models/              # Local cache for SentenceTransformer
│
├── tfidf_search.py            # From-scratch TF-IDF math engine
│
└── story_model/               # 13.7M Parameter Model Folder
    ├── story_model.py         # The 13.7M Transformer architecture
    ├── story_model.pth        # Trained weights (~50MB)
    ├── train_story.py         # Training loop (CPU optimized)
    ├── generate_story.py      # CLI inference script
    ├── bpe_tokenizer.py       # Custom BPE implementation (educational)
    └── prepare_data.py        # Data pipeline
```

## 💻 Run Locally

Clone the repository:
```bash
git clone https://github.com/Saniyathakur13/tiny_llm.git
cd tiny_llm
```

Create and activate a virtual environment:
```powershell
python -m venv .venv
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

## ⚠️ Hardware & Limitations
The 13.7M Storyteller model was trained on an **AMD Ryzen 5 CPU** (no GPU). While it generates grammatically correct sentences, it suffers from the context-window limits and slight repetition loops typical of small-scale models. The goal of this project is to demonstrate the end-to-end ML engineering process, not to compete with GPT-4.
```

***

### What to do right now:
1. Open `README.md` in VS Code.
2. Delete everything.
3. Paste the code above.
4. Save the file.
5. Run these three commands in your terminal to update GitHub:
   ```powershell
   git add README.md
   git commit -m "Updated README with 13M model and TF-IDF details"
   git push
   ```
