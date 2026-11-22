import streamlit as st
import torch
import json
from model import TinyLanguageModel
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import tiktoken
import sys
import os
import requests

# Import the story model
sys.path.append(os.path.join(os.path.dirname(__file__), "story_model"))
from story_model import TinyStoriesModel, block_size as story_block_size

# ---------------------------------------------------------
# 1. PAGE CONFIG & CUSTOM CSS
# ---------------------------------------------------------
st.set_page_config(page_title="Raw AI Orbit", page_icon="🧠", layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

decorations_html = """
<div class="decoration-container">
<div class="petal"></div><div class="petal"></div><div class="petal"></div><div class="petal"></div><div class="petal"></div><div class="petal"></div>
<div class="flower">🌸</div><div class="flower">🌺</div><div class="flower">🌸</div><div class="flower">🌸</div>
</div>
"""
st.markdown(decorations_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. LOAD MODELS
# ---------------------------------------------------------
with open("vocab.json", "r") as file:
    word_to_id = json.load(file)
word_to_id = {word: int(token_id) for word, token_id in word_to_id.items()}
id_to_word = {token_id: word for word, token_id in word_to_id.items()}

model = TinyLanguageModel()
model.load_state_dict(torch.load("tiny_llm.pth", map_location="cpu"))
model.eval()

@st.cache_resource
def load_story_model():
    model = TinyStoriesModel()
    model.load_state_dict(torch.load("story_model/story_model.pth", map_location="cpu"))
    model.eval()
    return model

story_model = load_story_model()
story_tokenizer = tiktoken.get_encoding("gpt2")

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('local_models/all-MiniLM-L6-v2')
embed_model = load_embedding_model()

# ---------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------
def generate_text(text, temperature, max_tokens):
    words = text.split()
    tokens = []
    for word in words:
        if word not in word_to_id:
            return "Unknown word: " + word
        tokens.append(word_to_id[word])

    tokens = torch.tensor([tokens])
    eos_token_id = word_to_id.get("<EOS>", -1)

    for i in range(max_tokens):
        input_tokens = tokens[:, -4:]
        with torch.no_grad():
            logits = model(input_tokens)
        
        last_logits = logits[:, -1, :]
        probabilities = torch.softmax(last_logits / temperature, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

        if next_token.item() == eos_token_id:
            break

    generated_words = []
    for token_id in tokens[0]:
        word = id_to_word[token_id.item()]
        if word != "<EOS>":
            generated_words.append(word)

    return " ".join(generated_words)

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# ---------------------------------------------------------
# 4. UI LAYOUT (Notice the new tab order!)
# ---------------------------------------------------------
st.markdown("<h1>🧠 Raw AI Orbit</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-style: italic;'>A collection of AI models built from scratch and open-source architectures.</p>", unsafe_allow_html=True)

# Tab 3 is now Tab 4! Model Stats is at the end.
tab1, tab2, tab3, tab4 = st.tabs(["⚡ Tiny LLM", "📄 Document Search", "📖 Storyteller", "📊 Model Stats"])

# --- TAB 1: TINY LLM ---
with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Agent Configuration")
        known_words = [w for w in word_to_id.keys() if w != "<EOS>"]
        default_index = known_words.index("Python") if "Python" in known_words else 0
        
        selected_word = st.selectbox("Choose a starting word from our 256-word vocabulary:", options=known_words, index=default_index)
        text = st.text_input("Or type a custom prompt (space-separated known words):", value=selected_word)
    
    with col2:
        st.subheader("Parameters")
        temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.7, step=0.1)
        max_tokens = st.slider("Max Tokens", min_value=5, max_value=50, value=20, step=1)

    if st.button("Run Generation"):
        result = generate_text(text, temperature, max_tokens)
        st.subheader("Output")
        st.success(result)

# --- TAB 2: DOCUMENT SEARCH ---
with tab2:
    st.subheader("Document Engine")
    st.write("Upload a PDF or TXT file. The AI will use Vector Math to find the most relevant paragraphs.")
    
    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            doc_text = extract_text_from_pdf(uploaded_file)
        else:
            doc_text = uploaded_file.read().decode("utf-8")

        # FIXED CHUNKING HERE
        raw_lines = [line.strip() for line in doc_text.split('\n') if len(line.strip()) > 20]
        chunks = []
        current_chunk = ""
        for line in raw_lines:
            if len(current_chunk) + len(line) < 500:
                current_chunk += line + " "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = line + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        if len(chunks) == 0:
            st.warning("Could not extract enough text from this document.")
        else:
            st.info(f"Document loaded! Split into {len(chunks)} logical paragraphs.")
            
            with st.spinner("Processing vectors..."):
                chunk_embeddings = embed_model.encode(chunks)

            query = st.text_input("Ask a question about the document:")
            
            if query:
                with st.spinner("Searching..."):
                    query_embedding = embed_model.encode([query])[0]
                    similarities = np.dot(chunk_embeddings, query_embedding) / (
                        np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
                    )
                    top_3_indices = np.argsort(similarities)[-3:][::-1]

                st.subheader("Top 3 Matches Found:")
                
                for i, idx in enumerate(top_3_indices):
                    score = similarities[idx]
                    chunk_text = chunks[idx]
                    
                    if score > 0.2:
                        with st.expander(f"Match #{i+1} (Similarity Score: {score:.3f})", expanded=(i==0)):
                            st.write(chunk_text)
                    else:
                        st.warning("The similarity score is too low. No relevant text found.")
                        break
# --- TAB 3: STORYTELLER ---
with tab3:
    st.subheader("📖 TinyStories 13.7M Model")
    st.write("A 13.7M parameter Transformer trained from scratch on 10,000 stories.")
    
    st.markdown("---")
    st.markdown("### Step 1: Choose how you want to start your story")
    
    start_mode = st.radio(
        "Select a starting method:",
        ("📚 Pick from all 10,000 stories", "🎲 Random from Dataset", "✍️ Write my own"),
        horizontal=True
    )
    
    prompt = ""
    
    if start_mode == "📚 Pick from all 10,000 stories":
        @st.cache_data
        def get_story_dataframe():
            file_path = "story_model/tiny_stories.parquet"
            url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00000-of-00004-2d5a1467fff1081b.parquet"
            if not os.path.exists(file_path):
                with st.spinner("Downloading TinyStories dataset for the first time..."):
                    response = requests.get(url, stream=True)
                    os.makedirs("story_model", exist_ok=True)
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=1024*1024):
                            if chunk:
                                f.write(chunk)
            df = pd.read_parquet(file_path)
            # ONLY GRAB THE FIRST 500 STORIES TO PREVENT LAG
            df = df.head(500) 
            df["prompt"] = df["text"].apply(lambda x: x.split(".")[0] + ".")
            return df["prompt"].tolist()
        
        all_prompts = get_story_dataframe()
        
        selected_row = st.dataframe(
            pd.DataFrame(all_prompts, columns=["Story Starters"]),
            width="stretch",
            height=200, 
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        if len(selected_row.selection.rows) > 0:
            selected_index = selected_row.selection.rows[0]
            prompt = all_prompts[selected_index]
            st.info(f"**Selected Starter:** *{prompt}*")

    elif start_mode == "🎲 Random from Dataset":
        if st.button("🎲 Pick a Random Story Starter"):
            file_path = "story_model/tiny_stories.parquet"
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                random_story = df.sample(1)["text"].values[0]
                first_sentence = random_story.split(".")[0] + "."
                st.session_state["random_prompt"] = first_sentence
            else:
                st.error("Dataset not downloaded yet. Please click the 'Pick from all 10,000 stories' tab first to trigger the download.")
            
        if "random_prompt" in st.session_state:
            prompt = st.session_state["random_prompt"]
            st.info(f"**Selected Starter:** *{prompt}*")

    else:
        prompt = st.text_input("Enter your starting text:", value="Once upon a time")

    st.markdown("### Step 2: Adjust the creativity")
    creativity = st.slider(
        "How creative should the story be?", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.4,
        step=0.1,
        help="Lower = more predictable. Higher = more creative."
    )
    
    st.markdown("---")
    
    if prompt and not (start_mode == "🎲 Random from Dataset" and "random_prompt" not in st.session_state):
        if st.button("✨ Generate Story", width="stretch"):
            with st.spinner("Writing your story..."):
                tokens = story_tokenizer.encode(prompt)
                tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
                
                for _ in range(100):
                    tokens_cond = tokens[:, -story_block_size:]
                    with torch.no_grad():
                        logits, _ = story_model(tokens_cond)
                    
                    logits = logits[:, -1, :] / creativity
                    
                    recent_tokens = tokens[0, -20:].tolist()
                    for token_id in set(recent_tokens):
                        logits[0, token_id] -= 2.0 
                    
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    tokens = torch.cat((tokens, next_token), dim=1)
                    
                    if next_token.item() == 50256:
                        break
                
                story_text = story_tokenizer.decode(tokens[0].tolist())
                story_text = story_text.replace("<|endoftext|>", "")
                
                if "." in story_text:
                    last_period_index = story_text.rfind(".")
                    story_text = story_text[:last_period_index + 1]
                
                st.success(story_text)
# --- TAB 4: MODEL STATS (Now covers ALL models) ---
with tab4:
    st.subheader("Architecture & Hyperparameters")
    st.write("A breakdown of the math and architecture behind all three engines in this app.")
    
    st.markdown("### 1. Tiny LLM (From Scratch)")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric(label="Vocabulary Size", value=len(word_to_id), delta="Tokens")
    col_b.metric(label="Embedding Dim", value=8, delta="d_model")
    col_c.metric(label="Context Length", value=4, delta="Tokens")
    
    col_d, col_e, col_f = st.columns(3)
    col_d.metric(label="Transformer Blocks", value=1, delta="Layers")
    col_e.metric(label="Attention Heads", value=2, delta="Heads")
    col_f.metric(label="Model Parameters", value="~2,500", delta="Weights")

    st.markdown("---")
    
    st.markdown("### 2. Document Search (Vector Math)")
    st.markdown("""
    * **Architecture:** `all-MiniLM-L6-v2` (Pre-trained Transformer)
    * **Embedding Dimension:** 384
    * **Algorithm:** Cosine Similarity (Dot product of normalized vectors)
    * **Purpose:** Demonstrates how RAG (Retrieval-Augmented Generation) finds relevant context without using generative AI.
    """)

    st.markdown("---")

    st.markdown("### 3. TinyStories Storyteller (From Scratch)")
    col_g, col_h, col_i = st.columns(3)
    col_g.metric(label="Vocabulary Size", value="50,257", delta="BPE Tokens")
    col_h.metric(label="Embedding Dim", value=128, delta="d_model")
    col_i.metric(label="Context Length", value=128, delta="Tokens")
    
    col_j, col_k, col_l = st.columns(3)
    col_j.metric(label="Transformer Blocks", value=4, delta="Layers")
    col_k.metric(label="Attention Heads", value=4, delta="Heads")
    col_l.metric(label="Model Parameters", value="13.7M", delta="Weights")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("**Note:** The Tiny LLM and TinyStories models were trained completely from scratch on a CPU. The Document Search uses a pre-trained model to demonstrate vector embeddings.")