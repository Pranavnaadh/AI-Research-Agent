import streamlit as st
import json, faiss, numpy as np
from openai import OpenAI

st.set_page_config(page_title="AI Research Agent", page_icon="🧠", layout="centered")

# --- Debug markers ---
st.write("Starting Streamlit app...")

try:
    api_key = "your-api-key"  # replace with your actual key
    client = OpenAI(api_key=api_key)
    st.success("OpenAI client initialized successfully.")
except Exception as e:
    st.error(f" OpenAI client initialization failed: {e}")
    st.stop()

try:
    st.write(" Loading FAISS index and metadata files...")
    index = faiss.read_index("data/vector_index.faiss")
    with open("data/repo_names.json") as f:
        repo_names = json.load(f)
    with open("data/repo_summaries.json") as f:
        summaries = json.load(f)
    st.success(" All data files loaded successfully.")
except Exception as e:
    st.error(f" Error loading FAISS or data files: {e}")
    st.stop()

st.title("AI Research Agent Dashboard")
st.write("Find and understand AI-related repositories using semantic search and LLM reasoning.")

query = st.text_input("🔍 Enter a research topic:", placeholder="e.g. Reinforcement Learning, LLM pipelines")

if query:
    try:
        st.info("Generating embeddings for your query...")
        q_emb = client.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        ).data[0].embedding

        st.info("Searching FAISS index...")
        D, I = index.search(np.array([q_emb], dtype=np.float32), 3)
        results = [repo_names[i] for i in I[0]]

        st.subheader("Top Matching Repositories:")
        for r in results:
            st.markdown(f"**{r}** — {summaries.get(r, 'No summary available.')}")

        # --- Generate explanation using GPT ---
        context = "\n\n".join([f"{r}: {summaries.get(r)}" for r in results])
        prompt = f"""
        You are an AI research assistant. Given these repositories and their summaries,
        explain how they relate to the query: '{query}'.
        Focus on their shared AI/ML ideas in 3–4 sentences.

        Repositories:
        {context}
        """
        st.info(" Generating explanation with LLM...")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        explanation = response.choices[0].message.content.strip() # type: ignore
        st.subheader(" LLM Explanation:")
        st.write(explanation)

    except Exception as e:
        st.error(f" Runtime error while processing query: {e}")
