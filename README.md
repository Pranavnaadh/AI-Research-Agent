# 🤖 AI Research Agent Dashboard

An intelligent dashboard that finds and summarizes AI-related GitHub repositories using
**semantic search (FAISS + OpenAI embeddings)** and **LLM reasoning**.

## 🚀 Features
- Summarizes repositories using GPT-4o-mini
- Builds FAISS vector index for semantic retrieval
- Provides contextual LLM explanations for any research query
- Interactive Streamlit dashboard

## 🧠 Tech Stack
- Python, OpenAI API, FAISS, NumPy
- Streamlit (UI)
- dotenv (for secure API keys)

## 🗂️ Project Structure
AI-RESEARCH-AGENT/
├── data/
├── src/
│ ├── clone_repos.py
│ ├── function_extractor.py
│ ├── research_agent_pipeline.py
│ └── app_streamlit.py
├── .env
├── .gitignore
└── requirements.txt

## 🧩 Setup
```bash
git clone https://github.com/<your-username>/AI-Research-Agent.git
cd AI-Research-Agent
python -m venv venv
venv\Scripts\activate      # (Windows)
pip install -r requirements.txt
