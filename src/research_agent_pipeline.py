# src/research_agent_pipeline.py
from openai import OpenAI
import json, os, numpy as np, faiss
from tqdm import tqdm

# ---------------------------------------------------------
# Initialize OpenAI client — Direct key (No dotenv)
# ---------------------------------------------------------
api_key = "your-api-key"  # replace with your actual key

if not api_key or not api_key.startswith("sk-"):
    raise ValueError(" Invalid or missing OpenAI API key.")

print(f" Using API key prefix: {api_key[:10]}...")
client = OpenAI(api_key=api_key)

# ---------------------------------------------------------
# Step 1: Load extracted function summaries
# ---------------------------------------------------------
os.makedirs("data", exist_ok=True)

with open("data/repo_code_summary.json", "r", encoding="utf-8") as f:
    code_data = json.load(f)

# ---------------------------------------------------------
# Step 2: Generate LLM Summaries
# ---------------------------------------------------------
def summarize_repo(repo_name, repo_content):
    repo_text = json.dumps(repo_content)[:4000]  # limit token size
    prompt = f"""
    You are an expert AI researcher. Summarize the purpose of this repository,
    based on its functions and classes. Respond in 2-3 sentences with key topics.

    Repository: {repo_name}
    Content: {repo_text}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    # Handle both dict & object responses safely
    msg = response.choices[0].message
    if isinstance(msg, dict):
        content = msg.get("content", "")
    else:
        content = getattr(msg, "content", "")
    return str(content).strip()

# ---------------------------------------------------------
# Step 3: Generate and Save Summaries
# ---------------------------------------------------------
summaries = {}
for repo, content in tqdm(code_data.items(), desc="Generating summaries"):
    try:
        summaries[repo] = summarize_repo(repo, content)
    except Exception as e:
        summaries[repo] = f"Error summarizing repo: {e}"

with open("data/repo_summaries.json", "w", encoding="utf-8") as f:
    json.dump(summaries, f, indent=2, ensure_ascii=False)

print("\n Summaries saved to data/repo_summaries.json")

# ---------------------------------------------------------
# Step 4: Create Embeddings + FAISS Index
# ---------------------------------------------------------
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

repo_names = list(summaries.keys())
repo_texts = [summaries[r] for r in repo_names]

embeddings = []
for t in tqdm(repo_texts, desc="Embedding repositories"):
    try:
        emb = get_embedding(t)
        embeddings.append(emb)
    except Exception as e:
        print(f" Skipping repo due to embedding error: {e}")

vectors = np.array(embeddings, dtype=np.float32)
if vectors.ndim == 1:
    vectors = vectors.reshape(1, -1)

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors) # type: ignore
faiss.write_index(index, "data/vector_index.faiss")

with open("data/repo_names.json", "w", encoding="utf-8") as f:
    json.dump(repo_names, f, indent=2, ensure_ascii=False)

print("\n Vector index created and saved as data/vector_index.faiss")

# ---------------------------------------------------------
# Step 5: Search Function
# ---------------------------------------------------------
def search_repos(query, k=3):
    q_emb = get_embedding(query)
    q_vec = np.array([q_emb], dtype=np.float32)
    index = faiss.read_index("data/vector_index.faiss")

    with open("data/repo_names.json", encoding="utf-8") as f:
        repo_names = json.load(f)

    D, I = index.search(q_vec, k)
    results = [repo_names[i] for i in I[0]]
    return results

# ---------------------------------------------------------
# Step 6: Run Interactive Search
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n Research Agent Pipeline Ready!")
    query = input("Enter a research topic to find related repositories: ")
    top_repos = search_repos(query)

    print("\n Top matching repositories:")
    for r in top_repos:
        print(" -", r)

    # Generate LLM reasoning/explanation
    explain_results(query, top_repos) # type: ignore


# ---------------------------------------------------------
# Step 7: Explain Results Using LLM
# ---------------------------------------------------------
# ---------------------------------------------------------
# Step 7: Explain Results Using LLM
# ---------------------------------------------------------
def explain_results(query, results):
    # Load repo summaries
    with open("data/repo_summaries.json", "r", encoding="utf-8") as f:
        summaries = json.load(f)

    # Build context for the LLM
    context = "\n\n".join(
        [f"{name}: {summaries.get(name, 'No summary available.')}" for name in results]
    )

    prompt = f"""
    You are an AI research assistant. Given the following repositories and their summaries,
    explain in 3-4 sentences how they relate to the query: "{query}".
    Focus on technical connections and key AI/ML or data science concepts.

    Repositories:
    {context}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    explanation = response.choices[0].message.content.strip() # type: ignore
    print("\n Explanation:\n", explanation)

