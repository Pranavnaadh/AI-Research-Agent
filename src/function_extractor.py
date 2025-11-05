print(" Python is running this script at all!")

import os
import json
import ast
from pathlib import Path
from tqdm import tqdm

def extract_definitions(file_path):
    """Extract function and class names from a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source)
        functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        return {"functions": functions, "classes": classes}
    except Exception as e:
        return {"error": str(e)}

def scan_repo(repo_path):
    """Scan a single repository folder for Python files."""
    repo_summary = {}
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                rel_path = os.path.relpath(path, repo_path)
                repo_summary[rel_path] = extract_definitions(path)
    return repo_summary

if __name__ == "__main__":
    print(" Script started running!")  # Force print
    base_dir = Path("repos")
    output_file = Path("repo_code_summary.json")

    print(" Current working directory:", os.getcwd())
    print(" Checking for repos folder:", base_dir.resolve())

    if not base_dir.exists():
        print(" repos folder not found!")
        exit(1)

    repos = [p for p in base_dir.iterdir() if p.is_dir()]
    print(f" Found {len(repos)} repositories to scan.\n")

    all_data = {}
    for repo in tqdm(repos, desc="Extracting code from repos"):
        all_data[repo.name] = scan_repo(repo)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    print(f"\n Extraction complete! Saved summaries to {output_file.resolve()}")
