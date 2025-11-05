from git import Repo
from pathlib import Path
from tqdm import tqdm

def clone_repo(url, dest_dir="repos"):
    repo_name = url.split("/")[-1]
    dest_path = Path(dest_dir) / repo_name

    if dest_path.exists():
        print(f"{repo_name} already exists — skipping.")
        return dest_path

    print(f" Cloning {repo_name} ...")
    Repo.clone_from(url, dest_path)
    print(f" Finished cloning {repo_name}")
    return dest_path


def clone_from_list(file_path="repos.txt"):
    Path("repos").mkdir(exist_ok=True)
    with open(file_path, "r") as f:
        urls = [line.strip() for line in f.readlines() if line.strip()]

    for url in tqdm(urls, desc="Cloning GitHub Repositories"):
        try:
            clone_repo(url)
        except Exception as e:
            print(f"  Failed to clone {url}: {e}")

if __name__ == "__main__":
    clone_from_list()
