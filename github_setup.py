"""Script to push code to existing GitHub repository"""
import os
import subprocess
from pathlib import Path

# GitHub Configuration
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = "macromaster"
OWNER = "sorgeudd"

def setup_git():
    """Initialize git and push code to existing repository"""
    try:
        # Initialize git if needed
        if not Path(".git").exists():
            subprocess.run(["git", "init"], check=True)
            print("Initialized git repository")

        # Configure remote with token
        remote_url = f"https://{GITHUB_TOKEN}@github.com/{OWNER}/{REPO_NAME}.git"
        subprocess.run(["git", "remote", "remove", "origin"], check=False)
        subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
        print("Configured git remote")

        # Add and commit files
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Updated Sound Macro Recorder application"], check=True)
        print("Committed changes")

        # Push to GitHub
        subprocess.run(["git", "push", "-u", "origin", "master"], check=True)
        print("Pushed code to GitHub")

    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
        raise

if __name__ == "__main__":
    try:
        if not GITHUB_TOKEN:
            raise ValueError("GITHUB_TOKEN not found in environment")

        setup_git()
        print(f"\nSuccess! Code pushed to: https://github.com/{OWNER}/{REPO_NAME}")

    except Exception as e:
        print(f"Error: {str(e)}")