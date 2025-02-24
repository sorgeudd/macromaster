"""Script to setup and push code to GitHub repository"""
import os
import requests
import subprocess
from pathlib import Path

# GitHub API Configuration
GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
REPO_NAME = "macromaster"
OWNER = "sorgeudd"

def create_repository():
    """Create GitHub repository if it doesn't exist"""
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Check if repository exists
    repo_url = f"{GITHUB_API}/repos/{OWNER}/{REPO_NAME}"
    response = requests.get(repo_url, headers=headers)
    
    if response.status_code == 404:
        # Create repository if it doesn't exist
        create_url = f"{GITHUB_API}/user/repos"
        data = {
            "name": REPO_NAME,
            "description": "Sound Macro Recorder - Trigger macros with sound",
            "private": False
        }
        response = requests.post(create_url, headers=headers, json=data)
        if response.status_code != 201:
            raise Exception(f"Failed to create repository: {response.text}")
        print(f"Created repository {REPO_NAME}")
    else:
        print(f"Repository {REPO_NAME} already exists")

def setup_git():
    """Initialize git and push code"""
    try:
        # Initialize git if needed
        if not Path(".git").exists():
            subprocess.run(["git", "init"], check=True)
            print("Initialized git repository")

        # Configure remote
        remote_url = f"https://{GITHUB_TOKEN}@github.com/{OWNER}/{REPO_NAME}.git"
        subprocess.run(["git", "remote", "remove", "origin"], check=False)
        subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
        print("Configured git remote")

        # Add and commit files
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit of Sound Macro Recorder"], check=True)
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
        
        create_repository()
        setup_git()
        print(f"\nSuccess! Repository available at: https://github.com/{OWNER}/{REPO_NAME}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
