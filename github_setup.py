"""Script to push code to existing GitHub repository"""
import os
import subprocess
from pathlib import Path

def setup_git():
    """Initialize git and push code to existing repository"""
    try:
        # Get GitHub token from environment
        github_token = os.environ.get("GITHUB_TOKEN")
        if not github_token:
            raise ValueError("GITHUB_TOKEN not found in environment")

        # Repository settings
        repo_name = "macromaster"
        owner = "sorgeudd"

        # Create .gitignore if it doesn't exist
        gitignore_path = Path(".gitignore")
        if not gitignore_path.exists():
            with open(gitignore_path, "w") as f:
                f.write("""
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Logs
*.log
logs/

# Project specific
*.wav
temp_recordings/
sound_triggers/
recorded_macro.json
macro_visualization.json
*.zip
                """.strip())

        # Initialize git if needed
        if not Path(".git").exists():
            subprocess.run(["git", "init"], check=True)
            print("Initialized git repository")

        # Configure remote with token
        remote_url = f"https://{github_token}@github.com/{owner}/{repo_name}.git"
        subprocess.run(["git", "remote", "remove", "origin"], check=False)
        subprocess.run(["git", "remote", "add", "origin", remote_url], check=True)
        print("Configured git remote")

        # Add and commit files
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Organized Sound Macro Recorder application files"], check=True)
        print("Committed changes")

        # Push to GitHub
        subprocess.run(["git", "push", "-u", "origin", "master"], check=True)
        print("Pushed code to GitHub")

        return True

    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    if setup_git():
        print(f"\nSuccess! Code pushed to GitHub repository")
    else:
        print("\nFailed to push code to GitHub")