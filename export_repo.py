import os

# --- Configuration ---
# Add file extensions you want to include
INCLUDE_EXTENSIONS = {'.py', '.js', '.html', '.css', '.md', '.json', '.ts', '.tsx', '.jsx', '.toml', '.yaml','.ipynb'}

# Add directories you want to exclude
EXCLUDE_DIRS = {'node_modules', '.git', '__pycache__', 'dist', 'build', '.vscode'}
# Output file name
OUTPUT_FILE = 'repo_context.txt'
# --- End Configuration ---

def get_repo_contents(root_dir):
    """Walks through the repo and concatenates the content of specified files."""
    repo_contents = ""
    
    # First, get the directory structure
    repo_contents += "Project Directory Structure:\n"
    for root, dirs, files in os.walk(root_dir):
        # Exclude specified directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        repo_contents += f"{indent}{os.path.basename(root)}/\n"
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            repo_contents += f"{sub_indent}{f}\n"
    
    repo_contents += "\n\n" + "="*80 + "\n\n"

    # Then, get the file contents
    for root, dirs, files in os.walk(root_dir):
        # Exclude specified directories
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        for file in files:
            if any(file.endswith(ext) for ext in INCLUDE_EXTENSIONS):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, root_dir)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        repo_contents += f"--- FILE: {relative_path} ---\n"
                        repo_contents += f"```\n{content}\n```\n\n"
                except Exception as e:
                    repo_contents += f"--- FILE: {relative_path} (Error reading) ---\n"
                    repo_contents += f"Could not read file: {e}\n\n"
    return repo_contents

if __name__ == "__main__":
    repo_path = os.getcwd()
    full_context = get_repo_contents(repo_path)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(full_context)
    print(f"Repository context has been written to {OUTPUT_FILE}")