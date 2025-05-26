from pathlib import Path

def get_project_root() -> Path:
    """Returns the Path object of the project root directory"""
    return Path(__file__).resolve().parent.parent.parent.parent