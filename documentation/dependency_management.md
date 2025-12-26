## Setup
1. Navigate to the root directory '\DLCV-Chess' in your terminal
2. Create a new Python environment
    """
    python -m venv .venv
    """
3. Activate the virtual environment in the terminal
    """
    .venv\Scripts\activate
    """
4. Install Poetry as the dependency manager
    """
    pip install poetry
    """
5. Install all dependencies (Poetry will read the pyproject.toml file):
    """
    poetry install
    """

### Adding new dependencies:
This command adds new dependencies to pyproject.toml and installs them
    """
    poetry add <dependency-name>
    poetry install
    """
By doing this, Poetry automatically updates pyproject.toml and poetry.lock, ensuring that everyone on the team uses the same dependencies.

### Add a Python kernel to run Python files:
Select the following interpreter (In VS Code: Ctrl+Shift+P -> Select Interpreter)
 - Windows: .venv\Scripts\python.exe
 - macOS: .venv/bin/python