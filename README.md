# recruitment-task

## Installation & Setup

Running this project is possible using either `uv` or `pip`.

### Using `uv`
```bash
# Install dependencies
uv sync
# Run tests
uv run pytest
# Run pipeline
uv run python pipeline.py
```

### Using `pip` & `venv`
```bash
# Create virtual environment
python -m venv .venv
# Activate virtual environment:
source .venv/bin/activate # Linux
.venv\Scripts\activate # Windows
# Install dependencies and run
pip install -r requirements.txt
pytest
python pipeline.py
```

## Assumptions & Design Decisions

Throughout completing the tasks there were a few architectural decisions that had to be made. These decisions are listed below.

### Setup
- Decided to keep only one `requirements.txt` and not separate a dev file for simplicity. There still is separation in the uv setup.

### Part 1

- The task specifies addition, subtraction, and multiplication. Since division (`/`) is not mentioned, it is treated as unsupported and rejected during validation. It could be easily changed by a small modification to `_OPERATOR_SPLIT_PATTERN` and `_VALID_ROLE_CHARS_PATTERN`.
