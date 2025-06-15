# This is the setup script for the project
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv pijepa-env --python 3.12

uv pip install -r requirements.txt