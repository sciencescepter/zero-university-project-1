"""Test setup: the tests never call a model, so they must never send traces either."""

import os

# Set before any project module loads .env (load_dotenv does not override this).
os.environ["LANGSMITH_TRACING"] = "false"
