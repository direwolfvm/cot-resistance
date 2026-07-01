"""Load a local .env (if present) so OPENAI_API_KEY etc. are available to any
entry point that imports the server package (web app or eval harness).

The key stays in the gitignored .env file — never in code or the chat.
"""

from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # dotenv optional; env vars can be set directly
    pass
