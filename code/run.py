import os
import sys
import argparse
from dotenv import load_dotenv

# Ensure the code directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env from explicit path
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)

# Initialize ARIZE tracing (before importing LangChain)
from tracing import setup_tracing
setup_tracing()

parser = argparse.ArgumentParser(description="No Panic Volleyball Assistant")
parser.add_argument(
    "--port", type=int, default=7860,
    help="Port to run Gradio on (default: 7860)"
)
parser.add_argument(
    "--share", action="store_true",
    help="Create a public Gradio share link"
)
parser.add_argument(
    "--force-reindex", action="store_true",
    help="Force rebuild the vector store from documents (clears existing index)"
)
args = parser.parse_args()

if __name__ == "__main__":
    from app import create_demo
    demo = create_demo(force_reindex=args.force_reindex)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )
