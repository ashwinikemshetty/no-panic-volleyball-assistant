# No Panic Volleyball Assistant

A RAG-powered chatbot for No Panic Volleyball Club using LangChain, LangGraph, ChromaDB, and OpenAI.

## Setup

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Set up environment variables**:
   Create a `.env` file in the `code/` directory:
   ```
   OPENAI_API_KEY=sk-...
   ```

3. **Run locally**:
   ```bash
   cd code
   python run.py
   ```
   Visit `http://localhost:7860`

## Deployment

### Render (recommended)

1. Push this repo to GitHub
2. Go to [render.com](https://render.com)
3. Create new Web Service
4. Connect this GitHub repo
5. Set environment variable: `OPENAI_API_KEY=sk-...`
6. Deploy!

### Gradio Spaces

1. Push to GitHub
2. Create new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
3. Connect GitHub repo, choose "Gradio"
4. Add secret: `OPENAI_API_KEY=sk-...`

## Files

- `code/app.py` - Gradio UI
- `code/rag.py` - RAG pipeline with LangGraph
- `code/run.py` - Entry point
- `code/tracing.py` - OpenTelemetry tracing setup
- `docs/` - PDF knowledge base
