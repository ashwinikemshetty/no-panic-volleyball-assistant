FROM python:3.11-slim
WORKDIR /app

# Install uv
RUN pip install uv

# Copy code directory (where pyproject.toml is)
COPY code/ .

# Copy docs directory for knowledge base
COPY docs/ ../docs/

# Install dependencies from pyproject.toml
RUN uv sync

# Expose port
EXPOSE 7860

# Run the app
CMD ["uv", "run", "python", "run.py"]
