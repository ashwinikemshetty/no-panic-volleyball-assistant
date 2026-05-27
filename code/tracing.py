import os
from opentelemetry import trace
from arize.otel import register, Endpoint
from openinference.instrumentation.langchain import LangChainInstrumentor

PROJECT_NAME = "no-panic-volleyball-rag"


def setup_tracing():
    """Initialize ARIZE tracing. Call once before app starts."""
    api_key = os.environ.get("ARIZE_API_KEY")
    space_id = os.environ.get("ARIZE_SPACE_ID")

    if not api_key or not space_id:
        print("[TRACING] ARIZE_API_KEY or ARIZE_SPACE_ID not set — tracing disabled.")
        return None

    tracer_provider = register(
        endpoint=Endpoint.ARIZE,
        space_id=space_id,
        api_key=api_key,
        project_name=PROJECT_NAME,
    )

    LangChainInstrumentor(tracer_provider=tracer_provider).instrument()

    print(f"[TRACING] ARIZE tracing enabled — project: {PROJECT_NAME}")
    return tracer_provider


def get_tracer():
    """Get tracer for custom spans."""
    return trace.get_tracer(PROJECT_NAME)
