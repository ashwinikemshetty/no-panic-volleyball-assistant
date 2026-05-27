import os
from typing import Dict, List, Optional
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from tracing import get_tracer


class GraphState(TypedDict):
    question: str
    expanded_question: str
    documents: List[Document]
    generated_answer: str
    chat_history: List[Dict]
    user_preferences: str


DOCS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "docs"
)

PDF_FILES = [
    "No Panic Coach Packet 2025.pdf",
    "No Panic Official Team Packet 2026.pdf",
]

CHROMA_PERSIST_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "chroma_db"
)

COLLECTION_NAME = "no_panic_volleyball"
CACHE_COLLECTION_NAME = "no_panic_qa_cache"
CACHE_SIMILARITY_THRESHOLD = 0.92

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

SYSTEM_PROMPT = """You are a helpful assistant for No Panic Volleyball Club, answering questions for parents and players.

You have access to the official No Panic Club Packet (2026) and Coach Packet (2025).
Always:
- Answer directly from the provided context
- If the information is not in the context, say so clearly rather than guessing
- Include specific details like dates, names, contact emails when available in the context
- Keep answers concise but complete
- When citing information, reference the source document and page number like: (Coach Packet 2025, p.12)
"""


class VolleyballRAGChat:
    """RAG chat interface for No Panic Volleyball Club Q&A."""

    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.cache_store = None
        self.graph = None
        self.memory_cache = {}  # Simple in-memory cache for exact matches

    def initialize(self) -> None:
        """Initialize LLM, embeddings, vector store, and graph."""
        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=COLLECTION_NAME,
        )

        # Guard: only load and embed docs if collection is empty
        existing_count = self.vector_store._collection.count()
        if existing_count == 0:
            print("Vector store empty — loading and embedding PDFs...")
            self._load_and_embed_docs()
        else:
            print(f"Vector store has {existing_count} chunks — skipping re-embedding.")

        # Initialize semantic cache for Q&A pairs
        self.cache_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=CACHE_COLLECTION_NAME,
        )

        self._build_graph()

    def _load_and_embed_docs(self) -> None:
        """Load PDFs, split into chunks, and embed into Chroma."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""],
        )

        all_chunks = []
        for pdf_file in PDF_FILES:
            file_path = os.path.join(DOCS_DIR, pdf_file)
            if not os.path.exists(file_path):
                print(f"WARNING: {pdf_file} not found at {file_path}")
                continue

            loader = PyPDFLoader(file_path)
            pages = loader.load()

            for page in pages:
                page.metadata["source"] = pdf_file
                page.metadata["page"] = page.metadata.get("page", 0)

            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
            print(f"Loaded {len(pages)} pages → {len(chunks)} chunks from {pdf_file}")

        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            print(f"Embedded {len(all_chunks)} total chunks into ChromaDB.")
        else:
            print("No chunks to embed — check PDF file paths.")

    def extract_preferences(self, state: GraphState) -> dict:
        """Extract user preferences from conversation history."""
        history = state.get("chat_history") or []
        if not history:
            return {"user_preferences": ""}

        history_text = "\n".join([
            f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
            for msg in history
        ])

        prompt = (
            "You are analyzing a volleyball club conversation to extract user preferences.\n\n"
            "Conversation so far:\n"
            f"{history_text}\n\n"
            f"Current question: {state['question']}\n\n"
            "Extract any stated or implied preferences (age group, gender, activity, team level, "
            "coach preference, timeframe, etc.). If the user says 'my preference' or 'my choice', "
            "identify exactly what they meant from the conversation history.\n\n"
            "If no preferences found, return: No specific preferences stated.\n\n"
            "Preferences (be concise and specific):"
        )

        prefs = self.llm.invoke(prompt).content.strip()
        return {"user_preferences": prefs}

    def expand_query(self, state: GraphState) -> dict:
        """Expand user question into document-friendly search terms, incorporating preferences."""
        pref_context = (
            f"\n\nEstablished user preferences:\n{state['user_preferences']}"
            if state.get("user_preferences") else ""
        )

        prompt = (
            "You are a search query optimizer for a volleyball club knowledge base.\n\n"
            "Rewrite the following question into a search-friendly query that:\n"
            "- Expands casual language into formal club document terms\n"
            "- Incorporates the user's stated preferences (e.g., if they said 'girls 14u', add that)\n"
            "- If user says 'my preference/choice/team', substitute the actual preference from history\n"
            "- Preserves the original intent\n"
            f"{pref_context}\n\n"
            f"Original question: {state['question']}\n\n"
            "Rewritten query (just the query, no explanation):"
        )
        expanded = self.llm.invoke(prompt).content.strip()
        return {"expanded_question": expanded}

    def retrieve_documents(self, state: GraphState) -> dict:
        """Retrieve relevant documents from vector store using expanded query."""
        docs = self.vector_store.similarity_search(state["expanded_question"], k=6)
        if not docs:
            print(f"WARNING: No docs retrieved for: {state['expanded_question']}")
        return {"documents": docs}

    def generate_answer(self, state: GraphState) -> dict:
        """Generate answer using retrieved documents and user preferences."""
        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in state["documents"]
        ])

        # Inject preferences into system prompt if present
        effective_system_prompt = SYSTEM_PROMPT
        if state.get("user_preferences"):
            effective_system_prompt += (
                f"\n\nPreferences established in this conversation:\n{state['user_preferences']}\n"
                "Prioritize information matching these preferences. When user says 'my preferred choice' "
                "or similar, use these preferences to answer specifically."
            )

        prompt = PromptTemplate.from_template(
            "{system_prompt}\n\n"
            "Question: {question}\n\n"
            "Context from club documents:\n{context}\n\n"
            "Answer:"
        )

        formatted = prompt.format(
            system_prompt=effective_system_prompt,
            question=state["question"],
            context=context,
        )

        answer = self.llm.invoke(formatted).content
        return {"generated_answer": answer}

    def _build_graph(self) -> None:
        """Build the LangGraph state machine."""
        tracer = get_tracer()

        def traced_extract_preferences(state: GraphState) -> dict:
            with tracer.start_as_current_span("langgraph.node.extract_preferences"):
                return self.extract_preferences(state)

        def traced_expand_query(state: GraphState) -> dict:
            with tracer.start_as_current_span("langgraph.node.expand_query"):
                return self.expand_query(state)

        def traced_retrieve_documents(state: GraphState) -> dict:
            with tracer.start_as_current_span("langgraph.node.retrieve_documents"):
                return self.retrieve_documents(state)

        def traced_generate_answer(state: GraphState) -> dict:
            with tracer.start_as_current_span("langgraph.node.generate_answer"):
                return self.generate_answer(state)

        builder = StateGraph(GraphState)
        builder.add_node("extract_preferences", traced_extract_preferences)
        builder.add_node("expand_query", traced_expand_query)
        builder.add_node("retrieve_documents", traced_retrieve_documents)
        builder.add_node("generate_answer", traced_generate_answer)
        builder.add_edge(START, "extract_preferences")
        builder.add_edge("extract_preferences", "expand_query")
        builder.add_edge("expand_query", "retrieve_documents")
        builder.add_edge("retrieve_documents", "generate_answer")
        builder.add_edge("generate_answer", END)
        self.graph = builder.compile()

    def process_message(self, message: str, chat_history: Optional[List[Dict]] = None) -> str:
        """Process a user message and return the assistant's response."""
        # 1. Check memory cache first (instant, exact match)
        if message in self.memory_cache:
            print(f"[MEMORY CACHE HIT] Instant response for: '{message[:50]}...'")
            return self.memory_cache[message]

        # 2. Check Chroma semantic cache (1 sec, similar questions)
        cache_count = self.cache_store._collection.count()
        if cache_count > 0:
            try:
                cached_results = self.cache_store.similarity_search(message, k=1, score_threshold=0.90)
                if cached_results:
                    print(f"[SEMANTIC CACHE HIT] Returning similar cached answer")
                    return cached_results[0].metadata["answer"]
            except Exception as e:
                pass

        # 3. Cache miss — run full RAG pipeline

        result = self.graph.invoke({
            "question": message,
            "chat_history": chat_history or [],
            "user_preferences": "",
        })
        answer = result["generated_answer"]

        # Store in both caches
        self.memory_cache[message] = answer  # For instant repeat questions
        self.cache_store.add_documents([
            Document(page_content=message, metadata={"answer": answer})
        ])

        return answer
