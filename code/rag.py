import os
import glob
import json
from typing import Dict, List, Optional
from typing_extensions import TypedDict

import pdfplumber
import pandas as pd

from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langgraph.graph import StateGraph, START, END

from tracing import get_tracer
from schedule_loader import ScheduleLoader, resolve_date


class GraphState(TypedDict):
    question: str
    expanded_question: str
    documents: List[Document]
    generated_answer: str
    critique: str
    chat_history: List[Dict]
    user_preferences: str
    query_type: str
    schedule_entities: Dict
    schedule_result: str


DOCS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "docs"
)

SCHEDULE_FILE = os.path.join(DOCS_DIR, "CrossbarCourtSchedule.xlsx")

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
CACHE_SIMILARITY_THRESHOLD = 0.90

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
ENSEMBLE_K = 8

SYSTEM_PROMPT = """You are a helpful assistant for No Panic Volleyball Club, answering questions for parents and players.

You have access to the official No Panic Club Packet (2026) and Coach Packet (2025).
Always:
- Answer directly from the provided context
- If the information is not in the context, say so clearly rather than guessing
- Include specific details like dates, names, contact emails when available in the context
- Keep answers concise but complete
- When citing information, reference the source document and page number like: (Coach Packet 2025, p.12)
"""


def convert_pdf_to_text(pdf_path: str) -> str:
    """Convert PDF to text using pdfplumber, preserving structure.

    Args:
        pdf_path: Full path to PDF file

    Returns:
        Extracted text with formatting preserved
    """
    text_parts = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract text with layout preserved
                text = page.extract_text() or ""
                if text.strip():
                    text_parts.append(f"--- Page {page_num} ---\n{text}\n")
        return "\n".join(text_parts)
    except Exception as e:
        print(f"ERROR converting {pdf_path}: {e}")
        return ""


def auto_convert_pdfs() -> None:
    """Auto-convert PDFs to .txt files if .txt doesn't already exist.

    This runs on startup to create clean text files from PDFs.
    Once created, .txt files are used in preference to PDFs.
    """
    pdf_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))

    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        # Create .txt filename from PDF (e.g., "Coach Packet.pdf" → "Coach Packet.txt")
        txt_name = os.path.splitext(pdf_name)[0] + ".txt"
        txt_path = os.path.join(DOCS_DIR, txt_name)

        # Skip conversion if .txt already exists
        if os.path.exists(txt_path):
            print(f"Skipping {pdf_name} — {txt_name} already exists")
            continue

        print(f"Converting {pdf_name} to text...")
        text = convert_pdf_to_text(pdf_path)

        if text:
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"  ✓ Created {txt_name} ({len(text)} chars)")
            except Exception as e:
                print(f"  ✗ Failed to write {txt_path}: {e}")
        else:
            print(f"  ✗ No text extracted from {pdf_name}")


class VolleyballRAGChat:
    """RAG chat interface for No Panic Volleyball Club Q&A."""

    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vector_store = None
        self.cache_store = None
        self.graph = None
        self.memory_cache = {}  # Simple in-memory cache for exact matches
        self.all_documents = []  # Store for BM25Retriever
        self.bm25_retriever = None
        self.schedule_loader = None  # Schedule query interface

    def initialize(self, force_reindex: bool = False) -> None:
        """Initialize LLM, embeddings, vector store, and graph.

        Args:
            force_reindex: If True, clear and rebuild the vector store from documents.
        """
        # Load schedule
        print("Loading schedule...")
        self.schedule_loader = ScheduleLoader(SCHEDULE_FILE)

        # Auto-convert PDFs to text on startup (if .txt files don't exist)
        print("Checking for PDFs to convert...")
        auto_convert_pdfs()

        self.llm = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=COLLECTION_NAME,
        )

        # Clear collection if force_reindex is True
        if force_reindex:
            print("Force reindex enabled — clearing existing collection...")
            self.vector_store._collection.delete(where={})

        # Load and embed docs if collection is empty or force_reindex is set
        existing_count = self.vector_store._collection.count()
        if existing_count == 0:
            print("Vector store empty — loading and embedding documents...")
            self._load_and_embed_docs()
        else:
            print(f"Vector store has {existing_count} chunks — skipping re-embedding.")
            # Still load documents into memory for BM25 even if vector store exists
            self._load_documents_to_memory()

        # Initialize semantic cache for Q&A pairs
        self.cache_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name=CACHE_COLLECTION_NAME,
        )

        # Build ensemble retriever (BM25 + semantic)
        if self.all_documents:
            self._build_ensemble_retriever()

        self._build_graph()

    def _load_documents_to_memory(self) -> None:
        """Load and chunk documents into memory (for BM25) without embedding.

        Used when vector store already exists but BM25 index needs to be rebuilt.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n\n", "\n\n", "\n", " ", ""],
        )

        all_docs = []

        # Load .txt files
        txt_files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
        for file_path in txt_files:
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                pages = loader.load()

                filename = os.path.basename(file_path)
                for page in pages:
                    page.metadata["source"] = filename
                    page.metadata["page"] = 0

                chunks = splitter.split_documents(pages)
                all_docs.extend(chunks)
                print(f"Loaded .txt file → {len(chunks)} chunks from {filename} (memory)")
            except Exception as e:
                print(f"WARNING: Failed to load {file_path}: {e}")
                continue

        # Load .pdf files (fallback)
        for pdf_file in PDF_FILES:
            file_path = os.path.join(DOCS_DIR, pdf_file)
            if not os.path.exists(file_path):
                continue

            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()

                for page in pages:
                    page.metadata["source"] = pdf_file
                    page.metadata["page"] = page.metadata.get("page", 0)

                chunks = splitter.split_documents(pages)
                all_docs.extend(chunks)
                print(f"Loaded {len(pages)} pages → {len(chunks)} chunks from {pdf_file} (memory)")
            except Exception as e:
                print(f"WARNING: Failed to load {pdf_file}: {e}")
                continue

        self.all_documents = all_docs
        print(f"Loaded {len(all_docs)} total chunks into memory for BM25.")

    def _load_and_embed_docs(self) -> None:
        """Load .txt and .pdf files, split into chunks, and embed into Chroma.

        Also keeps all documents in memory for BM25Retriever.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n\n", "\n\n", "\n", " ", ""],
        )

        all_chunks = []
        all_docs = []  # For BM25Retriever

        # Load .txt files (copy-pasted document text)
        txt_files = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
        for file_path in txt_files:
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                pages = loader.load()

                filename = os.path.basename(file_path)
                for page in pages:
                    page.metadata["source"] = filename
                    page.metadata["page"] = 0

                chunks = splitter.split_documents(pages)
                all_chunks.extend(chunks)
                all_docs.extend(chunks)
                print(f"Loaded .txt file → {len(chunks)} chunks from {filename}")
            except Exception as e:
                print(f"WARNING: Failed to load {file_path}: {e}")
                continue

        # Load .pdf files (fallback for any clean PDFs)
        for pdf_file in PDF_FILES:
            file_path = os.path.join(DOCS_DIR, pdf_file)
            if not os.path.exists(file_path):
                print(f"NOTE: {pdf_file} not found at {file_path}")
                continue

            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()

                for page in pages:
                    page.metadata["source"] = pdf_file
                    page.metadata["page"] = page.metadata.get("page", 0)

                chunks = splitter.split_documents(pages)
                all_chunks.extend(chunks)
                all_docs.extend(chunks)
                print(f"Loaded {len(pages)} pages → {len(chunks)} chunks from {pdf_file}")
            except Exception as e:
                print(f"WARNING: Failed to load {pdf_file}: {e}")
                continue

        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            self.all_documents = all_docs  # Keep for BM25Retriever
            print(f"Embedded {len(all_chunks)} total chunks into ChromaDB.")
        else:
            print("No chunks to embed — check document file paths.")

    def _build_ensemble_retriever(self) -> None:
        """Build hybrid BM25 + semantic ensemble retriever."""
        self.bm25_retriever = BM25Retriever.from_documents(self.all_documents)
        self.bm25_retriever.k = ENSEMBLE_K
        print(f"Built ensemble retriever with {len(self.all_documents)} documents for BM25.")

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
        """Retrieve relevant documents using hybrid BM25 + semantic search."""
        query = state["expanded_question"]

        if hasattr(self, 'bm25_retriever') and self.bm25_retriever is not None:
            # Hybrid search: BM25 (40%) + Semantic (60%)
            try:
                bm25_docs = self.bm25_retriever.invoke(query)
            except Exception as e:
                print(f"WARNING: BM25 retrieval failed: {e}, falling back to semantic")
                bm25_docs = []

            semantic_docs = self.vector_store.similarity_search(query, k=ENSEMBLE_K)

            # Combine results, preferring semantic but boosting BM25 matches
            seen = set()
            combined = []

            # Add semantic results first (60% weight)
            for doc in semantic_docs:
                doc_id = doc.page_content[:100]  # Use content hash
                if doc_id not in seen:
                    combined.append(doc)
                    seen.add(doc_id)

            # Add BM25 results (40% weight) if not already included
            for doc in bm25_docs:
                doc_id = doc.page_content[:100]
                if doc_id not in seen and len(combined) < ENSEMBLE_K:
                    combined.append(doc)
                    seen.add(doc_id)

            docs = combined[:ENSEMBLE_K]
        else:
            # Fallback to semantic search if BM25 not available
            print("Using semantic search only (BM25 not available)")
            docs = self.vector_store.similarity_search(query, k=ENSEMBLE_K)

        if not docs:
            print(f"WARNING: No docs retrieved for: {query}")
        return {"documents": docs}

    def generate_answer(self, state: GraphState) -> dict:
        """Generate answer using retrieved documents, schedule results, and user preferences."""
        query_type = state.get("query_type", "general")
        schedule_result = state.get("schedule_result", "")

        # Handle clarification requests
        if schedule_result == "CLARIFY_TEAM":
            answer = (
                "I found multiple teams in our conversation history. Which team would you like to know about? "
                "Please clarify which team you're interested in."
            )
            return {"generated_answer": answer}

        if schedule_result == "CLARIFY_DATE":
            answer = (
                "I know which team you're asking about, but I need to know which date. "
                "Are you asking about today, tomorrow, a specific date, or all upcoming sessions?"
            )
            return {"generated_answer": answer}

        # Build context based on query type
        context = ""

        if query_type == "schedule":
            # Use schedule result as primary context
            context = f"Court Schedule Results:\n{schedule_result}"
        elif query_type == "hybrid":
            # Combine schedule and document results
            schedule_context = f"Court Schedule:\n{schedule_result}"

            doc_context = "\n\n".join([
                f"[Source: {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
                for doc in state["documents"]
            ])

            context = f"{schedule_context}\n\nClub Information:\n{doc_context}"
        else:  # general
            # Use document retrieval (existing behavior)
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

        # For schedule queries, adjust system prompt
        if query_type in ["schedule", "hybrid"]:
            effective_system_prompt += (
                "\n\nYou are answering a court schedule question. Present the schedule information clearly, "
                "including team names, court numbers, and times. Be concise and organized."
            )

        prompt = PromptTemplate.from_template(
            "{system_prompt}\n\n"
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:"
        )

        formatted = prompt.format(
            system_prompt=effective_system_prompt,
            question=state["question"],
            context=context,
        )

        answer = self.llm.invoke(formatted).content
        return {"generated_answer": answer}

    def critique_answer(self, state: GraphState) -> dict:
        """Self-critique: verify answer is supported by retrieved context."""
        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')}, Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in state["documents"]
        ])

        critique_prompt = (
            "You are a fact-checker for a volleyball club knowledge base.\n\n"
            "Given the context chunks and the answer below, identify any factual claims in the answer "
            "that are NOT directly supported by the chunks. Be specific about what's unsupported.\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:\n{state['generated_answer']}\n\n"
            "Task: List unsupported claims, or respond with 'SUPPORTED' if everything is backed by context.\n\n"
            "Your assessment (be concise):"
        )

        critique = self.llm.invoke(critique_prompt).content.strip()

        # If critique found unsupported claims, append a note to the answer
        final_answer = state["generated_answer"]
        if critique and critique != "SUPPORTED":
            final_answer += (
                "\n\n⚠️ Note: Some details above may not be in the current documents. "
                "Please verify with your coach or club coordinator."
            )

        return {"critique": critique, "generated_answer": final_answer}

    def route_query(self, state: GraphState) -> dict:
        """Route query to schedule, general, or hybrid path."""
        history_text = ""
        if state.get("chat_history"):
            history_text = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in state["chat_history"][-3:]  # Last 3 turns
            ])

        prompt = (
            "You are a query router for a volleyball club assistant.\n\n"
            "Classify the user's question as one of: 'schedule', 'general', or 'hybrid'.\n\n"
            "- 'schedule': asks about court availability, session times, which team plays when, today's schedule, etc.\n"
            "- 'general': asks about club policy, jersey info, tryouts, coach details, etc.\n"
            "- 'hybrid': asks about both schedule and club info (e.g., 'session times and coach name')\n\n"
            f"Recent conversation:\n{history_text}\n\n"
            f"Current question: {state['question']}\n\n"
            "Classification (just one word: schedule, general, or hybrid):"
        )

        query_type = self.llm.invoke(prompt).content.strip().lower()
        if query_type not in ["schedule", "general", "hybrid"]:
            query_type = "general"  # Default to general if unclear

        return {"query_type": query_type}

    def resolve_schedule_entities(self, state: GraphState) -> dict:
        """Extract and resolve schedule entities from question + history."""
        history_text = ""
        if state.get("chat_history"):
            history_text = "\n".join([
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in state["chat_history"][-5:]  # Last 5 turns
            ])

        prompt = (
            "You are extracting structured schedule query parameters from a user question.\n\n"
            "Extract the following from the question and conversation history:\n"
            "- team_name: exact team name mentioned (or null if none)\n"
            "- gender_filter: 'girls' or 'boys' if asking about a gender group (or null)\n"
            "- date_expr: date mention like 'today', '05/27/2026', 'May 27', etc. (or null if none)\n"
            "- is_count: true if asking 'how many', false otherwise\n"
            "- court_only: true if only asking for court/space, false if also asking for times\n"
            "- needs_clarification: 'team' if ambiguous (multiple teams in history, pronoun used), "
            "'date' if team known but date missing, or null if clear\n\n"
            "Context resolution:\n"
            "1. If the question uses pronouns ('they', 'their', 'the team') without explicit name,\n"
            "   look back in chat history for the last explicitly mentioned team name.\n"
            "2. If TWO or more distinct teams appear in history and current question is ambiguous,\n"
            "   set needs_clarification='team' (don't guess which one).\n"
            "3. If a team is known but no date in question and no recent date in history,\n"
            "   set needs_clarification='date'.\n\n"
            f"Chat history:\n{history_text}\n\n"
            f"Current question: {state['question']}\n"
            f"User preferences: {state.get('user_preferences', '')}\n\n"
            "Return as JSON:\n"
            '{"team_name": ..., "gender_filter": ..., "date_expr": ..., "is_count": ..., '
            '"court_only": ..., "needs_clarification": ...}'
        )

        try:
            import json
            response = self.llm.invoke(prompt).content.strip()
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                entities = json.loads(response[start:end])
            else:
                entities = {}
        except Exception as e:
            print(f"WARNING: Failed to parse schedule entities: {e}")
            entities = {}

        return {"schedule_entities": entities}

    def query_schedule(self, state: GraphState) -> dict:
        """Execute schedule query using resolved entities."""
        if not self.schedule_loader or self.schedule_loader.df is None:
            return {"schedule_result": "Schedule data not available.", "schedule_entities": state.get("schedule_entities", {})}

        entities = state.get("schedule_entities", {})
        needs_clarification = entities.get("needs_clarification")

        # Handle clarification needs
        if needs_clarification == "team":
            return {
                "schedule_result": "CLARIFY_TEAM",
                "schedule_entities": entities
            }
        elif needs_clarification == "date":
            return {
                "schedule_result": "CLARIFY_DATE",
                "schedule_entities": entities
            }

        # Resolve date if provided
        date_val = None
        if entities.get("date_expr"):
            date_val = resolve_date(entities["date_expr"])

        # Execute appropriate query
        results = []
        if entities.get("team_name"):
            results = self.schedule_loader.query_by_team(entities["team_name"], date_val)
        elif entities.get("gender_filter"):
            # Classify teams if not done yet
            if not self.schedule_loader.team_gender_cache:
                def llm_classify(team_names):
                    prompt = (
                        "Classify each team name as 'girls', 'boys', 'mixed', or 'other'.\n\n"
                        "Teams:\n" + "\n".join(f"- {t}" for t in team_names) + "\n\n"
                        "Return JSON: {team_name: gender_label, ...}"
                    )
                    import json
                    response = self.llm.invoke(prompt).content.strip()
                    start = response.find("{")
                    end = response.rfind("}") + 1
                    if start >= 0 and end > start:
                        return json.loads(response[start:end])
                    return {}

                self.schedule_loader.classify_teams_by_gender(llm_classify)

            results = self.schedule_loader.query_by_gender(entities["gender_filter"], date_val)
        elif date_val:
            results = self.schedule_loader.query_by_date(date_val)

        # Format results
        if not results:
            schedule_result = "No sessions found matching your query."
        else:
            if entities.get("is_count"):
                schedule_result = f"Found {len(results)} session(s):\n"
            else:
                schedule_result = ""

            for row in results:
                court = row.get("space", "N/A")
                team = row.get("teams", "N/A")
                start = row.get("start", "N/A")
                end = row.get("end", "N/A")
                date_str = pd.to_datetime(row.get("date", "")).strftime("%a, %b %d")

                schedule_result += f"- {team} | {court} | {start}–{end} ({date_str})\n"

        return {"schedule_result": schedule_result, "schedule_entities": entities}

    def _build_graph(self) -> None:
        """Build the LangGraph state machine with schedule routing."""
        tracer = get_tracer()

        def traced_extract_preferences(state: GraphState) -> dict:
            with tracer.start_as_current_span("langgraph.node.extract_preferences"):
                return self.extract_preferences(state)

        def traced_route_query(state: GraphState) -> dict:
            with tracer.start_as_current_span("langgraph.node.route_query"):
                return self.route_query(state)

        def traced_resolve_schedule_entities(state: GraphState) -> dict:
            with tracer.start_as_current_span("langgraph.node.resolve_schedule_entities"):
                return self.resolve_schedule_entities(state)

        def traced_query_schedule(state: GraphState) -> dict:
            with tracer.start_as_current_span("langgraph.node.query_schedule"):
                return self.query_schedule(state)

        def traced_expand_query(state: GraphState) -> dict:
            with tracer.start_as_current_span("langgraph.node.expand_query"):
                return self.expand_query(state)

        def traced_retrieve_documents(state: GraphState) -> dict:
            with tracer.start_as_current_span("langgraph.node.retrieve_documents"):
                return self.retrieve_documents(state)

        def traced_generate_answer(state: GraphState) -> dict:
            with tracer.start_as_current_span("langgraph.node.generate_answer"):
                return self.generate_answer(state)

        def traced_critique_answer(state: GraphState) -> dict:
            with tracer.start_as_current_span("langgraph.node.critique_answer"):
                return self.critique_answer(state)

        def route_to_query_type(state: GraphState) -> str:
            """Route based on query type."""
            return state.get("query_type", "general")

        builder = StateGraph(GraphState)

        # Add all nodes
        builder.add_node("extract_preferences", traced_extract_preferences)
        builder.add_node("route_query", traced_route_query)
        builder.add_node("resolve_schedule_entities", traced_resolve_schedule_entities)
        builder.add_node("query_schedule", traced_query_schedule)
        builder.add_node("expand_query", traced_expand_query)
        builder.add_node("retrieve_documents", traced_retrieve_documents)
        builder.add_node("generate_answer", traced_generate_answer)
        builder.add_node("critique_answer", traced_critique_answer)

        # Main flow
        builder.add_edge(START, "extract_preferences")
        builder.add_edge("extract_preferences", "route_query")

        # Conditional routing based on query type
        builder.add_conditional_edges(
            "route_query",
            route_to_query_type,
            {
                "schedule": "resolve_schedule_entities",
                "general": "expand_query",
                "hybrid": "resolve_schedule_entities",  # Start with schedule for hybrid
            }
        )

        # Schedule path
        builder.add_edge("resolve_schedule_entities", "query_schedule")
        # After schedule query, go to expand_query for hybrid, or straight to generate for schedule
        def route_after_schedule(state: GraphState) -> str:
            if state.get("query_type") == "hybrid":
                return "expand_query"
            else:
                return "generate_answer"

        builder.add_conditional_edges(
            "query_schedule",
            route_after_schedule,
            {
                "expand_query": "expand_query",
                "generate_answer": "generate_answer",
            }
        )

        # General/Document path
        builder.add_edge("expand_query", "retrieve_documents")
        builder.add_edge("retrieve_documents", "generate_answer")

        # Final steps
        builder.add_edge("generate_answer", "critique_answer")
        builder.add_edge("critique_answer", END)

        self.graph = builder.compile()

    def _is_schedule_query(self, message: str) -> bool:
        """Quick heuristic: is this likely a schedule query?

        Schedule queries should never be cached since "today" changes daily.
        """
        schedule_keywords = [
            "court", "session", "time", "schedule", "when",
            "today", "tomorrow", "yesterday", "date",
            "game", "practice", "training", "match",
            "team", "girls", "boys", "coach",
            "which court", "what time", "what sessions",
        ]
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in schedule_keywords)

    def process_message(self, message: str, chat_history: Optional[List[Dict]] = None) -> str:
        """Process a user message and return the assistant's response."""
        # Skip caching for schedule queries (dates change daily)
        is_schedule = self._is_schedule_query(message)

        if not is_schedule:
            # 1. Check memory cache first (instant, exact match) — only for non-schedule queries
            if message in self.memory_cache:
                print(f"[MEMORY CACHE HIT] Instant response for: '{message[:50]}...'")
                return self.memory_cache[message]

            # 2. Check Chroma semantic cache (1 sec, similar questions) — only for non-schedule queries
            cache_count = self.cache_store._collection.count()
            if cache_count > 0:
                try:
                    cached_results = self.cache_store.similarity_search(message, k=1, score_threshold=0.90)
                    if cached_results:
                        print(f"[SEMANTIC CACHE HIT] Returning similar cached answer")
                        return cached_results[0].metadata["answer"]
                except Exception as e:
                    pass

        # 3. Cache miss or schedule query — run full RAG pipeline

        result = self.graph.invoke({
            "question": message,
            "chat_history": chat_history or [],
            "user_preferences": "",
            "expanded_question": "",
            "documents": [],
            "generated_answer": "",
            "critique": "",
            "query_type": "general",
            "schedule_entities": {},
            "schedule_result": "",
        })
        answer = result["generated_answer"]

        # Store in both caches
        self.memory_cache[message] = answer  # For instant repeat questions
        self.cache_store.add_documents([
            Document(page_content=message, metadata={"answer": answer})
        ])

        return answer
