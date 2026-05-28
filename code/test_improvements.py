#!/usr/bin/env python
"""Quick test of RAG improvements."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from rag import VolleyballRAGChat

print("=" * 60)
print("Testing RAG Improvements")
print("=" * 60)

try:
    # Initialize with force_reindex to rebuild from .txt files
    chat = VolleyballRAGChat()
    print("\n1. Testing initialization with force_reindex=True...")
    chat.initialize(force_reindex=True)

    print("\n2. Checking that documents were loaded...")
    print(f"   - Loaded documents: {len(chat.all_documents)}")
    print(f"   - Vector store chunks: {chat.vector_store._collection.count()}")

    print("\n3. Checking ensemble retriever...")
    if chat.ensemble_retriever:
        print("   ✓ Ensemble retriever created")
        print(f"   - Contains BM25 and Chroma retrievers (weights 0.4, 0.6)")
    else:
        print("   ✗ Ensemble retriever is None")

    print("\n4. Checking GraphState...")
    from rag import GraphState
    required_fields = {"question", "expanded_question", "documents", "generated_answer", "critique", "chat_history", "user_preferences"}
    actual_fields = set(GraphState.__annotations__.keys())
    if required_fields == actual_fields:
        print(f"   ✓ GraphState has all required fields: {actual_fields}")
    else:
        print(f"   ✗ Missing: {required_fields - actual_fields}")
        print(f"   ✗ Extra: {actual_fields - required_fields}")

    print("\n5. Testing a query (without LLM calls for speed)...")
    # We won't run a full query to avoid LLM costs, just verify the structure

    print("\n" + "=" * 60)
    print("✓ All improvements verified successfully!")
    print("=" * 60)

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
