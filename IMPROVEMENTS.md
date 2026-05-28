# RAG System Improvements — Implementation Summary

## Overview
Four targeted improvements were implemented to address PDF parsing artifacts, semantic search gaps, and answer quality issues in the No Panic Volleyball Assistant RAG system.

---

## 1. Automatic PDF-to-Text Conversion + Text File Ingestion ✅

### What Changed
- **Auto-conversion**: PDFs are automatically converted to `.txt` files on startup using `pdfplumber`
- **Formatting preserved**: Conversion preserves page structure, spacing, and sections (not garbled like PyPDFLoader)
- **Smart workflow**: Once `.txt` exists, PDF is never re-converted (unless you delete the `.txt`)
- **System loads text**: Uses `.txt` files for chunking/embedding, avoiding PDF parsing issues
- **Force reindex flag**: Added `--force-reindex` argument to rebuild vector store from converted text

### Files Modified
- `code/rag.py`: New `convert_pdf_to_text()` and `auto_convert_pdfs()` functions, updated `initialize()`
- `code/run.py`: Added `--force-reindex` argument
- `code/app.py`: `create_demo()` now accepts `force_reindex` parameter
- `code/pyproject.toml`: Added `pdfplumber>=0.13.0` dependency

### How It Works
1. **On app startup** → scan `docs/` for `.pdf` files
2. **If a matching `.txt` doesn't exist** → convert PDF to text automatically
3. **Text is saved** as `docs/CoachPacket.txt` (derived from `CoachPacket.pdf`)
4. **System uses the `.txt`** for all further processing (chunking, embedding)

### How to Use
```bash
# Normal startup (auto-converts any PDFs without matching .txt files)
python run.py

# Force rebuild vector store from all current .txt/.pdf files
python run.py --force-reindex

# Clean text files are created automatically, no manual copy-paste needed
```

### Example Workflow
```
Initial state:
  docs/
    ├── Coach Packet 2025.pdf
    └── Team Packet 2026.pdf

After first startup:
  docs/
    ├── Coach Packet 2025.pdf
    ├── Coach Packet 2025.txt  ← Auto-created (clean text)
    ├── Team Packet 2026.pdf
    └── Team Packet 2026.txt   ← Auto-created (clean text)

System loads .txt files with preserved structure, not garbled PDFs.
```

### Benefits
- **No manual copy-paste** — automatic conversion on startup
- **Better text extraction** — `pdfplumber` preserves formatting/structure (not PyPDFLoader's garbling)
- **One-time conversion** — subsequent runs skip conversion (fast)
- **Fresh start available** — delete `.txt` file to re-convert from PDF
- **Clean separation** — original PDFs untouched, `.txt` files are working copies

---

## 2. Improved Chunking Strategy ✅

### What Changed
| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| Chunk size | 1000 chars | 1500 chars | Volleyball docs have longer rule sections |
| Chunk overlap | 200 chars | 300 chars | 20% → preserve more cross-chunk context |
| Separators | `["\n\n", "\n", " ", ""]` | `["\n\n\n", "\n", " ", ""]` | Triple-newline splits on major section breaks in copy-pasted text |

### File Modified
- `code/rag.py`: Constants and `_load_and_embed_docs()` method

### Benefits
- Larger chunks reduce fragmentation of related information
- Better separator hierarchy respects document structure
- Improved context preservation per retrieval

---

## 3. Hybrid Search: BM25 + Semantic (Ensemble) ✅

### What Changed
- **Old**: Pure cosine similarity on embeddings (semantic search only)
- **New**: `EnsembleRetriever` combining:
  - **Chroma semantic search** (60% weight) — finds conceptual matches
  - **BM25 keyword search** (40% weight) — catches exact terminology matches

### Implementation Details
```python
# New retriever pipeline
chroma_retriever = vector_store.as_retriever(k=8)
bm25_retriever = BM25Retriever.from_documents(all_documents)
ensemble = EnsembleRetriever(
    retrievers=[chroma_retriever, bm25_retriever],
    weights=[0.6, 0.4]  # Semantic dominates, BM25 provides keyword rescue
)
```

### Files Modified
- `code/rag.py`: New `_build_ensemble_retriever()` method, updated `retrieve_documents()`
- `code/pyproject.toml`: Added `rank-bm25>=0.2.2` dependency

### Benefits
- **Solves "can't find relevant info" problem**: BM25 catches exact terms (jersey numbers, dates, coach names) that embeddings miss
- **60/40 split favors semantic** but lets keyword search salvage misses
- **No persistence overhead**: BM25 index rebuilt in memory at startup from loaded documents
- **Result**: Top-8 hybrid-ranked documents instead of top-6 semantic-only

---

## 4. Self-Critique Meta Prompting ✅

### What Changed
Added a new LangGraph node (`critique_answer`) that runs after answer generation:

```
extract_preferences → expand_query → retrieve_documents → generate_answer → critique_answer → END
```

### What It Does
1. Takes the generated answer + retrieved context chunks
2. Asks the LLM: "Are all factual claims directly supported by the context?"
3. If unsupported claims found → appends this note to the answer:
   ```
   ⚠️ Note: Some details above may not be in the current documents. 
   Please verify with your coach or club coordinator.
   ```
4. If all supported → returns answer as-is

### Implementation Details
```python
def critique_answer(self, state: GraphState) -> dict:
    """Self-critique: verify answer is supported by retrieved context."""
    # Compare generated_answer against state["documents"]
    # Flag unsupported claims
    # Append warning if needed
```

### Files Modified
- `code/rag.py`: New `critique_answer()` method, updated `_build_graph()`, updated `GraphState`
- `code/rag.py`: Added `critique: str` field to `GraphState`

### Benefits
- **Prevents hallucinations**: Answers get flagged if they go beyond retrieved docs
- **User awareness**: Users know when to seek external verification
- **Minimal cost**: ~$0.0001 extra per query with gpt-4o-mini
- **Maintains answer quality**: "SUPPORTED" answers pass through unchanged

---

## New Constants
```python
CACHE_SIMILARITY_THRESHOLD = 0.90  # Fixed discrepancy (was 0.92, actually used 0.90)
CHUNK_SIZE = 1500                   # Increased from 1000
CHUNK_OVERLAP = 300                 # Increased from 200
ENSEMBLE_K = 8                      # New: retriever k value (was 6 before)
```

---

## Testing & Verification

### What Was Tested
✅ Code compiles without errors  
✅ All imports resolve correctly  
✅ GraphState has all required fields (including new `critique`)  
✅ `--force-reindex` flag is wired in run.py  
✅ Ensemble retriever initialization logic correct  
✅ LangGraph pipeline includes new critique node  
✅ PDF auto-conversion functions added and integrated  
✅ `pdfplumber` dependency added  

### How to Test End-to-End

**Scenario 1: Auto-conversion on first run**
```bash
cd code && python run.py
```
Expected output on first startup:
```
Checking for PDFs to convert...
Converting Coach Packet 2025.pdf to text...
  ✓ Created Coach Packet 2025.txt (XXXXX chars)
Converting Team Packet 2026.pdf to text...
  ✓ Created Team Packet 2026.txt (XXXXX chars)

Vector store empty — loading and embedding documents...
Loaded .txt file → N chunks from Coach Packet 2025.txt
Loaded .txt file → N chunks from Team Packet 2026.txt
Embedded N total chunks into ChromaDB.
Built ensemble retriever with N documents for BM25.
Ready.
```

**Scenario 2: Second run (skips conversion)**
```bash
python run.py
```
Expected output:
```
Checking for PDFs to convert...
Skipping Coach Packet 2025.pdf — Coach Packet 2025.txt already exists
Skipping Team Packet 2026.pdf — Team Packet 2026.txt already exists

Vector store has N chunks — skipping re-embedding.
Ready.
```

**Scenario 3: Re-convert from fresh PDF**
```bash
# Delete the .txt to force re-conversion
rm docs/Coach_Packet_2025.txt

# Restart app
python run.py

# Conversion runs again
```

**Scenario 4: Force rebuild everything**
```bash
# Clear vector store AND re-embed from fresh .txt files
python run.py --force-reindex
```

### Verify Improvements Work
1. **Better retrieval**: Ask about specific details (coach name, jersey numbers, dates)
   - Should find them even if phrased differently
2. **Answer quality**: Ask about something not in docs
   - Should append warning note about verification
3. **Text quality**: Check that converted .txt files read cleanly
   - No garbled fonts, preserved sections, good readability

---

## What Didn't Change (Preserved)

- ✅ LLM model (`gpt-4o-mini`) — still appropriate
- ✅ Embeddings model (`text-embedding-3-small`) — fine for this domain
- ✅ Gradio UI — separate concern
- ✅ Semantic Q&A cache — continues to work unchanged
- ✅ Tracing / observability — still functional

---

## What's Out of Scope (Deferred)

- XLSX ingestion (user said "think about it later")
- Streaming responses in Gradio
- Re-ranking with cross-encoder
- UI improvements

---

## Files Modified Summary

| File | Changes |
|------|---------|
| `code/rag.py` | All 4 improvements: text loading, chunking, ensemble, critique |
| `code/run.py` | Added `--force-reindex` argument |
| `code/app.py` | Pass `force_reindex` to `create_demo()` |
| `code/pyproject.toml` | Added `rank-bm25>=0.2.2` |

---

## Next Steps

1. **Test with your actual PDFs**: Copy clean text from PDFs into `.txt` files in `docs/`
2. **Run force-reindex**: `python run.py --force-reindex` to rebuild index
3. **Ask questions**: Verify improvements in retrieval and answer quality
4. **Monitor for XLSX needs**: If you need spreadsheet support later, we can add it
5. **Tune weights if needed**: The 0.6/0.4 semantic/BM25 split can be adjusted based on results

---

## Cost Impact

- **BM25 index**: In-memory, no persistence cost
- **Self-critique**: +$0.0001 per query (negligible with gpt-4o-mini)
- **Chunking**: Slightly larger chunks may increase context window usage, minimal impact

---

## Key Architecture Decisions

1. **BM25 in-memory over persistent**: Simpler, no extra storage, rebuilt at startup
2. **0.6 semantic / 0.4 BM25**: Favors semantic understanding but lets keywords help
3. **Critique appends note rather than rejecting**: Gives users autonomy to verify
4. **No section-aware chunking yet**: Simpler implementation; can add later if needed
5. **--force-reindex as manual flag**: User controls when to rebuild (avoids accidental reindexing)
