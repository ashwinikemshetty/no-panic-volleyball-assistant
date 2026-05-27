Before You Start

It's very hard to predict upfront which optimizations your RAG system will need. Most of these decisions become clear only after you see real data and real failures. For this capstone, treat the sections below as a menu of ideas: pick what makes sense given your domain knowledge and use case. Everything might change based on what you see with actual data, but having a proposal of what you'd try and why is the exercise.

If you have domain expertise, you can sometimes anticipate issues. For example:

If your knowledge base uses a lot of jargon or synonyms (e.g., medical, legal), you can predict that semantic search alone won't cut it and hybrid search will likely help

If your documents are long, dense, and have rigid structures (e.g., instruction manuals, research papers), semantic chunking is worth considering so you preserve logical sections

If your queries require pulling information where one source depends on another (e.g., "what's the return policy for the product I bought last week?" requires first finding the order, then finding the return policy for that product category), you're looking at multi-hop retrieval

If your data is mostly tabular (e.g., financial reports, inventory), RAG may not be the right approach at all: consider text-to-SQL instead

RAG Quick Refresher

RAG has three phases: (1) Ingestion: parse your documents, split them into chunks, generate embeddings, and store them in a vector database. This is done offline, before users interact with the system. (2) Retrieval: when a user query comes in, convert it to an embedding, find the most similar chunks from the vector database, and return them. (3) Generation: pass the user query plus retrieved chunks to the LLM, which synthesizes a final response.

Design Decisions When Setting Up RAG

Chunking

There are several chunking strategies, and the right one depends on your data:

Fixed-length: chunk every N words. Simple but may break context mid-sentence.

Sentence-based: chunk after every N sentences. Same context-breaking risk.

Paragraph-based: preserves logical sections but chunks vary in size.

Semantic chunking: a smaller model decides where to split so context is preserved. Best for code, research papers, and technical documentation.

Hybrid: mix of the above.

Rule of thumb: chunk size should be roughly 5-10x the size of your expected answer. If answers are 1-2 words, chunks can be 50-100 tokens. If answers need multi-sentence explanations, aim for 300-400 tokens. This isn't an absolute rule — adjust based on what you see.

Exceptions: if answers span multiple documents, smaller chunks may improve diversity. If the task requires reasoning or synthesis, larger chunks help. For code and logs, use semantic chunking rather than token counts.

Common mistake: don't make chunks too small. If chunks are so small they lose context, the retrieval system won't find them even when they're relevant.

Don't chunk tables: embeddings can't capture the semantics of numbers and symbols reliably (89 and 98 look nearly identical in embedding space). For tabular data, use text-to-SQL instead of RAG.

Embedding Models

Your choice often depends on your cloud ecosystem (AWS, Azure, OpenAI each provide their own). Use the MTEB (Massive Text Embedding Benchmark) leaderboard to compare, focusing on retrieval task scores. Consider: number of tokens supported, multilingual support, and embedding dimension size (larger = higher storage cost). Starting with whatever your platform provides is fine.

Vector Databases

Most popular vector databases support similar features. Differentiate only if you have a specific requirement (e.g., hybrid search, geo search, multi-vector support). Use the VectorDB Comparison site (by Superlinked) to compare across dimensions like search types, data types, pricing, and integrations: https://superlinked.com/vector-db-comparison

General advice: start small. Build your RAG pipeline on a small subset of your knowledge base first. Embedding and storing at scale is expensive, so validate your approach before scaling out.

Retrieval Optimization Proposals

About 80% of RAG problems are retrieval issues, not generation issues. You won't know upfront which of these you'll need. But if you think about your use case and your domain, you can often recognize situations where a specific technique would help. Below are common situations and what you could propose for each. You don't have to pick any of these now: they're ideas to have in your back pocket.

Query expansion: Your users might phrase things differently from how your knowledge base is written. A customer says "can I expense this?" but your docs say "reimbursement policy." If you expect that gap between user language and document language, query expansion reformulates vague queries into domain terms, breaks complex queries into sub-queries, and expands with synonyms.

HyDE (Hypothetical Document Embeddings): Your users ask short or vague questions that don't have enough signal to match against long documents. HyDE generates a hypothetical answer first, then uses that answer as the search query. This gives the retriever more context to match against.

Hybrid search: Your knowledge base has exact terms that matter: policy numbers, product IDs, medical codes, variable names. Semantic search alone won't reliably match these. Hybrid search combines keyword search with semantic search so you catch what semantic misses. One reference implementation uses a 70/30 keyword-to-semantic ratio.

Re-ranking with cross-encoder models: You're retrieving a decent pool of results but the most relevant ones aren't always at the top. A cross-encoder passes each chunk along with the query and scores relevance more carefully. Apply this to your top 10-20 results (it's slower than initial retrieval, so you don't run it on everything). Cohere's re-ranker is a strong option.

Corrective RAG: Your knowledge base isn't complete, or you sometimes need the latest information that your docs don't cover yet. If web search can genuinely help fill gaps in your data, corrective RAG builds a post-retrieval filter: an LLM judge checks each chunk for relevance, drops what doesn't make the cut, and can replace weak chunks with web search results.

This list isn't exhaustive. If you have domain knowledge of your space, you can find other optimization techniques here: https://github.com/aishwaryanr/awesome-generative-ai-guide/blob/main/research_updates/rag_research_table.md

When You Need Multi-Hop Retrieval

Some queries require pulling information across multiple documents or connecting different pieces of knowledge. Two approaches:

Graph RAG

Store your knowledge base as a graph: entities as nodes, relationships as edges. Retrieval becomes graph traversal. Best when your data naturally forms a graph (e.g., retail: buyers, sellers, products, reviews) and most queries are multi-hop.

Trade-off: high upfront cost to build and maintain the graph, but low runtime cost per query.

Agentic RAG

The knowledge base becomes a tool that an autonomous agent decides when and how to call. The agent plans its own retrieval: it may decompose a query, call the knowledge base at different steps, rewrite queries, or fall back to web search.

Trade-off: low upfront cost (works with existing vector databases), but higher runtime cost since the model plans on the fly and may iterate many times.

Pitfalls: the agent can make many unnecessary calls, confuse internal knowledge base with web search, and is harder to debug since retrieval is buried inside a multi-step loop.

When to use which: if your data already exists as a graph or you can build one cost-effectively, Graph RAG gives you lower runtime costs. If building a graph is expensive and your queries are diverse, Agentic RAG is more practical. If your queries are not multi-hop, you don't need either: stick with the single-hop solutions above.

Cost Optimization

Once your RAG system is working well, you can optimize for cost and latency.

Semantic caching: 50-80% of queries in enterprise RAG systems are duplicates or near-duplicates. Store question-answer pairs in a vector database. When a new query comes in, check similarity against cached questions first. If there's a match, return the cached answer without calling the LLM.

Prompt caching: model providers cache repetitive prompt prefixes at roughly 1/10th of the cost. Place your system prompt (the repeating part) at the top and variable content (user query, retrieved chunks) at the bottom.

Optimize performance first, then cost. Don't try to reduce costs before you have a working system.