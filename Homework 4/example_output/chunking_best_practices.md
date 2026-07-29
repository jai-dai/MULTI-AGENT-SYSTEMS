# Best Practices for Chunking in Retrieval-Augmented Generation (RAG)

This report explores best practices for chunking in Retrieval-Augmented Generation (RAG) systems, focusing on how to optimally split documents into chunks for effective retrieval and generation by large language models (LLMs). The findings synthesize insights from recent expert articles and technical blogs published in 2024-2026.

## 1. Importance of Chunking in RAG

Chunking is the process of breaking down large documents into smaller, manageable pieces called chunks. This step is crucial because LLMs have limited context windows and embedding models have token limits. Proper chunking ensures that each chunk is semantically coherent and fits within these limits, enabling precise retrieval and high-quality generation.

Key reasons chunking matters:
- Improves retrieval accuracy by creating focused chunks that match queries precisely.
- Preserves context for generation, avoiding chunks that are too small (lack context) or too large (dilute relevance).
- Reduces hallucinations by grounding LLM responses in relevant, factual data.
- Enhances efficiency and reduces computational cost.

## 2. Chunking Strategies

### 2.1 Recursive Character Splitting
- Splits text at natural language boundaries using a hierarchy of separators: paragraphs (double newlines), lines (single newlines), sentences, and words.
- Adapts chunk size to respect these boundaries, avoiding mid-sentence or mid-word splits.
- Default chunk size around 400-512 tokens with 10-20% overlap is recommended.
- Suitable for most text types: articles, technical docs, research papers.
- Pros: preserves document structure, better context retention, reliable recall performance.
- Cons: more complex setup, variable chunk sizes.

### 2.2 Size-Based Chunking
- Splits text by fixed character or token counts, optionally with overlap.
- Simple and fast but ignores document structure.
- Token-based splitting aligns better with embedding model limits.
- May split sentences or ideas abruptly.

### 2.3 Semantic Chunking
- Uses semantic similarity to split text into meaningfully coherent chunks.
- Improves recall by up to 9% compared to simpler methods.
- More computationally expensive due to embedding every sentence and calculating similarity.

### 2.4 Page-Level Chunking
- Splits documents by page boundaries.
- Achieved highest accuracy in NVIDIA's 2024 benchmarks for paginated documents.
- Limited to documents with clear page structure.

### 2.5 LLM-Based Chunking
- Uses LLMs to analyze document structure and create chunks.
- High quality but costly and slower.

### 2.6 Sentence-Based and Late Chunking
- Sentence-based respects sentence boundaries but may still miss larger context.
- Late chunking performs chunking at query time on retrieved documents, allowing dynamic chunking but with added latency.

## 3. Trade-Offs and Considerations

- Chunk size: balance between precision and context. Too small loses context; too large dilutes relevance and burdens LLM attention.
- Document type: PDFs, code, and structured documents may require custom chunking strategies.
- Embedding model limits: chunk size must not exceed model token limits (typically up to 8K tokens).
- Cost: semantic and LLM-based chunking are more expensive.
- Use overlap between chunks (e.g., 10-20%) to preserve context across boundaries.

## 4. Practical Recommendations

- Start with recursive character splitting at 400-512 tokens with overlap for general use.
- Use semantic chunking if recall is critical and budget allows.
- For paginated documents, consider page-level chunking.
- Customize separators for domain-specific documents (e.g., code functions, section headers).
- Consider post-chunking for dynamic, query-aware chunking if infrastructure supports it.
- Preprocess PDFs and scanned documents to clean, structured text before chunking.

## 5. Conclusions

Effective chunking is foundational to building high-performing RAG systems. The choice of chunking strategy impacts retrieval accuracy, generation quality, computational cost, and system responsiveness. Recursive character splitting offers a strong default balance of structure awareness and simplicity. Semantic and LLM-based chunking can boost recall and context fidelity but at higher cost. Document type and use case specifics should guide customization of chunking approaches. Overlap between chunks helps maintain context continuity. Finally, preprocessing complex formats like PDFs is essential for reliable chunking.

By carefully selecting and tuning chunking methods, developers can significantly improve the user experience and effectiveness of RAG applications.

## Sources

1. https://www.firecrawl.dev/blog/best-chunking-strategies-rag
2. https://weaviate.io/blog/chunking-strategies-for-rag
3. https://unstructured.io/blog/chunking-for-rag-best-practices
4. https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase

---

This report consolidates current best practices and trade-offs for chunking in RAG systems to guide developers in optimizing retrieval and generation performance.