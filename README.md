# RAG-based Chatbot (FastAPI + WebSocket Streaming)

A document-aware chatbot built with FastAPI. It uses a Retrieval-Augmented Generation (RAG) pipeline with LangChain + Chroma to answer questions grounded on a local Markdown knowledge base, and supports real-time streaming responses via WebSocket. It also includes an image generation demo page.

## Features
- Real-time streaming chat using WebSocket (token/word-by-word streaming).
- RAG pipeline: load Markdown docs → split into chunks → embed → store in Chroma → retrieve relevant chunks → generate grounded answers.
- Conversational memory (chat history) for multi-turn conversations.
- Simple web UI (HTML/CSS/Bootstrap) + optional image generation page.

## How RAG Works

- Ingest documents from knowledge-base/
- Chunk text into smaller passages
- Embed chunks into vectors
- Store vectors in Chroma
- On each question: retrieve top relevant chunks
- Send context + user question to the LLM to generate a grounded answer
- Stream the response back to the UI via WebSocket

<img width="700" height="500" alt="RAG" src="https://github.com/user-attachments/assets/7b6dea91-cb94-4f8c-bd8f-4ee54edd56d1" />

## Demo

https://github.com/user-attachments/assets/6eb29748-4dd0-47e6-895f-578d653fcae2

## Limitations

- Local/dev setup (no authentication, rate limiting, or production deployment by default).
- Retrieval quality depends on chunking strategy, embeddings, and document quality.

## Future Work

- Add evaluation (retrieval metrics), better chunking (hybrid semantic + rules), and reranking.
- Docker + cloud deployment, auth, monitoring/logging.
- Support more file types (PDF/DOCX) and larger knowledge bases.
