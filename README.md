# documentationRag

#### built iteratively with LLM assistance

A modular, FAISS‑based RAG pipeline for technical documentation.  
Built to benchmark cheap GPUs (P102‑100, RX 6700) but should work on anything.

---

## What it does

- **Chunking** – Splits docs into semantic chunks, preserves `section_path`, `heading`, `project`.
- **Embedding** – Converts chunks to vectors (`.npy` + `.jsonl` or single `.pkl`).
- **Retrieval** – FAISS search with full metadata, ready for LLM integration.
- **LLM** – Works with `llama.cpp` server via OpenAI-compatible `/v1/chat/completions` endpoint.

---

## Repository structure

```
documentationRag/
├── chunking/               # raw docs → chunks.jsonl
├── embedding/              # chunks.jsonl → vectors + metadata
├── retrieval/              # query → LLM answer
│   └── rag_llm_integration.py   # main RAG script
├── archival/               # old / experimental scripts
├── requirements.txt
└── README.md
```

---

## Quick start

```bash
git clone https://github.com/compellingbytes/documentationRag
cd documentationRag
pip install -r requirements.txt
```

### 1. Get a pre‑chunked corpus

Download `chunks.jsonl` from [Hugging Face](https://huggingface.co/datasets/#tbd)  
Place it in the project root.

### 2. Embed

```bash
cd embedding
python embed_chunks_jsonl.py ../chunks.jsonl ../index
```

### 3. Start llama.cpp server (OpenAI-compatible)

```bash
llama-server -m /path/to/model.gguf --port 8080 -c 8192
```

### 4. Run RAG queries

```bash
cd ../retrieval
python rag_llm_integration.py
```

---

## `rag_llm_integration.py`

The main RAG script with:

- **OpenAI-compatible `/v1/chat/completions` endpoint** (thinking disabled for Qwen 3.5)
- **Project detection** via regex (handles "llama.cpp" vs "llamacpp")
- **Three prompt modes** (controlled by first argument)
- **Two query sets** (vague vs refined, via `--query-set` flag)
- **Streaming responses** with token-by-token output
- **File logging** to `rag_run_TIMESTAMP.txt` with full metadata

### Prompt Modes

| Mode | Behavior |
|------|----------|
| `default` | Use retrieved documents first, fall back to model knowledge if needed. Cite sources, say "I don't know" if docs missing. |
| `corrective` | Compare documents against model knowledge, correct errors in the docs. |
| `override` | Trust model knowledge over documents when they conflict. |

### Query Sets

| Set | Description |
|-----|-------------|
| `vague` | Original, underspecified questions (e.g., "How to convert video to mp4 with ffmpeg?") |
| `refined` | Specific, detailed questions with context (e.g., includes `.mov`, h.264, audio handling) |

### Usage

```bash
# Default: default prompt mode, vague queries
python rag_llm_integration.py

# Override mode with vague queries
python rag_llm_integration.py override

# Default mode with refined queries
python rag_llm_integration.py default --query-set refined

# Override mode with refined queries
python rag_llm_integration.py override --query-set refined

# Corrective mode with refined queries
python rag_llm_integration.py corrective --query-set refined
```

### Output

- **Terminal**: Streaming tokens, search time, LLM time, context usage bar
- **File**: `rag_run_YYYYMMDD_HHMMSS.txt` with full query, timing, and response

---

## Two embedders

| Script | Output | Best for |
|--------|--------|----------|
| `embed_chunks_pkl.py` | `embeddings.pkl` | pickle‑based retrieval (`test_rag_system.py`) |
| `embed_chunks_jsonl.py` | `.npy` + `.jsonl` | `query.py`, FAISS, Chroma |

Both read `chunks.jsonl` and preserve all metadata.  
See `embedding/README.md` for details.

---

## Metadata

Each chunk contains:

```json
{
  "project": "podman",
  "rel_path": "podman/rootless.md",
  "section_path": ["Podman", "Rootless", "User Actions", "Using volumes"],
  "heading": "Using volumes",
  "text": "..."
}
```

This lets you filter, cite, and group results precisely.

---

## Hardware tested

- NVIDIA P102‑100 (Pascal mining card, 10GB, CUDA)
- AMD RX 6700 / Sapphire GPRO X080 (RDNA2, 10GB, Vulkan)

---

## License

MIT
