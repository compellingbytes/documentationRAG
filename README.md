# documentationRag

A modular, FAISS‑based RAG pipeline for technical documentation.  
Built to benchmark cheap GPUs (P102‑100, RX 6700, at least initially) but should work on anything.

---

## What it does

- **Chunking** – Splits docs into semantic chunks, preserves `section_path`, `heading`, `project`.
- **Embedding** – Converts chunks to vectors (`.npy` + `.jsonl` or single `.pkl`).
- **Retrieval** – FAISS search with full metadata, ready for LLM integration.
- **LLM** – Works with `llama.cpp` server (CUDA / Vulkan) for generation.

---

## Repository structure

```
documentationRag/
├── chunking/               # raw docs → chunks.jsonl
├── embedding/              # chunks.jsonl → vectors + metadata
├── retrieval/              # query → LLM answer
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

### 3. Retrieve + generate

```bash
cd ../retrieval
python rag_llm_integration.py
```

---

## Two embedders

| Script | Output | Best for |
|--------|--------|----------|
| `embed_chunks_pkl.py` | `embeddings.pkl` | pickle‑based retrieval (`test_rag_system.py`) |
| `embed_chunks_jsonl.py` | `.npy` + `.jsonl` | `query.py`, FAISS, Chroma |

Both read `chunks.jsonl` and preserve all metadata.
Theres an additional reademe in the `embedding` directory.

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

- NVidia P102‑100 (Pascal mining card, 10GB, CUDA)
- AMD RX 6700 / Sapphire GPRO X080 (RDNA2 mining card, 10GB, Vulkan)

---

## License

MIT
