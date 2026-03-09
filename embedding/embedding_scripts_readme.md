# Embedding Scripts

Two embedders are provided. Both read `chunks.jsonl` and output vector + metadata files. Choose the one that fits your workflow:

---

## `embed_chunks_pkl.py` (original simple version)

**Output:** Single pickle file `embeddings.pkl` containing:

```python
{
    'embeddings': np.array(...),        # float16 vectors
    'texts': [...],                      # chunk texts
    'metadatas': [...]                   # full metadata per chunk
}
```

**Usage:**
```bash
python embed_chunks_pkl.py chunks.jsonl
```
- First argument: path to `chunks.jsonl`
- Output is always `embeddings.pkl` in current directory

**Best for:** Local testing, staying in Python, smaller file size (float16)

---

## `embed_chunks_jsonl.py` (standard pipeline version)

**Output:** Two files in `./index/` (or custom directory)

- `embeddings.npy` – vectors (float32)
- `embeddings_meta.jsonl` – metadata (one JSON line per chunk, aligned with vectors)

**Usage:**
```bash
python embed_chunks_jsonl.py chunks.jsonl
python embed_chunks_jsonl.py chunks.jsonl ./my_index   # optional output dir
```
- First argument: path to `chunks.jsonl`
- Second argument (optional): output directory (defaults to `./index`)

**Best for:** Sharing, cross‑language use, compatibility with `query.py` and other tools

---

## Key differences

|                         | `embed_chunks_pkl.py`          | `embed_chunks_jsonl.py`               |
|-------------------------|--------------------------------|----------------------------------------|
| Output files            | `embeddings.pkl`               | `embeddings.npy` + `embeddings_meta.jsonl` |
| Vector dtype            | float16 (smaller)              | float32 (standard)                     |
| Metadata                | Inside pickle                  | Separate JSONL (human‑readable)        |
| Language                | Python only                    | Any language                           |
| Use with                | `test_rag_system.py`, `rag_llm_integration.py` | `query.py`, FAISS, Chroma, etc. |

---

## How `sys.argv` works in both

Both scripts use the same simple argument pattern:

```python
import sys

# First argument is required: chunks.jsonl
chunks_file = sys.argv[1]

# Second argument is optional (only in jsonl version)
if len(sys.argv) > 2:
    outdir = sys.argv[2]
else:
    outdir = "./index"
```

- `sys.argv[0]` is always the script name (ignored)
- `sys.argv[1]` is the first thing you type after the script
- `sys.argv[2]` is the second thing (if present)

**Example:**
```bash
python embed_chunks_jsonl.py chunks.jsonl custom_folder
```
- `sys.argv[1]` = `"chunks.jsonl"`
- `sys.argv[2]` = `"custom_folder"`
