#!/usr/bin/env python3
import sys
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def main():
    if len(sys.argv) < 2:
        print("Usage: python embed.py <chunks.jsonl> [outdir]")
        sys.exit(1)

    chunks_file = Path(sys.argv[1])
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./index")
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading chunks from {chunks_file}...")
    records = []
    texts = []
    with open(chunks_file, 'r') as f:
        for line in tqdm(f):
            rec = json.loads(line)
            if rec.get('text', '').strip():
                records.append(rec)
                texts.append(rec['text'])

    print(f"Loaded {len(texts)} chunks")
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Save
    emb_path = outdir / "embeddings.npy"
    meta_path = outdir / "embeddings_meta.jsonl"

    np.save(emb_path, embeddings)
    with open(meta_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')

    print(f"\nSaved:")
    print(f"  {emb_path}  shape={embeddings.shape}")
    print(f"  {meta_path}")

if __name__ == "__main__":
    main()
