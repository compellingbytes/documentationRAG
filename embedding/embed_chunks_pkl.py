# embed_simple.py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU only

import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys
import time

def main():
    # Get input file
    if len(sys.argv) > 1:
        chunks_file = sys.argv[1]
    else:
        print("Usage: python embed_simple.py <chunks.jsonl>")
        sys.exit(1)

    output_file = "embeddings.pkl"

    print(f"Loading chunks from {chunks_file}...")

    # Load all chunks
    texts = []
    metadatas = []

    with open(chunks_file, 'r') as f:
        for line in tqdm(f):
            chunk = json.loads(line)
            texts.append(chunk['text'])

            # Save ALL metadata except 'text'
            metadata = {k: v for k, v in chunk.items() if k != 'text'}
            metadatas.append(metadata)

    print(f"Loaded {len(texts)} chunks")

    # Load model and embed
    print("\nLoading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Creating embeddings...")
    start = time.time()
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True)
    elapsed = time.time() - start

    print(f"Done in {elapsed:.2f}s ({len(texts)/elapsed:.1f} chunks/sec)")

    # Save everything
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump({
            'embeddings': embeddings.astype(np.float16),
            'texts': texts,
            'metadatas': metadatas
        }, f)

    print(f"\n✅ Done!")
    print(f"   Saved {len(embeddings)} embeddings")
    print(f"   File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")

    # Show metadata fields
    if metadatas:
        print(f"\nMetadata fields saved:")
        for key in metadatas[0].keys():
            print(f"   - {key}")

if __name__ == "__main__":
    main()
