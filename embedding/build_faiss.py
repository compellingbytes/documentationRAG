# build_faiss.py
import pickle
import numpy as np
import faiss
import time
import os

print("🔧 BUILDING FAISS INDEX")
print("=" * 60)

# 1. Load embeddings
print("📥 Loading embeddings...")
load_start = time.time()
with open('embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

embeddings = data['embeddings'].astype(np.float32)  # FAISS needs float32
load_time = time.time() - load_start

print(f"✅ Loaded {embeddings.shape[0]:,} embeddings")
print(f"   Dimensions: {embeddings.shape[1]}")
print(f"   Load time: {load_time:.1f}s")
print(f"   Metadata fields: {list(data['metadatas'][0].keys())}")

# 2. Normalize for cosine similarity
print("\n📐 Normalizing embeddings...")
norm_start = time.time()
faiss.normalize_L2(embeddings)
norm_time = time.time() - norm_start
print(f"✅ Normalized in {norm_time:.1f}s")

# 3. Create FAISS index
print("\n🔨 Creating FAISS index...")
index_start = time.time()
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine similarity

# Add in batches to monitor progress
batch_size = 50000
for i in range(0, len(embeddings), batch_size):
    batch = embeddings[i:i + batch_size]
    index.add(batch)
    print(f"   Added {min(i + batch_size, len(embeddings)):,} / {len(embeddings):,} vectors")

index_time = time.time() - index_start
print(f"✅ Index created with {index.ntotal:,} vectors")
print(f"   Index time: {index_time:.1f}s")

# 4. Save index
print("\n💾 Saving index...")
faiss.write_index(index, 'faiss_index.bin')

# Save metadata mapping
import json
metadata_info = {
    'total_vectors': index.ntotal,
    'dimension': dimension,
    'index_type': 'IndexFlatIP',
    'metric': 'cosine_similarity',
    'model': 'all-MiniLM-L6-v2',
    'embedding_file': 'embeddings.pkl',
    'created_at': time.time(),
    'chunk_count': len(data['texts']),
    'metadata_fields': list(data['metadatas'][0].keys())
}

with open('faiss_index_meta.json', 'w') as f:
    json.dump(metadata_info, f, indent=2)

print(f"✅ Index saved: faiss_index.bin")
print(f"   Metadata saved: faiss_index_meta.json")

# 5. Test the index
print("\n🧪 Testing index with sample query...")
# Quick test
test_vector = np.random.randn(1, dimension).astype(np.float32)
faiss.normalize_L2(test_vector)

k = 3
distances, indices = index.search(test_vector, k)
print(f"   Sample query returned {k} results")
print(f"   Indices: {indices[0]}")
print(f"   Similarities: {distances[0]}")

# 6. Stats
print("\n📊 FINAL STATISTICS")
print("=" * 60)
print(f"Total chunks: {len(data['texts']):,}")
print(f"FAISS vectors: {index.ntotal:,}")
print(f"Embedding dimension: {dimension}")
print(f"Index size on disk: {os.path.getsize('faiss_index.bin') / 1024 / 1024:.1f} MB")
print(f"Total processing time: {load_time + norm_time + index_time:.1f}s")
print(f"\n✅ READY FOR RAG QUERIES!")
