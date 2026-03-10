# rag_fixed.py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pathlib
import pickle
import re
import sys
import time

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

PROJECTS = [
    "podman",
    "docker",
    "systemd",
    "ffmpeg",
    "mesa",
    "openvino",
    "llama.cpp",
    "sd.ccp",
    "stablediffusion.cpp",
]


def detect_project(query):
    query_lower = query.lower()

    for proj in PROJECTS:
        # Create flexible pattern:
        # "systemd" -> "system\s*d"
        # "llama.cpp" -> "llama\.?\s*cpp"
        parts = re.split(r"[._-]", proj)
        if len(parts) > 1:
            pattern = r"\s*".join(re.escape(p) for p in parts)
            pattern = pattern.replace(r"\.", r"\.?")  # keeps dot optional
        else:
            pattern = re.escape(proj)

        if re.search(rf"\b{pattern}\b", query_lower):
            return proj

    return None

    # Set mode from command line, default to "default"
    mode = sys.argv[1] if len(sys.argv) > 1 else "default"


class FixedRAG:
    def __init__(self):
        print("🚀 FIXED RAG SYSTEM")
        print("=" * 60)

        with open("embeddings.pkl", "rb") as f:
            self.data = pickle.load(f)

        self.index = faiss.read_index("faiss_index.bin")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model.to("cpu")

        # Get context size from llama.cpp server
        try:
            props = requests.get("http://localhost:8080/props").json()
            self.max_context = props.get("ctx_size", 8192)  # fallback if missing
            print(f"📊 LLM context size: {self.max_context:,} tokens")
        except:
            self.max_context = 8192  # fallback if server unreachable
            print(
                f"⚠️  Could not query LLM server, assuming {self.max_context} token context"
            )

        print(f"✅ Loaded {self.index.ntotal:,} documents")

    def update_status_bar(self, current_tokens):
        """Draw a yellow context usage meter at bottom left"""
        percent = min(100, int((current_tokens / self.max_context) * 100))
        bar_length = 20
        filled = int(bar_length * percent / 100)
        bar = "█" * filled + "░" * (bar_length - filled)

        status = f"\r\033[33mContext: [{bar}] {percent}% ({current_tokens}/{self.max_context} tokens)\033[0m\n"
        sys.stdout.write(status)
        sys.stdout.flush()

    def query(self, question, k=12, mode="default"):
        print("\n" + "─" * 58)
        print(f"🔍 QUERY: {question}")
        print(f"📌 MODE: {mode}")
        print("─" * 58)
        project = detect_project(question)

        # 0. Zoom into project via regex match, if possible
        if project:
            enhanced_question = f"{question} {project}"
        else:
            enhanced_question = question

        # 1. Search
        start = time.time()
        query_vec = self.model.encode([enhanced_question], device="cpu").astype(
            np.float32
        )
        faiss.normalize_L2(query_vec)
        distances, indices = self.index.search(query_vec, k)
        search_time = time.time() - start

        print(f"⚡ Search: {search_time * 1000:.0f}ms")

        # 2. Build context WITHOUT [Source X]
        context_parts = []
        for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
            meta = self.data["metadatas"][idx]
            text = self.data["texts"][idx]

            # Numbered excerpts, no brackets
            excerpt = f"--- Excerpt {i + 1} ---\n"
            if meta.get("project"):
                excerpt += f"From: {meta['project']}\n"
            if meta.get("heading"):
                excerpt += f"Section: {meta['heading']}\n"
            excerpt += f"Relevance: {score:.3f}\n\n"
            excerpt += f"{text}\n"

            context_parts.append(excerpt)

        context = "\n".join(context_parts)

        # Choose prompt based on mode
        if mode == "default":
            system = """You are a technical documentation assistant. Use the provided document excerpts to answer questions.
If the documents contain the answer, quote or summarize from them. Synthesize a coherent, concise response.
If not, use your knowledge but clarify what comes from documents vs your knowledge.
Provide complete, usable commands when applicable."""

        elif mode == "corrective":
            system = """You are a technical documentation assistant with strong technical knowledge.
            Use the provided document excerpts first, but compare them against your own knowledge.
            If the documents contain errors or omit important details, correct them in your answer. Synthesize a coherent, concise response.
            Explain what came from docs vs. what you corrected based on your knowledge."""

        elif mode == "override":
            system = """You are a technical documentation assistant with strong technical knowledge.
            Your knowledge takes precedence over the provided documents. Synthesize a coherent, concise response.
            Use the documents as reference, but if they conflict with your knowledge, trust yourself.
            Explain when you're overriding the documents with your own knowledge."""

        else:
            system = "You are a helpful assistant."

        prompt = f"""<|im_start|>system {system}<|im_end>
        <|im_start|>user
        DOCUMENT EXCERPTS:
        {context}
        QUESTION: {question}<|im_end|>
        <|im_start|>assistant
        """

        # Rest of method (token counting, LLM call) unchanged

        context_tokens = int(len(context.split()) * 1.3)  # rough estimate
        self.update_status_bar(context_tokens)

        # 4. Send to LLM
        payload = {
            "prompt": prompt,
            "n_predict": 800,
            "temperature": 0.1,
            "top_p": 0.9,
            "stop": ["</s>", "Question:", "Excerpt", "\n\n\n"],
        }

        print(f"🤖 Querying LLM...")
        llm_start = time.time()

        try:
            response = requests.post(
                "http://localhost:8080/completion", json=payload, timeout=120
            )
            answer = response.json()["content"].strip()
            llm_time = time.time() - llm_start

            print(f"⚡ LLM: {llm_time:.1f}s")
            print(f"\n💬 Answer:")
            print("-" * 50)
            print(answer)
            print("-" * 50)

        except Exception as e:
            print(f"❌ Error: {e}")

        print("─" * 58)
        print("✅ Query complete. Ready for next.")
        print("─" * 58 + "\n")
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    def test_all(self):
        """Test the fixed system"""
        test_queries = [
            "How do I run a podman container in rootless mode?",
            "What is the difference between a systemd service and socket?",
            "How do you convert a .mov video to .mp4 (h.264 codec) with ffmpeg?",
            "What CUDA version is compatible with a Pascal GPU?",
        ]

        for query in test_queries:
            self.query(query)
            if query != test_queries[-1]:
                input("\nPress Enter for the next query...")


if __name__ == "__main__":
    rag = FixedRAG()
    rag.test_all()
