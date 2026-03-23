# rag_llm_integration.py
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
import pickle
import re
import sys
import time
from datetime import datetime

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
        parts = re.split(r"[._-]", proj)
        if len(parts) > 1:
            pattern = r"\s*".join(re.escape(p) for p in parts)
            pattern = pattern.replace(r"\.", r"\.?")
        else:
            pattern = re.escape(proj)
        if re.search(rf"\b{pattern}\b", query_lower):
            return proj
    return None


class FixedRAG:
    def __init__(self):
        print("🚀 FIXED RAG SYSTEM")
        print("=" * 60)
        with open("embeddings.pkl", "rb") as f:
            self.data = pickle.load(f)
        self.index = faiss.read_index("faiss_index.bin")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model.to("cpu")
        try:
            props = requests.get("http://localhost:8080/props", timeout=5).json()
            self.max_context = props.get("ctx_size", 8192)
            self.model_name = props.get("model_name", "unknown")
            print(f"📊 LLM model: {self.model_name}")
            print(f"📊 LLM context size: {self.max_context:,} tokens")
        except:
            self.max_context = 8192
            self.model_name = "unknown"
            print("⚠️  Could not query LLM server, assuming defaults")
        print(f"✅ Loaded {self.index.ntotal:,} documents")

    def update_status_bar(self, current_tokens):
        percent = min(100, int((current_tokens / self.max_context) * 100))
        bar_length = 20
        filled = int(bar_length * percent / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        status = f"\r\033[33mContext: [{bar}] {percent}% ({current_tokens}/{self.max_context} tokens)\033[0m\n"
        sys.stdout.write(status)
        sys.stdout.flush()

    def query(self, question, k=12, mode="default", output_file=None):
        print("\n" + "─" * 58)
        print(f"🔍 QUERY: {question}")
        print(f"📌 MODE: {mode}")
        print("─" * 58)

        project = detect_project(question)
        enhanced_question = f"{question} {project}" if project else question

        start = time.time()
        query_vec = self.model.encode([enhanced_question], device="cpu").astype(
            np.float32
        )
        faiss.normalize_L2(query_vec)
        distances, indices = self.index.search(query_vec, k)
        search_time = time.time() - start
        print(f"⚡ Search: {search_time * 1000:.0f}ms")

        context_parts = []
        for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
            meta = self.data["metadatas"][idx]
            text = self.data["texts"][idx]
            excerpt = f"--- Excerpt {i + 1} ---\n"
            if meta.get("project"):
                excerpt += f"From: {meta['project']}\n"
            if meta.get("heading"):
                excerpt += f"Section: {meta['heading']}\n"
            excerpt += f"Relevance: {score:.3f}\n\n{text}\n"
            context_parts.append(excerpt)

        context = "\n".join(context_parts)
        context_tokens = int(len(context.split()) * 1.3)
        self.update_status_bar(context_tokens)

        if mode == "default":
            system = """You are a technical documentation assistant. Use the provided document excerpts to answer questions.
If the documents contain the answer, quote or summarize from them. Synthesize a coherent, concise response.
If the documentation does not return any information related to the question, say you do not know the answer.
Cite where in the documentation you got the information.
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

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"DOCUMENT EXCERPTS:\n{context}\n\nQUESTION: {question}",
            },
        ]

        if mode == "override":
            timeout = 180
        elif mode == "corrective":
            timeout = 150
        else:
            timeout = 120

        payload = {
            "messages": messages,
            "max_tokens": 1800,
            "stream": True,
            "temperature": 0.3,
            "top_p": 0.85,
            "top_k": 20,
            "presence_penalty": 0.0,
            "repeat_penalty": 1.0,
        }

        if (
            "qwen3.5" in self.model_name.lower()
            or "qwen-3.5" in self.model_name.lower()
        ):
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        print(f"🤖 Querying LLM...")
        llm_start = time.time()
        full_response = []

        try:
            with requests.post(
                "http://localhost:8080/v1/chat/completions",
                json=payload,
                stream=True,
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    if line.startswith("data: "):
                        chunk = line[len("data: ") :]
                        if chunk == "[DONE]":
                            break
                        try:
                            obj = json.loads(chunk)
                            delta = obj.get("choices", [{}])[0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                print(token, end="", flush=True)
                                full_response.append(token)
                        except json.JSONDecodeError:
                            pass
            print()
            llm_time = time.time() - llm_start
            print(f"\n⚡ LLM: {llm_time:.1f}s")

            if output_file:
                full_text = "".join(full_response)
                with open(output_file, "a") as f:
                    f.write(f"QUERY: {question}\n")
                    f.write(f"SEARCH: {search_time * 1000:.0f}ms\n")
                    f.write(f"LLM: {llm_time:.1f}s\n")
                    f.write(f"CONTEXT: {context_tokens}/{self.max_context} tokens\n")
                    f.write("-" * 40 + "\n")
                    f.write(full_text)
                    f.write("\n\n" + "=" * 60 + "\n\n")

            print("--" * 58)
            print("✅ Query complete. Ready for the next one.")
            print("--" * 58 + "\n")

        except Exception as e:
            print(f"❌ Error: {e}")
            print("─" * 58)
            print("✅ Query complete. Ready for next.")
            print("─" * 58 + "\n")

    def test_all(self, mode="default", query_set="vague"):
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"rag_run_{run_timestamp}.txt"

        with open(output_file, "w") as f:
            f.write(f"RAG Test Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Query set: {query_set}\n")
            f.write("=" * 60 + "\n\n")

        vague_queries = [
            "How do I run a podman container in rootless mode?",
            "What is the difference between systemd service and socket?",
            "How to convert video to mp4 with ffmpeg?",
            "What CUDA architecture is compatible with a Pascal GPU?",
        ]

        refined_queries = [
            "How do I run a podman container in rootless mode? (with specific steps for user namespace configuration and networking)",
            "What is the difference between a systemd service and socket? (explain activation behavior and file descriptor passing)",
            "How do you convert a .mov video to .mp4 using the h.264 codec with ffmpeg? (include audio handling and optimization flags)",
            "What is the latest CUDA version that's compatible with a Pascal GPU? (consider the deprecation timeline)",
        ]

        queries = vague_queries if query_set == "vague" else refined_queries
        print("\n" + "═" * 60)
        print(f"📋 {query_set.upper()} QUERIES")
        print("═" * 60)

        for query in queries:
            self.query(query, mode=mode, output_file=output_file)
            if query != queries[-1]:
                input("\nPress Enter for the next query...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Query System")
    parser.add_argument(
        "mode",
        nargs="?",
        default="default",
        choices=["default", "corrective", "override"],
        help="Prompt mode: default, corrective, override",
    )
    parser.add_argument(
        "--query-set",
        choices=["vague", "refined"],
        default="vague",
        help="Query set: vague or refined",
    )

    args = parser.parse_args()
    print(f"🔧 Mode: {args.mode} | Query set: {args.query_set}")

    rag = FixedRAG()
    rag.test_all(mode=args.mode, query_set=args.query_set)
