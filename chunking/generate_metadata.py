#!/usr/bin/env python3

import json
import pathlib
import re
from datetime import datetime

ROOT = pathlib.Path.home() / "Desktop" / "rag_local" / "docs"

EXT_MAP = {
    ".md": "markdown",
    ".rst": "rst",
    ".html": "html",
    ".htm": "html",
    ".xml": "xml",
    ".txt": "txt",
    ".texi": "texi",
}

MAX_READ = 64_000  # bytes


def read_first_k(path: pathlib.Path, k: int = MAX_READ) -> str:
    try:
        with path.open("rb") as f:
            return f.read(k).decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_title(doc_type: str, text: str) -> str | None:
    if not text:
        return None

    if doc_type == "markdown":
        m = re.search(r"^\s*#\s+(.+)$", text, re.M)
        return m.group(1).strip() if m else None

    if doc_type == "rst":
        lines = text.splitlines()
        for i in range(len(lines) - 1):
            if lines[i].strip() and re.fullmatch(r"[=~\-^`#*+]+", lines[i + 1].strip()):
                return lines[i].strip()
        return None

    if doc_type == "html":
        m = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
        return None

    if doc_type == "xml":
        m = re.search(r"<refname>(.*?)</refname>", text, re.I | re.S)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
        m = re.search(r"<title>(.*?)</title>", text, re.I | re.S)
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
        return None

    return None


def main():
    if not ROOT.exists():
        raise RuntimeError(f"Docs root does not exist: {ROOT}")

    count = 0

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        doc_type = EXT_MAP.get(ext)
        if not doc_type:
            continue

        text = read_first_k(path)
        title = extract_title(doc_type, text)

        stat = path.stat()
        rel_path = path.relative_to(ROOT)
        project = rel_path.parts[0]

        meta = {
            "project": project,
            "path": str(rel_path),
            "filename": path.name,
            "extension": ext,
            "doc_type": doc_type,
            "size_bytes": stat.st_size,
            "mtime_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "title": title,
        }

        out_path = path.with_suffix(path.suffix + ".meta.json")
        out_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        count += 1

    print(f"Metadata written for {count} documents.")


if __name__ == "__main__":
    main()
