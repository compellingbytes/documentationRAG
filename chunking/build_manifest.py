#!/usr/bin/env python3

import json
import pathlib
import re
from datetime import datetime

ROOT = pathlib.Path.home() / "Desktop" / "rag_local" / "docs"
OUT = ROOT / "manifest.jsonl"

EXT_MAP = {
    ".md": "markdown",
    ".rst": "rst",
    ".html": "html",
    ".htm": "html",
    ".xml": "xml",
    ".txt": "txt",
    ".texi": "texi",
}

def read_sample(path, limit=64000):
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except Exception:
        return ""

def extract_title(text, kind):
    if not text:
        return None

    if kind == "markdown":
        m = re.search(r"^#\s+(.+)$", text, re.M)
        return m.group(1).strip() if m else None

    if kind == "rst":
        lines = text.splitlines()
        for i in range(len(lines) - 1):
            if lines[i].strip() and re.fullmatch(r"[=~\-^`#*+]+", lines[i+1].strip()):
                return lines[i].strip()
        return None

    if kind in ("html", "xml"):
        m = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
        return re.sub(r"\s+", " ", m.group(1)).strip() if m else None

    if kind == "texi":
        m = re.search(r"@settitle\s+(.+)", text)
        return m.group(1).strip() if m else None

    return None

def main():
    records = []
    count = 0

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        kind = EXT_MAP.get(ext)
        if not kind:
            continue

        rel = path.relative_to(ROOT)
        project = rel.parts[0]

        stat = path.stat()
        text = read_sample(path)
        title = extract_title(text, kind)

        rec = {
            "project": project,
            "rel_path": str(rel),
            "ext": ext,
            "kind": kind,
            "bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "title": title,
        }

        records.append(rec)
        count += 1

    with OUT.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {count} records to {OUT}")

if __name__ == "__main__":
    main()
