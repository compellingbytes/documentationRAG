#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Iterable, Optional, Tuple, List


DOCS_ROOT_DEFAULT = pathlib.Path.home() / "Desktop" / "rag_local" / "docs"
OUT_DEFAULT = DOCS_ROOT_DEFAULT / "chunks.jsonl"

SUPPORTED_EXTS = {
    ".md": "markdown",
    ".rst": "rst",
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
    ".txt": "txt",
    ".texi": "texi",  # treat as plain text
}

def read_text(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def load_sidecar_meta(p: pathlib.Path) -> Optional[dict]:
    sidecar = p.with_suffix(p.suffix + ".meta.json")
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception:
        return None

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

@dataclass
class Section:
    section_path: List[str]
    text: str

def chunk_paragraphs(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for para in paras:
        if not buf:
            buf = para
        elif len(buf) + 2 + len(para) <= max_chars:
            buf += "\n\n" + para
        else:
            flush()
            buf = para

    flush()

    if overlap_chars > 0 and len(chunks) > 1:
        overlapped: List[str] = []
        prev_tail = ""
        for i, c in enumerate(chunks):
            if i == 0:
                overlapped.append(c)
            else:
                head = (prev_tail + "\n\n" + c) if prev_tail else c
                overlapped.append(head)
            prev_tail = c[-overlap_chars:] if len(c) > overlap_chars else c
        return overlapped

    return chunks

# ---------- Markdown ----------
MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", re.M)

def parse_markdown_sections(text: str) -> List[Section]:
    matches = list(MD_HEADING_RE.finditer(text))
    if not matches:
        return [Section(section_path=[], text=text)]

    sections: List[Section] = []
    stack: List[Tuple[int, str]] = []

    for idx, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        section_path = [t for _, t in stack]
        combined = f"{title}\n\n{body}".strip() if body else title
        sections.append(Section(section_path=section_path, text=combined))

    return sections

# ---------- RST ----------
def parse_rst_sections(text: str) -> List[Section]:
    lines = text.splitlines()
    heading_idxs: List[Tuple[int, str]] = []
    for i in range(len(lines) - 1):
        title = lines[i].strip()
        underline = lines[i + 1].strip()
        if title and re.fullmatch(r"[=~`^\"'#*+\-]{3,}", underline):
            heading_idxs.append((i, title))

    if not heading_idxs:
        return [Section(section_path=[], text=text)]

    sections: List[Section] = []
    for j, (i, title) in enumerate(heading_idxs):
        start_line = i + 2
        end_line = heading_idxs[j + 1][0] if j + 1 < len(heading_idxs) else len(lines)
        body = "\n".join(lines[start_line:end_line]).strip()
        combined = f"{title}\n\n{body}".strip() if body else title
        sections.append(Section(section_path=[title], text=combined))

    return sections

# ---------- HTML ----------
class HParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_heading = False
        self.heading_level = 0
        self.heading_text = ""
        self.current_path: List[str] = []
        self.current_buf: List[str] = []
        self.sections: List[Section] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("h1", "h2", "h3"):
            self.flush_section()
            self.in_heading = True
            self.heading_level = int(tag[1])
            self.heading_text = ""

    def handle_endtag(self, tag):
        if tag in ("h1", "h2", "h3") and self.in_heading:
            title = norm_ws(self.heading_text)
            if title:
                while len(self.current_path) >= self.heading_level:
                    self.current_path.pop()
                while len(self.current_path) < self.heading_level - 1:
                    self.current_path.append("Untitled")
                self.current_path.append(title)
            self.in_heading = False
            self.heading_level = 0
            self.heading_text = ""

    def handle_data(self, data):
        if not data:
            return
        if self.in_heading:
            self.heading_text += data
        else:
            s = norm_ws(data)
            if s:
                self.current_buf.append(s)

    def flush_section(self):
        if self.current_buf:
            text = "\n\n".join(self.current_buf).strip()
            self.sections.append(Section(section_path=self.current_path.copy(), text=text))
            self.current_buf = []

def parse_html_sections(text: str) -> List[Section]:
    p = HParser()
    p.feed(text)
    p.flush_section()
    if not p.sections:
        stripped = re.sub(r"<[^>]+>", " ", text)
        return [Section(section_path=[], text=norm_ws(stripped))]
    return p.sections

# ---------- XML (systemd-ish manpages) ----------
def parse_xml_sections(text: str) -> List[Section]:
    try:
        root = ET.fromstring(text)
    except Exception:
        return [Section(section_path=[], text=norm_ws(re.sub(r"<[^>]+>", " ", text)))]

    def collect_text(elem: ET.Element) -> str:
        parts = []
        for t in elem.itertext():
            s = norm_ws(t)
            if s:
                parts.append(s)
        return "\n\n".join(parts).strip()

    refname = None
    for rn in root.findall(".//refname"):
        refname = norm_ws("".join(rn.itertext()))
        if refname:
            break

    sections: List[Section] = []
    for rs1 in root.findall(".//refsect1"):
        title_elem = rs1.find("title")
        title = norm_ws("".join(title_elem.itertext())) if title_elem is not None else "Section"
        body = collect_text(rs1)
        if body:
            path = ([refname] if refname else []) + ([title] if title else [])
            sections.append(Section(section_path=path, text=body))

    if not sections:
        sections.append(Section(section_path=[refname] if refname else [], text=collect_text(root)))

    return sections

def iter_source_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.endswith(".meta.json"):
            continue
        ext = p.suffix.lower()
        if ext in SUPPORTED_EXTS:
            yield p

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-root", default=str(DOCS_ROOT_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--max-chars", type=int, default=4000)
    ap.add_argument("--overlap-chars", type=int, default=250)
    args = ap.parse_args()

    docs_root = pathlib.Path(args.docs_root).expanduser().resolve()
    out_path = pathlib.Path(args.out).expanduser().resolve()

    n_docs = 0
    n_chunks = 0

    with out_path.open("w", encoding="utf-8") as out:
        for fpath in iter_source_files(docs_root):
            ext = fpath.suffix.lower()
            kind = SUPPORTED_EXTS.get(ext, "unknown")

            rel_path = str(fpath.relative_to(docs_root))
            project = rel_path.split("/", 1)[0] if "/" in rel_path else rel_path

            meta = load_sidecar_meta(fpath) or {}
            title = meta.get("title")

            text = read_text(fpath)
            if not text.strip():
                continue

            if kind == "markdown":
                sections = parse_markdown_sections(text)
            elif kind == "rst":
                sections = parse_rst_sections(text)
            elif kind == "html":
                sections = parse_html_sections(text)
            elif kind == "xml":
                sections = parse_xml_sections(text)
            else:
                sections = [Section(section_path=[], text=text)]

            doc_key = f"{project}:{rel_path}"
            n_docs += 1

            for sec in sections:
                sec_text = sec.text.strip()
                if not sec_text:
                    continue
                chunks = chunk_paragraphs(sec_text, args.max_chars, args.overlap_chars)

                for idx, chunk in enumerate(chunks):
                    chunk_clean = chunk.strip()
                    if not chunk_clean:
                        continue
                    heading = sec.section_path[-1] if sec.section_path else None

                    chunk_id = f"{doc_key}:{sha1((heading or '') + '|' + str(idx) + '|' + chunk_clean[:200])}"

                    rec = {
                        "chunk_id": chunk_id,
                        "doc_key": doc_key,
                        "project": project,
                        "rel_path": rel_path,
                        "kind": kind,
                        "title": title,
                        "section_path": sec.section_path,
                        "heading": heading,
                        "chunk_index": idx,
                        "text": chunk_clean,
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_chunks += 1

    print(f"Wrote {n_chunks} chunks from {n_docs} documents to {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
