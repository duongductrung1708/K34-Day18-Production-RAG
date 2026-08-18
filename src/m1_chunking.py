from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────

_semantic_model = None


def _get_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _semantic_model


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    import numpy as np

    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(text=sentences[0], metadata={**metadata, "strategy": "semantic", "chunk_index": 0})]

    model = _get_semantic_model()
    embeddings = model.encode(sentences)

    chunks = []
    current_group = [sentences[0]]
    for i in range(1, len(sentences)):
        v1 = embeddings[i - 1]
        v2 = embeddings[i]
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        sim = float(np.dot(v1, v2) / denom)

        if sim < threshold:
            chunk_text = " ".join(current_group).strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, metadata={**metadata, "strategy": "semantic", "chunk_index": len(chunks)}))
            current_group = [sentences[i]]
        else:
            current_group.append(sentences[i])

    if current_group:
        chunk_text = " ".join(current_group).strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, metadata={**metadata, "strategy": "semantic", "chunk_index": len(chunks)}))

    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return ([], [])

    parents: list[Chunk] = []
    children: list[Chunk] = []

    def _add_parent_and_children(parent_text: str, parent_idx: int):
        pid = f"parent_{parent_idx}"
        p_chunk = Chunk(
            text=parent_text.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
            parent_id=pid,
        )
        parents.append(p_chunk)

        # Split parent text into child chunks
        p_paras = [p.strip() for p in parent_text.split("\n\n") if p.strip()]
        cur_child = ""
        for para in p_paras:
            if len(cur_child) + len(para) > child_size and cur_child:
                children.append(Chunk(
                    text=cur_child.strip(),
                    metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                    parent_id=pid,
                ))
                cur_child = ""

            if len(para) > child_size:
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', para) if s.strip()]
                for sent in sentences:
                    if len(cur_child) + len(sent) > child_size and cur_child:
                        children.append(Chunk(
                            text=cur_child.strip(),
                            metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                            parent_id=pid,
                        ))
                        cur_child = ""
                    if len(sent) > child_size:
                        for step in range(0, len(sent), child_size):
                            part = sent[step:step + child_size].strip()
                            if part:
                                children.append(Chunk(
                                    text=part,
                                    metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                                    parent_id=pid,
                                ))
                    else:
                        cur_child += sent + " "
            else:
                cur_child += para + "\n\n"

        if cur_child.strip():
            children.append(Chunk(
                text=cur_child.strip(),
                metadata={**metadata, "chunk_type": "child", "parent_id": pid},
                parent_id=pid,
            ))

    current_parent = ""
    for para in paragraphs:
        if len(current_parent) + len(para) > parent_size and current_parent:
            _add_parent_and_children(current_parent, len(parents))
            current_parent = ""
        current_parent += para + "\n\n"

    if current_parent.strip():
        _add_parent_and_children(current_parent, len(parents))

    return (parents, children)


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    lines = text.split("\n")
    chunks = []
    current_header = ""
    current_lines: list[str] = []

    for line in lines:
        header_match = re.match(r'^(#{1,3})\s+(.+)$', line.strip())
        if header_match:
            if current_lines:
                chunk_text = "\n".join(current_lines).strip()
                if chunk_text:
                    meta = {**metadata, "strategy": "structure"}
                    if current_header:
                        meta["section"] = current_header
                    chunks.append(Chunk(text=chunk_text, metadata=meta))
                current_lines = []
            current_header = header_match.group(2).strip()
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text:
            meta = {**metadata, "strategy": "structure"}
            if current_header:
                meta["section"] = current_header
            chunks.append(Chunk(text=chunk_text, metadata=meta))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
