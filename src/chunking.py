"""Chunking for White Nights: structure split -> normalize -> recursive chunk.

Pure-local (no OpenAI). Shared by ingest.py (embedding) and scenes.py (segmentation)
so both agree on the exact same 84 chunks and their ids.
"""
import re
from pathlib import Path

import tiktoken
from chonkie import OverlapRefinery, RecursiveChunker

ROOT = Path(__file__).resolve().parent.parent
TEXT_PATH = ROOT / "data" / "white_nights.txt"

CHUNK_SIZE = 350          # tokens
CHUNK_OVERLAP = 50        # tokens


# example ─ in:  "...front matter...\nFIRST NIGHT\nIt was a wonderful night...\nSECOND NIGHT\n..."
#           out: [{"chapter": "First Night", "text": "It was a wonderful night..."},
#                 {"chapter": "Second Night", "text": "..."}, ...]   # 5 sections, front matter dropped
def structure_split(raw: str) -> list[dict]:
    """Cut the novel at its section headings; drop the front matter (parts[0])."""
    headings = ["FIRST NIGHT", "SECOND NIGHT", "THIRD NIGHT", "FOURTH NIGHT", "MORNING"]
    # (?m)^...$  -> headings must sit alone on their own line (safe against prose matches)
    pattern = r"(?m)^(" + "|".join(headings) + r")\s*$"
    parts = re.split(pattern, raw)
    # parts = [front_matter, "FIRST NIGHT", text1, "SECOND NIGHT", text2, ...]
    sections = []
    for i in range(1, len(parts), 2):
        sections.append({"chapter": parts[i].title(), "text": parts[i + 1].strip()})
    return sections


# example ─ in:  "...accustomed\n\n\x0cto meet at the\nsame time..."   (PDF hard-wraps + form feed)
#           out: "...accustomed to meet at the same time..."          (\n\n paragraph breaks kept)
def normalize(text: str) -> str:
    """Clean PDF-extraction artifacts while preserving paragraph breaks (\\n\\n).

    Run AFTER the structure split, so headings on their own lines aren't merged away.
    """
    # form-feed page breaks often land mid-sentence -> join with a single space
    text = re.sub(r"\s*\x0c\s*", " ", text)
    # collapse 3+ newlines down to a single paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    # join hard-wrapped lines inside a paragraph (a lone newline -> space)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # squeeze runs of spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# example ─ in:  [{"chapter": "First Night", "text": "It was a wonderful night..."}, ...]
#           out: [{"id": "white_nights_first_night_001", "work": "White Nights",
#                  "chapter": "First Night", "section": 1, "text": "It was a wonderful night...",
#                  "prev_id": None, "next_id": "white_nights_first_night_002"}, ...]   # 84 chunks
def build_chunks(sections: list[dict]) -> list[dict]:
    """Recursive-chunk each section, keeping chapter provenance on every chunk.

    Token-based sizing uses cl100k_base (the tokenizer text-embedding-3-small uses),
    so CHUNK_SIZE means the same thing to the chunker and the embedder. Overlap is
    added per section via OverlapRefinery, so it never bleeds across a NIGHT boundary.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    chunker = RecursiveChunker(tokenizer=enc, chunk_size=CHUNK_SIZE)
    refinery = OverlapRefinery(
        tokenizer=enc, context_size=CHUNK_OVERLAP, mode="token", method="prefix", merge=True
    )
    chunks = []
    for sec in sections:
        slug = sec["chapter"].replace(" ", "_").lower()
        refined = refinery(chunker.chunk(normalize(sec["text"])))
        for j, ch in enumerate(refined, start=1):
            piece = ch.text
            chunks.append({
                "id": f"white_nights_{slug}_{j:03d}",
                "work": "White Nights",
                "chapter": sec["chapter"],
                "section": j,
                "text": piece,
            })
    # link each chunk to its neighbours (flat list -> links cross section boundaries)
    for i, c in enumerate(chunks):
        c["prev_id"] = chunks[i - 1]["id"] if i > 0 else None
        c["next_id"] = chunks[i + 1]["id"] if i < len(chunks) - 1 else None
    return chunks


# example ─ in:  ()   out: the 84 chunk dicts above  (reads white_nights.txt, splits, chunks)
def load_chunks() -> list[dict]:
    """Convenience: read the novel and return the ordered chunk list."""
    return build_chunks(structure_split(TEXT_PATH.read_text(encoding="utf-8")))
