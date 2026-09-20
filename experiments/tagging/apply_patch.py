"""Apply a fixer-produced old/new patch to a source file, matching `old` against the file
whitespace-insensitively (the fixer reliably gets the CONTENT right but not always the exact
hard-wrap positions of the source), while writing the fixer's `new` text back verbatim.

Pure bookkeeping: this resolves WHERE the substring is in the real file; it does not alter,
paraphrase, or judge the patch content in any way.

Usage:
    .venv/bin/python experiments/tagging/apply_patch.py <target_file> <fixer_output_file>

fixer_output_file must contain "old: ..." then a blank line then "new: ...", matching the
fixer prompt's required output format.
"""
import re
import sys
from pathlib import Path


def normalize_with_map(s: str):
    """Collapse whitespace runs to a single space; return (normalized_str, index_map) where
    index_map[i] is the position in `s` that normalized char i corresponds to."""
    norm_chars = []
    index_map = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            start = i
            while i < n and s[i].isspace():
                i += 1
            norm_chars.append(" ")
            index_map.append(start)
        else:
            norm_chars.append(c)
            index_map.append(i)
            i += 1
    return "".join(norm_chars), index_map


def whitespace_insensitive_replace(content: str, old: str, new: str) -> str:
    norm_content, cmap = normalize_with_map(content)
    norm_old, _ = normalize_with_map(old.strip())

    idx = norm_content.find(norm_old)
    if idx == -1:
        raise ValueError("old text not found in target file, even whitespace-insensitively")
    second = norm_content.find(norm_old, idx + 1)
    if second != -1:
        raise ValueError("old text matches multiple locations in target file -- ambiguous")

    start_orig = cmap[idx]
    end_norm = idx + len(norm_old) - 1
    end_orig = cmap[end_norm]
    if content[end_orig].isspace():
        j = end_orig
        while j < len(content) and content[j].isspace():
            j += 1
        end_orig = j
    else:
        end_orig += 1

    return content[:start_orig] + new + content[end_orig:]


def parse_fixer_output(text: str):
    m = re.search(r"old:\s*(.*?)\n\s*\nnew:\s*(.*)", text, re.S)
    if not m:
        raise ValueError(f"could not parse old:/new: from fixer output:\n{text}")
    old = m.group(1).strip("\n")
    new = m.group(2).strip("\n")
    if old.strip() == "NONE" or new.strip() == "NONE":
        return None, None
    return old, new


def main():
    if len(sys.argv) != 3:
        print("usage: apply_patch.py <target_file> <fixer_output_file>")
        sys.exit(1)
    target_path = Path(sys.argv[1])
    fixer_path = Path(sys.argv[2])

    old, new = parse_fixer_output(fixer_path.read_text())
    if old is None:
        print("Fixer returned old: NONE / new: NONE -- no patch to apply.")
        sys.exit(2)

    content = target_path.read_text()
    try:
        patched = whitespace_insensitive_replace(content, old, new)
    except ValueError as e:
        print(f"PATCH FAILED: {e}")
        sys.exit(1)

    target_path.write_text(patched)
    print(f"Patch applied -> {target_path}")
    print(f"  old ({len(old)} chars): {old[:120]!r}{'...' if len(old) > 120 else ''}")
    print(f"  new ({len(new)} chars): {new[:120]!r}{'...' if len(new) > 120 else ''}")


if __name__ == "__main__":
    main()
