from __future__ import annotations

import re

_MULTISPACE = re.compile(r"[ \t]+")
_MULTINEWLINE = re.compile(r"\n{3,}")

def clean_text(text: str) -> str:
    # Normalize line endings first
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # Collapse internal runs of spaces/tabs to one space.
    text = _MULTISPACE.sub(" ", text)
    # Collapse 3+ newlines to a single paragraph break
    text = _MULTINEWLINE.sub("\n\n", text)
    return text.strip()