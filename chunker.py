"""Structure-aware chunking for SOP markdown (text + tables).

Rules
-----
* The document is parsed into SOPs (H2 headings) -> sections (H3+ headings and
  stand-alone **bold** lines used as pseudo headings) -> units (numbered point,
  paragraph, table).
* A numbered point is NEVER cut in the middle; points are grouped up to MAX_CHARS.
* A table is its own chunk, kept whole (header + rows). Very long tables are split
  by rows and the header is repeated. Header-only (blank template) tables are dropped.
* Every chunk carries its breadcrumb ("SOP > Section > Sub-section") both in the
  embedded text and in metadata, so a chunk is understandable on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import HARD_MAX_CHARS, MAX_CHARS, MAX_TABLE_ROWS

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
HRULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
LIST_START = re.compile(r"^(\d+[.)]|[-*\u2022])\s+")
BOLD_ONLY = re.compile(r"^\*\*(.+?)\*\*\s*[:\uff1a]?\s*$")
TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
SENTENCE = re.compile(r"(?<=[.!?])\s+")
SEP = " \u203a "  # breadcrumb separator


@dataclass
class Unit:
    kind: str                 # "item" | "para" | "table"
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines).strip()


@dataclass
class Section:
    sop_idx: int
    path: list[str]
    units: list[Unit]


@dataclass
class Sop:
    idx: int
    title: str
    start: int
    end: int = -1
    markdown: str = ""
    n_points: int = 0
    n_tables: int = 0


def _clean(t: str) -> str:
    t = re.sub(r"[*_`#]+", "", t)
    return re.sub(r"\s+", " ", t).strip(" :-\u2014")[:110]


def parse(md: str) -> tuple[list[Sop], list[Section]]:
    lines = md.splitlines()
    sops: list[Sop] = []
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    pseudo: str | None = None
    doc_title = ""
    units: list[Unit] = []
    cur: Unit | None = None

    def path() -> list[str]:
        return [t for _, t in stack] + ([pseudo] if pseudo else [])

    def end_unit():
        nonlocal cur
        if cur and cur.text:
            units.append(cur)
        cur = None

    def end_section():
        nonlocal units
        end_unit()
        if units and sops:
            sections.append(Section(len(sops) - 1, path(), units))
        units = []

    def ensure_sop(i: int):
        if not sops:
            sops.append(Sop(0, doc_title or "Document", i))

    i = 0
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()
        m = HEADING.match(line)
        if m:
            end_section()
            level, title = len(m.group(1)), _clean(m.group(2))
            if level == 1 and not doc_title:
                doc_title = title
            else:
                if level == 1:
                    level = 3                      # stray H1 inside the body = a sub-section
                if level == 2:
                    if sops:
                        sops[-1].end = i
                    sops.append(Sop(len(sops), title, i))
                    stack, pseudo = [], None
                else:
                    ensure_sop(i)
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                    stack.append((level, title))
                    pseudo = None
            i += 1
            continue
        if not line.strip():
            end_unit(); i += 1; continue
        if HRULE.match(line):
            end_unit(); i += 1; continue
        if line.lstrip().startswith("|"):
            end_unit()
            tbl = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                tbl.append(lines[i].rstrip()); i += 1
            ensure_sop(i)
            units.append(Unit("table", tbl))
            continue
        ensure_sop(i)
        bold = BOLD_ONLY.match(line.strip())
        if bold and not raw[:1].isspace():
            end_section()
            pseudo = _clean(bold.group(1))
            i += 1
            continue
        if not raw[:1].isspace() and LIST_START.match(line):
            end_unit()
            cur = Unit("item", [line])
        elif cur is None:
            cur = Unit("para", [line])
        else:
            cur.lines.append(line)
        i += 1
    end_section()
    if sops:
        sops[-1].end = len(lines)

    # dedupe titles, attach raw markdown + counters
    seen: dict[str, int] = {}
    for s in sops:
        seen[s.title] = seen.get(s.title, 0) + 1
        if seen[s.title] > 1:
            s.title = f"{s.title} ({seen[s.title]})"
        body = lines[s.start + 1 if HEADING.match(lines[s.start]) else s.start : s.end]
        s.markdown = re.sub(r"\n{3,}", "\n\n", "\n".join(body)).strip().rstrip("-").strip()
    for sec in sections:
        for u in sec.units:
            if u.kind == "item":
                sops[sec.sop_idx].n_points += 1
            elif u.kind == "table" and _table_parts(u.lines)[1]:
                sops[sec.sop_idx].n_tables += 1
    return [s for s in sops if s.markdown], sections


# ----------------------------------------------------------------------- tables
def _table_parts(tlines: list[str]) -> tuple[list[str], list[str]]:
    """Return (header_lines, body_rows). Body rows that are entirely blank are dropped."""
    if len(tlines) >= 2 and TABLE_SEP.match(tlines[1]):
        header, body = tlines[:2], tlines[2:]
    else:
        header, body = tlines[:1], tlines[1:]
    body = [r for r in body if re.sub(r"[|\s:\-]", "", r)]
    return header, body


def _split_table(tlines: list[str], caption: str) -> list[str]:
    header, body = _table_parts(tlines)
    if not body:
        return []                                   # empty template / form skeleton
    out, group, size = [], [], 0
    for row in body:
        if group and (len(group) >= MAX_TABLE_ROWS or size + len(row) > MAX_CHARS * 2):
            out.append(group); group, size = [], 0
        group.append(row); size += len(row)
    out.append(group)
    head = ("**" + caption + "**\n\n") if caption else ""
    return [head + "\n".join(header + g) for g in out]


# ------------------------------------------------------------------------ text
def _split_long(text: str) -> list[str]:
    if len(text) <= HARD_MAX_CHARS:
        return [text]
    parts, cur = [], ""
    for s in SENTENCE.split(text):
        if cur and len(cur) + len(s) > MAX_CHARS:
            parts.append(cur.strip()); cur = ""
        cur += s + " "
    return parts + ([cur.strip()] if cur.strip() else [])


def chunk_section(sec: Section) -> list[tuple[str, str]]:
    """Return [(chunk_type, body)] for one section."""
    out: list[tuple[str, str]] = []
    buf: list[str] = []

    def flush():
        nonlocal buf
        if buf:
            out.append(("text", "\n".join(buf)))
        buf = []

    for u in sec.units:
        if u.kind == "table":
            caption = ""
            if buf and len(buf[-1]) < 150 and not LIST_START.match(buf[-1]):
                caption = re.sub(r"\*+", "", buf.pop()).strip()   # short lead-in line = caption
            flush()
            out.extend(("table", t) for t in _split_table(u.lines, caption))
            continue
        for piece in _split_long(u.text):
            if buf and sum(map(len, buf)) + len(piece) > MAX_CHARS:
                flush()
            buf.append(piece)
    flush()
    return out


def build_chunks(md: str) -> tuple[list[Sop], list[dict]]:
    sops, sections = parse(md)
    title_of = {s.idx: s.title for s in sops}
    chunks: list[dict] = []
    counters: dict[int, int] = {}
    for sec in sections:
        if sec.sop_idx not in title_of:
            continue
        sop_title = title_of[sec.sop_idx]
        crumb = SEP.join([sop_title] + sec.path)
        for ctype, body in chunk_section(sec):
            n = counters.get(sec.sop_idx, 0)
            counters[sec.sop_idx] = n + 1
            chunks.append({
                "id": f"sop{sec.sop_idx:02d}-c{n:03d}",
                "document": f"[{crumb}]\n\n{body}",      # what gets embedded
                "body": body,
                "meta": {
                    "sop_id": f"sop{sec.sop_idx:02d}",
                    "sop_title": sop_title,
                    "section": SEP.join(sec.path) or "General",
                    "chunk_type": ctype,
                    "chunk_index": n,
                    "n_chars": len(body),
                },
            })
    return sops, chunks
