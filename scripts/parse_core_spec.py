#!/usr/bin/env python3
"""Standalone C++ standard specification parser (N4950) into data/processed/chunks.jsonl.

Supports two modes in a single self-contained script:
1. Chapter sweep (default): walks the Core Language chapters (Clauses 1-15,
   everything before Clause 16 [library]), skips container sections that have
   no paragraphs of their own, skips clauses already present in the output
   file, and parses the missing clauses.
2. Single-clause or explicit-clause mode (--clauses ...): parses specific
   clauses or cppwp URLs and updates/appends them in the chunk file.

A clause with four or more numbered paragraphs is split into one chunk per
paragraph (`clause:<id>#p<N>`). Shorter clauses and glossary definitions
(`[intro.defs]`, which use `div.texpara`) stay whole (`clause:<id>`).

Examples:
    python scripts/parse_core_spec.py --dry-run
    python scripts/parse_core_spec.py -o data/processed/chunks.jsonl
    python scripts/parse_core_spec.py --chapters basic dcl.dcl class --dry-run
    python scripts/parse_core_spec.py --clauses expr.post.incr intro.execution -o data/processed/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "chunks.jsonl"

BASE_URL = "https://timsong-cpp.github.io/cppwp"
USER_AGENT = "Mozilla/5.0 (compatible; cppwp-chunk-parser/1.0)"
HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
TEXT_BLOCK_CLASSES = {"para", "example"}
DEFNS_TEXT_BLOCK_CLASSES = {"para", "example", "texpara"}
SKIP_LINK_CLASSES = {"footnoteref", "marginalized", "sourceLink", "index"}
SKIP_CONTAINER_CLASSES = {"marginalizedparent", "sourceLinkParent"}
PARAGRAPH_SPLIT_AT = 4

# Default Core Language chapter pages on timsong-cpp.github.io/cppwp/n4950/
# (Clauses 1 through 15, stopping before Clause 16 [library]).
CORE_CHAPTERS = (
    "intro.scope",
    "intro.refs",
    "intro.defs",
    "intro",
    "lex",
    "basic",
    "expr",
    "stmt.stmt",
    "dcl.dcl",
    "module",
    "class",
    "over",
    "temp",
    "except",
    "cpp",
)


class Node:
    def __init__(self, tag: str | None, attrs: list[tuple[str, str | None]]) -> None:
        self.tag = tag
        self.attrs = {key: value or "" for key, value in attrs}
        self.children: list[Node | str] = []

    def classes(self) -> list[str]:
        return self.attrs.get("class", "").split()

    def has_class(self, name: str) -> bool:
        return name in self.classes()

    def has_any_class(self, names: set[str]) -> bool:
        return bool(set(self.classes()) & names)

    def get(self, key: str, default: str | None = None) -> str | None:
        return self.attrs.get(key, default)

    def find_all(self, tag: str | None = None, class_name: str | None = None) -> list[Node]:
        found: list[Node] = []

        def walk(node: Node) -> None:
            if node.tag and (tag is None or node.tag == tag):
                if class_name is None or node.has_class(class_name):
                    found.append(node)
            for child in node.children:
                if isinstance(child, Node):
                    walk(child)

        walk(self)
        return found

    def text(self) -> str:
        parts: list[str] = []

        def walk(node: Node | str) -> None:
            if isinstance(node, str):
                parts.append(node)
                return
            for child in node.children:
                walk(child)

        walk(self)
        return "".join(parts)


class TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node(None, [])
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        self.stack.append(node)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                self.stack = self.stack[:index]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


def fetch_html(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        raise SystemExit(f"Failed to fetch {url}: HTTP {error.code}") from error
    except urllib.error.URLError as error:
        raise SystemExit(f"Failed to fetch {url}: {error.reason}") from error


def parse_html(html: str) -> Node:
    builder = TreeBuilder()
    builder.feed(html)
    return builder.root


def collapse_ws(text: str) -> str:
    for character in ("\u00a0", "\u2002", "\u2003", "\u2009", "\u200b"):
        text = text.replace(character, " ")
    return re.sub(r"\s+", " ", text).strip()


def tidy_prose(text: str) -> str:
    text = re.sub(r" +([.,;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return re.sub(r" +", " ", text).strip()


def clean_anchor_title(title: str) -> str:
    for character in ("\u00a0", "\u2002", "\u2003"):
        title = title.replace(character, " ")
    title = re.sub(r"\s*\[[^\]]+\]\s*$", "", title)
    title = re.sub(r"^[0-9]+(?:\.[0-9]+)*\s+", "", title)
    return re.sub(r"\s+", " ", title).strip()


def clause_label(heading: Node) -> str:
    for anchor in heading.find_all("a", "abbr_ref"):
        match = re.fullmatch(r"\[(.+)\]", collapse_ws(anchor.text()))
        if match:
            return match.group(1)
    return ""


def heading_title(heading: Node) -> str:
    parts: list[str] = []

    def walk(node: Node | str) -> None:
        if isinstance(node, str):
            parts.append(node)
            return
        if node.has_class("secnum") or node.has_class("abbr_ref"):
            return
        for child in node.children:
            walk(child)

    walk(heading)
    return collapse_ws("".join(parts))


def skipped(node: Node) -> bool:
    classes = set(node.classes())
    if classes & SKIP_CONTAINER_CLASSES:
        return True
    return node.tag == "a" and bool(classes & SKIP_LINK_CLASSES)


def render_block(element: Node) -> str:
    """Render a paragraph, texpara, example, or footnote as cleaned plain text."""
    pieces: list[tuple[str, str]] = []

    def walk(node: Node | str) -> None:
        if isinstance(node, str):
            pieces.append(("text", node.replace("\n", " ")))
            return
        if skipped(node):
            return
        if node.has_class("codeblock"):
            lines = [
                re.sub(r"[ \t]+", " ", line).rstrip()
                for line in node.text().replace("\u00a0", " ").splitlines()
            ]
            code = "\n".join(line for line in lines if line.strip())
            pieces.append(("code", code))
            return
        for child in node.children:
            walk(child)

    walk(element)
    blocks: list[str] = []
    prose: list[str] = []

    def flush() -> None:
        if not prose:
            return
        text = tidy_prose(collapse_ws("".join(prose)))
        prose.clear()
        if text:
            blocks.append(text)

    for kind, value in pieces:
        if kind == "text":
            prose.append(value)
        else:
            flush()
            if value:
                blocks.append(value)
    flush()
    return "\n\n".join(blocks)


def block_classes_for(clause_or_chapter: str) -> set[str]:
    if clause_or_chapter == "intro.defs" or clause_or_chapter.startswith("defns."):
        return DEFNS_TEXT_BLOCK_CLASSES
    return TEXT_BLOCK_CLASSES


def clause_body(wrapper: Node, heading: Node, clause_id: str = "") -> tuple[list[Node], list[Node]]:
    blocks: list[Node] = []
    footnotes: list[Node] = []
    allowed_classes = block_classes_for(clause_id)
    past_heading = False
    for child in wrapper.children:
        if not isinstance(child, Node):
            continue
        if child is heading:
            past_heading = True
            continue
        if not past_heading:
            continue
        if child.tag in HEADING_TAGS:
            break
        if child.has_class("footnote"):
            footnotes.append(child)
        elif child.has_any_class(allowed_classes):
            blocks.append(child)
    return blocks, footnotes


def cleaned_text(blocks: list[Node]) -> str:
    parts = [render_block(block) for block in blocks]
    return "\n\n".join(part for part in parts if part)


def cross_references(blocks: list[Node], parent_clause: str) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    seen: set[str] = set()
    for block in blocks:
        for anchor in block.find_all("a"):
            href = anchor.get("href") or ""
            if not href or href.startswith(("#", "http", ".")) or href == parent_clause:
                continue
            if href in seen:
                continue
            seen.add(href)
            title = clean_anchor_title(anchor.get("title") or "")
            if not title:
                title = collapse_ws(anchor.text()).strip("[]")
            references.append(
                {
                    "target_clause_id": href,
                    "anchor_text": title,
                    "edge_type": "references",
                }
            )
    return references


def source_url(edition: str, parent_clause: str, clause_id: str) -> str:
    if not parent_clause:
        return f"{BASE_URL}/{edition.lower()}/{clause_id}"
    if clause_id.startswith(parent_clause + "."):
        fragment = clause_id[len(parent_clause) + 1 :]
    else:
        fragment = clause_id
    return f"{BASE_URL}/{edition.lower()}/{parent_clause}#{fragment}"


def section_info(headings: list[Node]) -> tuple[str, str]:
    """Return (section_id, section title) from the chapter heading on the page."""
    chapter = next((heading for heading in headings if heading.tag == "h1"), None)
    if chapter is None and headings:
        chapter = headings[0]
    if chapter is None:
        return "", ""
    return clause_label(chapter), heading_title(chapter)


def paragraph_kind(block: Node) -> str:
    if block.has_class("nonNormativeOnly"):
        return "note"
    if block.has_class("example"):
        return "example"
    return "clause_subsection"


def referenced_footnotes(block: Node, footnotes: list[Node]) -> list[Node]:
    wanted = {
        (anchor.get("href") or "").lstrip("#")
        for anchor in block.find_all("a", "footnoteref")
    }
    wanted.discard("")
    return [note for note in footnotes if (note.get("id") or "") in wanted]


def make_chunk(
    clause_id: str,
    section_id: str,
    section: str,
    parent_clause: str,
    title: str,
    chunk_type: str,
    text: str,
    url: str,
    edition: str,
    references: list[dict[str, str]],
    paragraph: str | None = None,
) -> dict[str, object]:
    chunk_id = f"clause:{clause_id}"
    if paragraph:
        chunk_id = f"{chunk_id}#p{paragraph}"
    chunk: dict[str, object] = {
        "chunk_id": chunk_id,
        "clause_id": clause_id,
        "section_id": section_id,
        "section": section,
        "parent_clause": parent_clause,
        "title": title,
        "chunk_type": chunk_type,
    }
    if paragraph:
        chunk["paragraph"] = paragraph
    chunk["text"] = text
    chunk["source_url"] = url
    chunk["spec_edition"] = edition
    chunk["cross_references"] = references
    return chunk


def parse_clause(
    html: str, edition: str, clause_id: str, paragraphs: bool | None = None
) -> list[dict[str, object]]:
    root = parse_html(html)
    wrapper = next((node for node in root.find_all("div", "wrapper") if node.tag), None)
    if wrapper is None:
        raise SystemExit("Page has no clause content.")

    headings = [
        child for child in wrapper.children if isinstance(child, Node) and child.tag in HEADING_TAGS
    ]
    heading = next((item for item in headings if clause_label(item) == clause_id), None)
    if heading is None:
        found = ", ".join(clause_label(item) for item in headings) or "none"
        raise SystemExit(f"Clause [{clause_id}] is not a heading on this page. Found: {found}.")
    heading_index = headings.index(heading)
    parent_clause = clause_label(headings[heading_index - 1]) if heading_index else ""
    if not parent_clause and "." in clause_id:
        parent_clause = clause_id.rsplit(".", 1)[0]
    section_id, section = section_info(headings)

    blocks, footnotes = clause_body(wrapper, heading, clause_id)
    title = heading_title(heading)
    paragraph_blocks = [block for block in blocks if block.has_class("para")]
    split = (
        len(paragraph_blocks) >= PARAGRAPH_SPLIT_AT if paragraphs is None else paragraphs
    )
    if not split or not paragraph_blocks:
        return [
            make_chunk(
                clause_id,
                section_id,
                section,
                parent_clause,
                title,
                "clause",
                cleaned_text(blocks),
                source_url(edition, parent_clause, clause_id),
                edition,
                cross_references(blocks + footnotes, parent_clause),
            )
        ]

    chunks: list[dict[str, object]] = []
    for index, block in enumerate(paragraph_blocks, start=1):
        paragraph = block.get("id") or str(index)
        references = cross_references(
            [block, *referenced_footnotes(block, footnotes)], parent_clause
        )
        chunks.append(
            make_chunk(
                clause_id,
                section_id,
                section,
                parent_clause,
                title,
                paragraph_kind(block),
                render_block(block),
                f"{BASE_URL}/{edition.lower()}/{clause_id}#{paragraph}",
                edition,
                references,
                paragraph,
            )
        )
    return chunks


def clause_page_url(clause: str, edition: str) -> tuple[str, str, str]:
    """Return (fetch URL, edition, clause id) for a clause id or a cppwp URL."""
    if clause.startswith("http://") or clause.startswith("https://"):
        match = re.search(r"/cppwp/([^/]+)/([^#?]+)", clause)
        if not match:
            raise SystemExit(f"Unrecognized clause URL: {clause}")
        edition = match.group(1).upper()
        clause_id = match.group(2)
        fragment = clause.split("#", 1)[1] if "#" in clause else ""
        if fragment:
            clause_id = f"{clause_id}.{fragment}" if not fragment.startswith(clause_id) else fragment
        return f"{BASE_URL}/{edition.lower()}/{clause_id}", edition, clause_id
    return f"{BASE_URL}/{edition.lower()}/{clause}", edition, clause


def load_chunks(path: Path) -> list[dict[str, object]]:
    raw = path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    chunks: list[dict[str, object]] = []
    index = 0
    length = len(raw)
    while index < length:
        while index < length and raw[index].isspace():
            index += 1
        if index >= length:
            break
        chunk, index = decoder.raw_decode(raw, index)
        chunks.append(chunk)
    return chunks


def stored_clause_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(chunk.get("clause_id") or "") for chunk in load_chunks(path)} - {""}


def belongs_to_clause(item: dict[str, object], clause_id: str) -> bool:
    chunk_id = item.get("chunk_id")
    prefix = f"clause:{clause_id}"
    return chunk_id == prefix or (isinstance(chunk_id, str) and chunk_id.startswith(prefix + "#"))


def replace_or_append(
    existing: list[dict[str, object]], new_chunks: list[dict[str, object]]
) -> list[dict[str, object]]:
    if not new_chunks:
        return existing
    clause_id = str(new_chunks[0]["clause_id"])
    kept: list[dict[str, object]] = []
    inserted = False
    for item in existing:
        if belongs_to_clause(item, clause_id):
            if not inserted:
                kept.extend(new_chunks)
                inserted = True
            continue
        kept.append(item)
    if not inserted:
        kept.extend(new_chunks)
    return kept


def write_all_chunks(chunks: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(item, ensure_ascii=False, indent=2) + "\n" for item in chunks),
        encoding="utf-8",
    )


def direct_children(node: Node, class_name: str) -> list[Node]:
    return [
        child
        for child in node.children
        if isinstance(child, Node) and child.has_class(class_name)
    ]


def section_heading(section: Node) -> Node | None:
    for child in section.children:
        if isinstance(child, Node) and child.tag in HEADING_TAGS:
            return child
    return None


def has_own_text(section: Node, chapter_id: str = "") -> bool:
    """A section has text when a paragraph or example is its own, not a child's."""
    allowed_classes = block_classes_for(chapter_id)
    for child in section.children:
        if isinstance(child, Node) and child.has_any_class(allowed_classes):
            return True
    return False


def chapter_url(edition: str, chapter_id: str) -> str:
    return f"{BASE_URL}/{edition.lower()}/{chapter_id}"


def chapter_sections(html: str, chapter_id: str) -> list[dict[str, object]]:
    """Return all sections on a chapter page, in document order."""
    root = parse_html(html)
    wrapper = next((node for node in root.find_all("div", "wrapper")), None)
    if wrapper is None:
        raise SystemExit(f"Chapter page [{chapter_id}] has no wrapper content.")

    top_sections = direct_children(wrapper, "section")
    if not top_sections:
        heading = next(
            (child for child in wrapper.children if isinstance(child, Node) and child.tag in HEADING_TAGS),
            None,
        )
        cid = clause_label(heading) if heading is not None else chapter_id
        return [
            {
                "chapter_id": chapter_id,
                "clause_id": cid,
                "title": heading_title(heading) if heading is not None else "",
                "parent_clause": "",
                "container": not has_own_text(wrapper, chapter_id),
            }
        ]

    found: list[dict[str, object]] = []

    def walk(section: Node, parent_id: str) -> None:
        heading = section_heading(section)
        clause_id = clause_label(heading) if heading is not None else ""
        if not clause_id:
            raise SystemExit(f"Section {section.get('id') or ''} in {chapter_id} has no clause id.")
        found.append(
            {
                "chapter_id": chapter_id,
                "clause_id": clause_id,
                "title": heading_title(heading) if heading is not None else "",
                "parent_clause": parent_id,
                "container": not has_own_text(section, chapter_id),
            }
        )
        for child in direct_children(section, "section"):
            walk(child, clause_id)

    for section in top_sections:
        walk(section, chapter_id)
    return found


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Standalone C++ standard specification parser into chunks.jsonl."
    )
    parser.add_argument("--edition", default="N4950", help="Working draft edition. Defaults to N4950.")
    parser.add_argument(
        "--chapters",
        nargs="+",
        default=list(CORE_CHAPTERS),
        help="Chapter IDs to walk. Defaults to Clauses 1-15 (Core Language).",
    )
    parser.add_argument(
        "--clauses",
        nargs="+",
        help="Specific clause IDs (or cppwp URLs) to parse instead of walking chapters.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Chunk file to append/update. Defaults to data/processed/chunks.jsonl.",
    )
    parser.add_argument(
        "--paragraphs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Split into numbered paragraphs. The default splits a clause with "
            f"{PARAGRAPH_SPLIT_AT} or more paragraphs. --no-paragraphs keeps it whole."
        ),
    )
    parser.add_argument(
        "--include-containers",
        action="store_true",
        help="Also parse container sections that have no paragraphs of their own.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print per-chapter section counts and missing clauses without writing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="In dry-run mode, also list every individual missing clause id.",
    )
    args = parser.parse_args(argv)

    if args.clauses:
        if args.dry_run:
            for item in args.clauses:
                _, _, cid = clause_page_url(item, args.edition)
                print(f"Would parse clause: {cid}")
            return
        chunks = load_chunks(args.output) if args.output.exists() else []
        for index, item in enumerate(args.clauses, start=1):
            url, edition, cid = clause_page_url(item, args.edition)
            print(f"[{index}/{len(args.clauses)}] {cid}", flush=True)
            new_chunks = parse_clause(fetch_html(url), edition, cid, paragraphs=args.paragraphs)
            chunks = replace_or_append(chunks, new_chunks)
        write_all_chunks(chunks, args.output)
        print(f"Saved {len(chunks)} total chunks to {args.output}.")
        return

    present = stored_clause_ids(args.output)
    all_sections: list[dict[str, object]] = []
    chapter_summaries: list[tuple[str, int, int, int, int]] = []

    for chapter_id in args.chapters:
        html = fetch_html(chapter_url(args.edition, chapter_id))
        sections = chapter_sections(html, chapter_id)
        containers = [s for s in sections if s["container"]]
        selected = sections if args.include_containers else [s for s in sections if not s["container"]]
        already = [s for s in selected if s["clause_id"] in present]
        missing = [s for s in selected if s["clause_id"] not in present]
        chapter_summaries.append(
            (chapter_id, len(sections), len(containers), len(already), len(missing))
        )
        all_sections.extend(sections)

    total_sections = sum(row[1] for row in chapter_summaries)
    total_containers = sum(row[2] for row in chapter_summaries)
    total_already = sum(row[3] for row in chapter_summaries)
    total_missing = sum(row[4] for row in chapter_summaries)

    print(f"{'Chapter':<14} {'Total':>7} {'Containers':>11} {'Stored':>8} {'To Parse':>10}")
    print("-" * 54)
    for chapter_id, total, cont, already, miss in chapter_summaries:
        print(f"{chapter_id:<14} {total:>7} {cont:>11} {already:>8} {miss:>10}")
    print("-" * 54)
    print(
        f"{'TOTAL':<14} {total_sections:>7} {total_containers:>11} "
        f"{total_already:>8} {total_missing:>10}"
    )

    selected_all = (
        all_sections if args.include_containers else [s for s in all_sections if not s["container"]]
    )
    missing_all = [s for s in selected_all if s["clause_id"] not in present]

    if args.verbose:
        for section in missing_all:
            marker = "container" if section["container"] else "text"
            print(f"  [{section['chapter_id']}] {section['clause_id']} ({marker}) - {section['title']}")

    if args.dry_run or not missing_all:
        return

    chunks = load_chunks(args.output) if args.output.exists() else []
    failed: list[str] = []
    current_chapter = ""
    for index, section in enumerate(missing_all, start=1):
        clause_id = str(section["clause_id"])
        chapter_id = str(section["chapter_id"])
        if chapter_id != current_chapter and current_chapter:
            write_all_chunks(chunks, args.output)
        current_chapter = chapter_id
        print(f"[{index}/{len(missing_all)}] ({chapter_id}) {clause_id}", flush=True)
        try:
            url = f"{BASE_URL}/{args.edition.lower()}/{clause_id}"
            new_chunks = parse_clause(
                fetch_html(url), args.edition, clause_id, paragraphs=args.paragraphs
            )
            chunks = replace_or_append(chunks, new_chunks)
        except (SystemExit, Exception) as exc:
            failed.append(clause_id)
            print(f"Failed: {clause_id} ({exc})", file=sys.stderr)

    write_all_chunks(chunks, args.output)
    print(f"Saved {len(chunks)} total chunks to {args.output}.")
    if failed:
        raise SystemExit(f"Failed to parse {len(failed)} clauses: {', '.join(failed)}")


if __name__ == "__main__":
    main()
