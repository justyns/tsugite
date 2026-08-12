"""Reading and surgically editing docx packages offline.

A docx is a zip of XML parts, and a document that has been near Word carries
comments, tracked changes, numbering and styles that no markdown round trip
survives. So every edit here mutates only the parts it has to and copies every
other entry through byte for byte.
"""

from __future__ import annotations

import copy
import hashlib
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from tsugite_onlyoffice.documents import replacing

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W14 = "http://schemas.microsoft.com/office/word/2010/wordml"
W15 = "http://schemas.microsoft.com/office/word/2012/wordml"
MC = "http://schemas.openxmlformats.org/markup-compatibility/2006"
CONTENT_TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"
PACKAGE_RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
XML = "http://www.w3.org/XML/1998/namespace"

CONTENT_TYPES_PART = "[Content_Types].xml"
DOCUMENT_PART = "word/document.xml"
DOCUMENT_RELS_PART = "word/_rels/document.xml.rels"
COMMENTS_PART = "word/comments.xml"
COMMENTS_EXTENDED_PART = "word/commentsExtended.xml"

# Written into the zip entry of any part this module creates. A fixed stamp keeps
# a save reproducible, which is the only property the format needs from it.
NEW_PART_DATE = (1980, 1, 1, 0, 0, 0)

_XMLNS = re.compile(r'xmlns(?::([^\s=]+))?="([^"]*)"')


def q(uri: str, tag: str) -> str:
    """Qualify a tag or attribute name with a namespace, ElementTree style."""
    return f"{{{uri}}}{tag}"


def w(tag: str) -> str:
    """Qualify a name with the WordprocessingML namespace."""
    return q(W, tag)


# w14:paraId identifies a w:p; the w15 trio lives on the w15:commentEx that threads it.
PARA_ID = q(W14, "paraId")
EX_PARA_ID = q(W15, "paraId")
PARA_ID_PARENT = q(W15, "paraIdParent")
DONE = q(W15, "done")

# Children whose text is not part of how the document currently reads: properties
# carry no body text, and a tracked deletion's w:delText is text that was removed.
_UNRENDERED = (w("pPr"), w("rPr"), w("del"))

_DECLARATION = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'

# Content type, document relationship and empty part, for each part a comment needs.
_COMMENT_PARTS = {
    COMMENTS_PART: (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.comments+xml",
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/comments",
        f'<w:comments xmlns:w="{W}" xmlns:w14="{W14}" xmlns:mc="{MC}" mc:Ignorable="w14"/>',
    ),
    COMMENTS_EXTENDED_PART: (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.commentsExtended+xml",
        "http://schemas.microsoft.com/office/2011/relationships/commentsExtended",
        f'<w15:commentsEx xmlns:w15="{W15}"/>',
    ),
}


def _lowest_unused(used: set, form: str) -> str:
    """The lowest positive number whose `form` rendering is not already in `used`."""
    number = 1
    while form.format(number) in used:
        number += 1
    return form.format(number)


def _end_of_tag(text: str, start: int) -> int:
    """Index of the '>' closing the tag opening at `start`, ignoring quoted ones."""
    quote = ""
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            quote = "" if char == quote else quote
        elif char in "\"'":
            quote = char
        elif char == ">":
            return index
    raise ValueError("docx: unterminated start tag")


def _split_root(data: bytes) -> tuple[str, str, str, dict[str, str]]:
    """Take a part apart into everything ElementTree cannot be trusted to reproduce.

    Returns:
        The prolog, the root start tag without its angle brackets, whatever
        trailed the root, and the prefix-to-URI map the root declares.
    """
    text = data.decode("utf-8")
    index = 0
    while True:
        index = text.index("<", index)
        if text[index + 1] not in "?!":
            break
        index = text.index(">", index) + 1
    tag = text[index + 1 : _end_of_tag(text, index)].rstrip()
    if tag.endswith("/"):
        tag = tag[:-1].rstrip()
    return text[:index], tag, text[text.rindex(">") + 1 :], dict(_XMLNS.findall(tag))


class XmlPart:
    """One XML part of a package, able to go back to bytes the way it arrived.

    ElementTree rewrites namespace prefixes and emits declarations only for the
    ones it sees used, which strips `w14`, `w15` and the prefixes `mc:Ignorable`
    names off a Word document and makes Word reject the file. The prolog and the
    root start tag therefore come from the original bytes, not from the tree.
    """

    def __init__(self, data: bytes) -> None:
        self._prolog, self._start_tag, self._epilog, self.namespaces = _split_root(data)
        self.root = ET.fromstring(data)

    def ensure_namespace(self, prefix: str, uri: str) -> None:
        """Declare a namespace on the root unless the part already carries it."""
        if uri in self.namespaces.values():
            return
        self.namespaces[prefix] = uri
        self._start_tag += f' xmlns:{prefix}="{uri}"'

    def to_bytes(self) -> bytes:
        """Serialize the part, keeping its prolog and root declarations intact."""
        for prefix, uri in self.namespaces.items():
            ET.register_namespace(prefix, uri)
        # ElementTree pads every empty element out to "<w:b />". It escapes ">"
        # in both text and attributes, so the sequence cannot occur anywhere else
        # in its output and squeezing it back is safe.
        body = ET.tostring(self.root, encoding="unicode").replace(" />", "/>")
        head, _, rest = body.partition(">")
        opened = f"<{self._start_tag}/>" if head.endswith("/") else f"<{self._start_tag}>{rest}"
        return (self._prolog + opened + self._epilog).encode("utf-8")


def _text_nodes(element: ET.Element, run: ET.Element | None = None):
    """Walk a paragraph, yielding `(run, node, rendered text)` in document order."""
    for child in element:
        tag = child.tag
        if tag in _UNRENDERED:
            continue
        if tag == w("t"):
            yield run, child, child.text or ""
        elif tag == w("tab"):
            yield run, child, "\t"
        elif tag == w("br"):
            yield run, child, " "
        elif tag == w("r"):
            yield from _text_nodes(child, child)
        else:
            yield from _text_nodes(child, run)


def _paragraph_text(paragraph: ET.Element) -> str:
    """Render one `w:p` the way `Document.text` renders it."""
    return "".join(rendered for _run, _node, rendered in _text_nodes(paragraph))


class _Piece(NamedTuple):
    """One text-bearing node, placed in the paragraph's rendering."""

    run: ET.Element | None
    node: ET.Element
    text: str
    start: int

    @property
    def stop(self) -> int:
        return self.start + len(self.text)


def _pieces(paragraph: ET.Element) -> list[_Piece]:
    pieces = []
    offset = 0
    for run, node, rendered in _text_nodes(paragraph):
        pieces.append(_Piece(run, node, rendered, offset))
        offset += len(rendered)
    return pieces


def _parents(element: ET.Element) -> dict[ET.Element, ET.Element]:
    """Map everything under an element to its parent, which ElementTree does not track."""
    return {child: parent for parent in element.iter() for child in parent}


def _set_text(node: ET.Element, text: str) -> None:
    """Set a `w:t`'s content, keeping whitespace at its edges significant."""
    node.text = text
    if text != text.strip():
        node.set(q(XML, "space"), "preserve")


def _split_before(run: ET.Element, node: ET.Element, parent: ET.Element) -> ET.Element | None:
    """Move `node` and everything after it into a fresh run alongside this one.

    Args:
        run: The run to split.
        node: The first child that belongs to the second half.
        parent: The run's parent, which may be the paragraph, a hyperlink or an
            insertion rather than the paragraph itself.

    Returns:
        The new run, or None when `node` already starts the run.
    """
    children = list(run)
    index = children.index(node)
    if index == 0 or (index == 1 and children[0].tag == w("rPr")):
        return None
    tail = ET.Element(w("r"), dict(run.attrib))
    props = run.find(w("rPr"))
    if props is not None:
        # Both halves need their own copy, or the second one loses the formatting.
        tail.append(copy.deepcopy(props))
    for child in children[index:]:
        run.remove(child)
        tail.append(child)
    parent.insert(list(parent).index(run) + 1, tail)
    return tail


def _split_at(paragraph: ET.Element, position: int) -> None:
    """Make sure no run straddles `position` in the paragraph's rendering."""
    parents = _parents(paragraph)
    for piece in _pieces(paragraph):
        if piece.start == position:
            _split_before(piece.run, piece.node, parents[piece.run])
            return
        if piece.start < position < piece.stop:
            # A tab and a break render as one character each, so only a w:t can
            # have a position strictly inside it.
            cut = position - piece.start
            tail_node = ET.Element(w("t"))
            _set_text(tail_node, piece.text[cut:])
            _set_text(piece.node, piece.text[:cut])
            piece.run.insert(list(piece.run).index(piece.node) + 1, tail_node)
            _split_before(piece.run, tail_node, parents[piece.run])
            return


def _run_is_empty(run: ET.Element) -> bool:
    return all(child.tag == w("rPr") or (child.tag == w("t") and not child.text) for child in run)


def _set_run_text(run: ET.Element, text: str) -> None:
    """Make a run render exactly `text`, leaving anything that is not text alone."""
    nodes = [child for child in run if child.tag in (w("t"), w("tab"), w("br"))]
    for extra in nodes[1:]:
        run.remove(extra)
    if nodes and nodes[0].tag == w("t"):
        _set_text(nodes[0], text)
        return
    for node in nodes:
        run.remove(node)
    node = ET.Element(w("t"))
    _set_text(node, text)
    run.insert(1 if run.find(w("rPr")) is not None else 0, node)


def _insert_at(paragraph: ET.Element, position: int, text: str) -> None:
    """Put `text` into the paragraph's rendering at `position`.

    Args:
        paragraph: The `w:p` to edit.
        position: Where in its rendering the text goes.
        text: What to insert. It joins the run already carrying that position,
            taking its formatting, and the run on the left of a boundary.
    """
    landing = [p for p in _pieces(paragraph) if p.node.tag == w("t") and p.start <= position <= p.stop]
    if not landing:
        run = ET.SubElement(paragraph, w("r"))
        _set_text(ET.SubElement(run, w("t")), text)
        return
    host = landing[0]
    cut = position - host.start
    _set_text(host.node, host.text[:cut] + text + host.text[cut:])


def _replace_span(paragraph: ET.Element, start: int, end: int, replacement: str) -> None:
    """Swap `replacement` in for the paragraph's rendering between `start` and `end`."""
    touched = _covered_runs(paragraph, start, end)
    for index, run in enumerate(touched):
        _set_run_text(run, replacement if index == 0 else "")
    parents = _parents(paragraph)
    for run in touched:
        if _run_is_empty(run):
            parents[run].remove(run)


def _covered_runs(paragraph: ET.Element, start: int, end: int) -> list[ET.Element]:
    """Split runs until the span is a whole number of them, and return those runs.

    Returns:
        The runs the span covers, in document order.
    """
    _split_at(paragraph, end)
    _split_at(paragraph, start)
    covered: list[ET.Element] = []
    for piece in _pieces(paragraph):
        if piece.start >= piece.stop or piece.start < start or piece.stop > end:
            continue
        if piece.run is not None and piece.run not in covered:
            covered.append(piece.run)
    return covered


def _walk(element: ET.Element):
    """Depth-first pre-order over a paragraph, leaving tracked deletions out."""
    for child in element:
        if child.tag in _UNRENDERED:
            continue
        yield child
        yield from _walk(child)


def _comment_anchors(document_root: ET.Element) -> dict[str, str]:
    """The document text each comment range covers, keyed by comment id.

    Keys arrive in document order, which is the order `comments` reports in.
    """
    spans: dict[str, list[str]] = {}
    active: list[str] = []
    for paragraph in document_root.iter(w("p")):
        for node in _walk(paragraph):
            marked = node.get(w("id"))
            if node.tag == w("commentRangeStart"):
                active.append(marked)
                spans.setdefault(marked, [])
            elif node.tag == w("commentRangeEnd") and marked in active:
                active.remove(marked)
            elif node.tag == w("commentReference"):
                # A comment can mark a point rather than a range, and it still has
                # a place in the text that a reader meets it at.
                spans.setdefault(marked, [])
            elif node.tag == w("t"):
                for comment_id in active:
                    spans[comment_id].append(node.text or "")
    return {comment_id: "".join(parts) for comment_id, parts in spans.items()}


def _thread_key(comment: ET.Element) -> str | None:
    """The `w14:paraId` a comment's `commentsExtended` entry keys on.

    None for a comment written before that part existed. This is the package's
    own internal wiring and never leaves this module: the document server rewrites
    it on every save, along with `w:id`, as a position in document order.
    """
    return comment.findall(w("p"))[-1].get(PARA_ID)


def _comment_text(comment: ET.Element) -> str:
    """A comment's body, rendered the way a read reports it."""
    return "\n".join(_paragraph_text(paragraph) for paragraph in comment.findall(w("p")))


def _content_id(comment: ET.Element) -> str:
    """Address a comment by what it says instead of by where it sits.

    Every number the package carries for a comment is a position that the
    document server renumbers on save, so a number read before a save names the
    comment's neighbour after one. Author, date and body text come back from a
    save untouched, so a digest of the three either still names the comment or
    names nothing at all, and a caller holding a stale one is told so.
    """
    material = "\x00".join([comment.get(w("author")) or "", comment.get(w("date")) or "", _comment_text(comment)])
    return hashlib.blake2b(material.encode("utf-8"), digest_size=6).hexdigest()


def _comment_ids(bodies: XmlPart | None) -> dict[ET.Element, str]:
    """Every addressable comment's id, keyed by the element it names.

    Two comments identical in author, date and text take a numbered suffix, in
    the order the part stores them: their content cannot tell them apart, and
    neither can anything else that a save leaves alone. A comment carrying no
    `w14:paraId` is left out, because nothing can key a thread entry to it.
    """
    if bodies is None:
        return {}
    ids: dict[ET.Element, str] = {}
    seen: dict[str, int] = {}
    for element in bodies.root.findall(w("comment")):
        base = _content_id(element)
        count = seen[base] = seen.get(base, 0) + 1
        if _thread_key(element) is not None:
            ids[element] = base if count == 1 else f"{base}-{count}"
    return ids


def _reference_run(comment_id: str) -> ET.Element:
    """The run that carries a comment's `w:commentReference` in the body."""
    run = ET.Element(w("r"))
    props = ET.SubElement(run, w("rPr"))
    ET.SubElement(props, w("rStyle"), {w("val"): "CommentReference"})
    ET.SubElement(run, w("commentReference"), {w("id"): comment_id})
    return run


def _comment_body(comment_id: str, author: str, text: str, para_id: str) -> ET.Element:
    """One `w:comment` for `word/comments.xml`."""
    initials = "".join(word[0] for word in author.split()[:3]).upper()
    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    comment = ET.Element(
        w("comment"),
        {w("id"): comment_id, w("author"): author, w("date"): stamp, w("initials"): initials},
    )
    paragraph = ET.SubElement(comment, w("p"), {PARA_ID: para_id})
    ET.SubElement(ET.SubElement(paragraph, w("pPr")), w("pStyle"), {w("val"): "CommentText"})
    marker = ET.SubElement(paragraph, w("r"))
    ET.SubElement(ET.SubElement(marker, w("rPr")), w("rStyle"), {w("val"): "CommentReference"})
    ET.SubElement(marker, w("annotationRef"))
    _set_text(ET.SubElement(ET.SubElement(paragraph, w("r")), w("t")), text)
    return comment


class Document:
    """A docx package held in memory, part by part."""

    def __init__(self, path: Path, entries: dict[str, bytes], infos: dict[str, zipfile.ZipInfo]) -> None:
        self.path = path
        self._entries = entries
        self._infos = infos
        self._parsed: dict[str, XmlPart] = {}
        self._dirty: set[str] = set()

    @classmethod
    def open(cls, path: Path) -> Document:
        """Read a docx package off disk.

        Callers that took the path from HTTP or from a tool argument resolve it
        through `documents.resolve_existing` first.
        """
        entries: dict[str, bytes] = {}
        infos: dict[str, zipfile.ZipInfo] = {}
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                entries[info.filename] = archive.read(info)
                infos[info.filename] = info
        return cls(path, entries, infos)

    # ── parts ──

    def _part(self, name: str) -> XmlPart | None:
        """Parse a part for reading, or None when the package does not carry it."""
        if name not in self._entries:
            return None
        if name not in self._parsed:
            self._parsed[name] = XmlPart(self._entries[name])
        return self._parsed[name]

    def _edit(self, name: str) -> XmlPart:
        """Parse a part for mutation, marking it as one the save has to rewrite.

        Raises:
            ValueError: The package carries no such part.
        """
        part = self._part(name)
        if part is None:
            raise ValueError(f"docx: no part named {name!r}")
        self._dirty.add(name)
        return part

    def _add_part(self, name: str, data: bytes) -> XmlPart:
        """Add a part the package did not have, and open it for mutation."""
        info = zipfile.ZipInfo(name, date_time=NEW_PART_DATE)
        info.compress_type = zipfile.ZIP_DEFLATED
        self._infos[name] = info
        self._entries[name] = data
        return self._edit(name)

    # ── reading ──

    def _document(self) -> XmlPart:
        """The main document part.

        Raises:
            ValueError: The package carries no `word/document.xml`.
        """
        part = self._part(DOCUMENT_PART)
        if part is None:
            raise ValueError(f"docx: {self.path.name} carries no {DOCUMENT_PART}")
        return part

    def _paragraphs(self) -> list[ET.Element]:
        """Every `w:p` in the document body, in document order."""
        return list(self._document().root.iter(w("p")))

    def text(self) -> str:
        """Render the document body as numbered paragraphs.

        One line per `w:p` in document order, prefixed with a 1-based number that
        the editing operations take as a paragraph anchor. Runs are concatenated,
        a tab renders as a tab, and a line break renders as a single space so that
        a paragraph always occupies exactly one line. Tracked deletions are left
        out, since they are not part of how the document currently reads.

        Returns:
            The rendering, with no trailing newline. An empty paragraph is its
            number and a full stop on their own.
        """
        lines = []
        for number, paragraph in enumerate(self._paragraphs(), start=1):
            body = _paragraph_text(paragraph)
            lines.append(f"{number}. {body}" if body else f"{number}.")
        return "\n".join(lines)

    # ── editing ──

    def _locate(self, anchor: int | str) -> tuple[ET.Element, int, int]:
        """Turn an anchor into the paragraph it names and the span it covers.

        Args:
            anchor: A paragraph number from `text`, which covers that whole
                paragraph, or a literal string, which covers its first occurrence.

        Returns:
            The `w:p`, and where the span starts and ends in its rendering.

        Raises:
            ValueError: No such paragraph, or the string does not appear.
        """
        paragraphs = self._paragraphs()
        if isinstance(anchor, int):
            if not 1 <= anchor <= len(paragraphs):
                raise ValueError(f"docx: no paragraph {anchor} (the document has {len(paragraphs)})")
            paragraph = paragraphs[anchor - 1]
            return paragraph, 0, len(_paragraph_text(paragraph))
        for paragraph in paragraphs:
            found = _paragraph_text(paragraph).find(anchor)
            if found >= 0:
                return paragraph, found, found + len(anchor)
        raise ValueError(f"docx: {anchor!r} does not appear in the document")

    def insert(self, anchor: int | str, text: str) -> None:
        """Insert text after an anchor.

        Args:
            anchor: A paragraph number from `text`, which anchors to that whole
                paragraph, or a literal string, which anchors to its first
                occurrence. The inserted text takes on the anchor's formatting.
            text: What to insert.

        Raises:
            ValueError: No such paragraph, or the string does not appear.
        """
        paragraph, _start, end = self._locate(anchor)
        self._edit(DOCUMENT_PART)
        _insert_at(paragraph, end, text)

    def replace(self, target: str, replacement: str) -> int:
        """Replace every occurrence of a literal string.

        Args:
            target: The text to find, matched against the rendering `text`
                produces. It has to lie within a single paragraph.
            replacement: What to put in its place. An empty string deletes.

        Returns:
            How many occurrences were replaced.

        Raises:
            ValueError: The target does not appear in the document.
        """
        self._edit(DOCUMENT_PART)
        hits = 0
        for paragraph in self._paragraphs():
            starts = [match.start() for match in re.finditer(re.escape(target), _paragraph_text(paragraph))]
            # Right to left, so an earlier replacement cannot shift a later offset.
            for start in reversed(starts):
                _replace_span(paragraph, start, start + len(target), replacement)
            hits += len(starts)
        if not hits:
            raise ValueError(f"docx: {target!r} does not appear in the document")
        return hits

    # ── comments ──

    def _ensure_comment_part(self, name: str) -> XmlPart:
        """Get one of the comment parts, creating it and registering it when absent.

        A package can ship `word/_rels/comments.xml.rels` with nothing behind it,
        so the relationship file says nothing about whether the part is there. A
        part this creates needs its content type override and its relationship
        from `word/document.xml` as well, or Word will not read it.

        Returns:
            The part, open for mutation.
        """
        if name in self._entries:
            return self._edit(name)
        content_type, relationship, skeleton = _COMMENT_PARTS[name]
        part = self._add_part(name, (_DECLARATION + skeleton).encode("utf-8"))

        types = self._edit(CONTENT_TYPES_PART)
        override = q(CONTENT_TYPES, "Override")
        if not any(entry.get("PartName") == f"/{name}" for entry in types.root.findall(override)):
            ET.SubElement(types.root, override, {"PartName": f"/{name}", "ContentType": content_type})

        rels = self._edit(DOCUMENT_RELS_PART)
        tag = q(PACKAGE_RELS, "Relationship")
        existing = rels.root.findall(tag)
        if not any(entry.get("Type") == relationship for entry in existing):
            rel_id = _lowest_unused({entry.get("Id") for entry in existing}, "rId{}")
            target = name.rpartition("/")[2]
            ET.SubElement(rels.root, tag, {"Id": rel_id, "Type": relationship, "Target": target})
        return part

    def _next_reference_id(self) -> str:
        """The lowest unused `w:id`, which only ties a body to its anchor in the body text."""
        bodies = self._part(COMMENTS_PART)
        used = {element.get(w("id")) for element in bodies.root} if bodies is not None else set()
        # An anchor left in document.xml can outlive the body it pointed at.
        used |= {node.get(w("id")) for node in self._document().root.iter(w("commentReference"))}
        return _lowest_unused(used, "{}")

    def _next_para_id(self) -> str:
        """The lowest unused paragraph id, as the eight hex digits the format wants."""
        used = set()
        for name in (DOCUMENT_PART, COMMENTS_PART):
            existing = self._part(name)
            if existing is not None:
                used |= {paragraph.get(PARA_ID) for paragraph in existing.root.iter(w("p"))}
        return _lowest_unused(used, "{:08X}")

    def _addresses(self) -> dict[ET.Element, str]:
        """Every addressable comment's id, keyed by the element it names."""
        return _comment_ids(self._part(COMMENTS_PART))

    def _comment_by_id(self, comment_id: str) -> ET.Element:
        """The `w:comment` a caller-supplied id names.

        Raises:
            ValueError: No comment has that id. An id whose comment was reworded
                or deleted lands here rather than on whatever took its place.
        """
        for element, address in self._addresses().items():
            if address == comment_id:
                return element
        raise ValueError(f"docx: no comment with id {comment_id!r}")

    def _write_comment(self, reference_id: str, text: str, author: str, parent_key: str | None) -> str:
        """Add a comment body and its thread entry, and return the new comment's id.

        `parent_key` is the `w14:paraId` of the comment being replied to, or None
        for one that opens its own thread.
        """
        bodies = self._ensure_comment_part(COMMENTS_PART)
        bodies.ensure_namespace("w14", W14)
        para_id = self._next_para_id()
        body = _comment_body(reference_id, author, text, para_id)
        bodies.root.append(body)
        threads = self._ensure_comment_part(COMMENTS_EXTENDED_PART)
        entry = ET.SubElement(threads.root, q(W15, "commentEx"), {EX_PARA_ID: para_id})
        if parent_key is not None:
            entry.set(PARA_ID_PARENT, parent_key)
        entry.set(DONE, "0")
        return self._addresses()[body]

    def comments(self) -> list[dict]:
        """Every comment in the document, in the order the body text anchors them.

        `date` comes back exactly as the package holds it. The document server
        stamps a comment made in the editor with its own local time and suffixes a
        `Z`, so dates from different authors are not on one clock and cannot be
        compared or used to order a thread.

        Returns:
            One entry per comment, carrying its `id`, `author`, `date`, `text`, the
            `anchor` text its range covers, whether it is `resolved`, and the `id`
            of the comment it is a `parent` reply to, or None when it starts a thread.
            The `id` is derived from the author, date and text, so it survives a
            save and changes when the comment is reworded. It is None for a comment
            old enough to carry no paragraph id, which nothing can thread a reply or
            a done flag to.
        """
        bodies = self._part(COMMENTS_PART)
        if bodies is None:
            return []
        addresses = self._addresses()
        by_key = {_thread_key(element): address for element, address in addresses.items()}
        threads = self._part(COMMENTS_EXTENDED_PART)
        by_para = {entry.get(EX_PARA_ID): entry for entry in threads.root} if threads is not None else {}
        anchors = _comment_anchors(self._document().root)
        # Document order, not the order the part happens to store them in: the
        # dates are not comparable between authors, so where a comment sits in the
        # text is the only sequence a reader can trust. Anything the body text
        # never anchors keeps its stored place, at the end.
        placed = {marked: index for index, marked in enumerate(anchors)}
        ordered = sorted(bodies.root.findall(w("comment")), key=lambda e: placed.get(e.get(w("id")), len(placed)))

        found = []
        for element in ordered:
            key = _thread_key(element)
            entry = by_para.get(key) if key is not None else None
            found.append(
                {
                    "id": addresses.get(element),
                    "author": element.get(w("author")),
                    "date": element.get(w("date")),
                    "text": _comment_text(element),
                    "anchor": anchors.get(element.get(w("id")), ""),
                    "resolved": entry is not None and entry.get(DONE) == "1",
                    "parent": by_key.get(entry.get(PARA_ID_PARENT)) if entry is not None else None,
                }
            )
        return found

    def comment(self, anchor: int | str, text: str, author: str) -> str:
        """Attach a comment to a span of the document.

        Args:
            anchor: A paragraph number from `text`, which comments on the whole
                paragraph, or a literal string, which comments on its first
                occurrence.
            text: The comment body.
            author: Who the comment is from.

        Returns:
            The new comment's id.

        Raises:
            ValueError: No such paragraph, or the string does not appear.
        """
        paragraph, start, end = self._locate(anchor)
        reference_id = self._next_reference_id()
        self._edit(DOCUMENT_PART)
        covered = _covered_runs(paragraph, start, end)
        parents = _parents(paragraph)
        marker_start = ET.Element(w("commentRangeStart"), {w("id"): reference_id})
        marker_end = ET.Element(w("commentRangeEnd"), {w("id"): reference_id})
        if covered:
            # The markers are siblings of the runs, which may sit inside a hyperlink
            # or an insertion rather than directly under the paragraph.
            head = parents[covered[0]]
            head.insert(list(head).index(covered[0]), marker_start)
            tail = parents[covered[-1]]
            at = list(tail).index(covered[-1]) + 1
            tail.insert(at, marker_end)
            tail.insert(at + 1, _reference_run(reference_id))
        else:
            paragraph.extend([marker_start, marker_end, _reference_run(reference_id)])
        return self._write_comment(reference_id, text, author, None)

    def reply(self, comment_id: str, text: str, author: str) -> str:
        """Reply to a comment, in its thread and on its anchor.

        Args:
            comment_id: The comment being replied to.
            text: The reply body.
            author: Who the reply is from.

        Returns:
            The reply's own id.

        Raises:
            ValueError: There is no comment with that id.
        """
        parent = self._comment_by_id(comment_id)
        parent_reference = parent.get(w("id"))
        parent_key = _thread_key(parent)
        reference_id = self._next_reference_id()
        root = self._edit(DOCUMENT_PART).root
        parents = _parents(root)
        # The reply gets its own copy of the parent's anchor, right beside it.
        for tag in (w("commentRangeStart"), w("commentRangeEnd")):
            for node in list(root.iter(tag)):
                if node.get(w("id")) == parent_reference:
                    holder = parents[node]
                    holder.insert(list(holder).index(node) + 1, ET.Element(tag, {w("id"): reference_id}))
        for node in list(root.iter(w("commentReference"))):
            if node.get(w("id")) == parent_reference:
                run = parents[node]
                holder = parents[run]
                holder.insert(list(holder).index(run) + 1, _reference_run(reference_id))
        return self._write_comment(reference_id, text, author, parent_key)

    def resolve(self, comment_id: str) -> None:
        """Mark a comment's thread as done.

        Args:
            comment_id: The comment to resolve.

        Raises:
            ValueError: There is no comment with that id.
        """
        key = _thread_key(self._comment_by_id(comment_id))
        threads = self._ensure_comment_part(COMMENTS_EXTENDED_PART)
        for entry in threads.root:
            if entry.get(EX_PARA_ID) == key:
                entry.set(DONE, "1")
                return
        ET.SubElement(threads.root, q(W15, "commentEx"), {EX_PARA_ID: key, DONE: "1"})

    # ── writing ──

    def save(self) -> None:
        """Write the package back out in one step, re-serializing only what was edited."""
        with replacing(self.path) as tmp, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in self._entries.items():
                if name in self._dirty:
                    data = self._parsed[name].to_bytes()
                archive.writestr(self._infos[name], data)
