"""Adding, threading and resolving comments inside the docx itself.

Comments are three parts working together: the bodies in `word/comments.xml`, the
anchors in `word/document.xml`, and the thread structure in
`word/commentsExtended.xml`. A package that has never been commented on carries
none of them, and one that has been commented on and then had every comment
deleted can still ship `word/_rels/comments.xml.rels` with nothing behind it, so
the relationship file proves nothing about the part.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile

import pytest
from onlyoffice_helpers import runs
from onlyoffice_helpers import zip_entries as entries
from onlyoffice_helpers import zip_part as part
from tsugite_onlyoffice.docx import (
    COMMENTS_EXTENDED_PART,
    COMMENTS_PART,
    CONTENT_TYPES_PART,
    DOCUMENT_PART,
    DOCUMENT_RELS_PART,
    DONE,
    EX_PARA_ID,
    PARA_ID,
    PARA_ID_PARENT,
    Document,
    w,
)

AUTHOR = "Review Agent"


def names(path):
    with zipfile.ZipFile(path) as archive:
        return archive.namelist()


def commented(path, *operations):
    """Run operations against a document, save in place, and reopen it."""
    document = Document.open(path)
    for operation in operations:
        operation(document)
    document.save()
    return Document.open(path)


def children(path, number):
    """The tag names of one saved paragraph's direct children, in order."""
    paragraph = list(ET.fromstring(part(path, DOCUMENT_PART)).iter(w("p")))[number - 1]
    return [child.tag.rpartition("}")[2] for child in paragraph]


_MARKER = re.compile(rb'(<w:(?:comment|commentRangeStart|commentRangeEnd|commentReference) w:id=")(\d+)"')
_PARA_MARKER = re.compile(rb'(w1[45]:paraId(?:Parent)?=")([0-9A-Fa-f]{8})"')


def _document_order(document_xml):
    """The `w:id` of every comment the body anchors, in the order a reader meets them."""
    order = []
    for node in ET.fromstring(document_xml).iter():
        if node.tag in (w("commentRangeStart"), w("commentReference")) and node.get(w("id")) not in order:
            order.append(node.get(w("id")))
    return order


def _resequenced_para_ids(document_xml, comments_xml):
    """Map every `paraId` in the package to the one a save would hand back.

    A save renumbers them as a plain sequence in document order, so a comment
    inherits whatever number the comment before it in the text used to have.
    """
    paragraphs = {
        element.get(w("id")): [p.get(PARA_ID) for p in element.findall(w("p"))]
        for element in ET.fromstring(comments_xml).findall(w("comment"))
    }
    order = _document_order(document_xml)
    order += [marked for marked in paragraphs if marked not in order]
    renumbered = {}
    for marked in order:
        for para_id in paragraphs.get(marked, []):
            renumbered[para_id] = f"{len(renumbered) + 1:08X}"
    return renumbered


def rewrite(path, change):
    """Rebuild a package in place, passing every part through `change(name, data)`."""
    with zipfile.ZipFile(path) as archive:
        entries = [(info, archive.read(info)) for info in archive.infolist()]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for info, data in entries:
            archive.writestr(info, change(info.filename, data))


def renumber_comment_ids(path):
    """Renumber every comment identifier positionally, the way an ONLYOFFICE save does.

    The document server treats both `w:id` and `w15:paraId` as positions and
    rewrites them on save, consistently across the parts, so threading survives
    and neither number does. An id read before a save therefore names the comment
    next to the one it named, rather than nothing at all.
    """
    resequenced = _resequenced_para_ids(part(path, DOCUMENT_PART), part(path, COMMENTS_PART))

    def shift(match):
        return match.group(1) + str(int(match.group(2)) - 1).encode() + b'"'

    def reseat(match):
        return match.group(1) + resequenced[match.group(2).decode()].encode() + b'"'

    def change(name, data):
        if name in (DOCUMENT_PART, COMMENTS_PART):
            data = _MARKER.sub(shift, data)
        if name in (COMMENTS_PART, COMMENTS_EXTENDED_PART):
            data = _PARA_MARKER.sub(reseat, data)
        return data

    rewrite(path, change)


def edit_comment_text(path, before, after):
    """Reword one comment, the way its author editing it in the editor would."""
    swap = (f"<w:t>{before}</w:t>".encode(), f"<w:t>{after}</w:t>".encode())
    rewrite(path, lambda name, data: data.replace(*swap) if name == COMMENTS_PART else data)


# ── creating the parts ──


def test_the_first_comment_creates_the_comments_part(plain_docx):
    commented(plain_docx, lambda d: d.comment("Quarterly review", "Is this the final title?", AUTHOR))

    assert COMMENTS_PART in names(plain_docx)


def test_the_first_comment_registers_its_content_type(plain_docx):
    commented(plain_docx, lambda d: d.comment(1, "Worth a look.", AUTHOR))

    types = part(plain_docx, CONTENT_TYPES_PART).decode("utf-8")
    assert '<Override PartName="/word/comments.xml"' in types
    assert "wordprocessingml.comments+xml" in types


def test_the_first_comment_registers_its_relationship(plain_docx):
    commented(plain_docx, lambda d: d.comment(1, "Worth a look.", AUTHOR))

    rels = ET.fromstring(part(plain_docx, DOCUMENT_RELS_PART))
    targets = {rel.get("Type").rpartition("/")[2]: rel.get("Target") for rel in rels}
    assert targets["comments"] == "comments.xml"
    ids = [rel.get("Id") for rel in rels]
    assert len(ids) == len(set(ids))


def test_comment_rels_without_a_comments_part_still_gets_one(orphan_comment_rels_docx):
    assert COMMENTS_PART not in names(orphan_comment_rels_docx)

    document = commented(orphan_comment_rels_docx, lambda d: d.comment(1, "First note.", AUTHOR))

    assert COMMENTS_PART in names(orphan_comment_rels_docx)
    assert [entry["text"] for entry in document.comments()] == ["First note."]


def test_a_second_comment_joins_the_part_the_first_one_made(plain_docx):
    document = commented(
        plain_docx,
        lambda d: d.comment(1, "First note.", AUTHOR),
        lambda d: d.comment(2, "Second note.", "Another Reviewer"),
    )

    entries_read = document.comments()
    assert [entry["text"] for entry in entries_read] == ["First note.", "Second note."]
    assert len({entry["id"] for entry in entries_read}) == 2
    assert [entry["author"] for entry in entries_read] == [AUTHOR, "Another Reviewer"]


def test_comments_is_empty_for_a_package_that_has_none(plain_docx):
    assert Document.open(plain_docx).comments() == []


# ── anchoring ──


def test_a_comment_wraps_its_anchor_in_a_range(plain_docx):
    document = commented(plain_docx, lambda d: d.comment("Quarterly review", "Is this the title?", AUTHOR))

    assert children(plain_docx, 1) == ["commentRangeStart", "r", "commentRangeEnd", "r"]
    assert document.comments()[0]["anchor"] == "Quarterly review"


def test_a_comment_anchored_mid_run_splits_it_and_keeps_the_formatting(styled_docx):
    document = commented(styled_docx, lambda d: d.comment("start", "Odd word.", AUTHOR))

    assert runs(styled_docx, 1) == [
        ("Bold ", ["b"]),
        ("start", ["b"]),
        ("", ["rStyle"]),
        (" ", ["b"]),
        ("and italic tail", ["i"]),
    ]
    assert children(styled_docx, 1) == ["r", "commentRangeStart", "r", "commentRangeEnd", "r", "r", "r"]
    assert document.comments()[0]["anchor"] == "start"
    assert document.text().startswith("1. Bold start and italic tail")


def test_a_comment_on_a_paragraph_number_covers_the_whole_paragraph(plain_docx):
    document = commented(plain_docx, lambda d: d.comment(2, "Which pilot?", AUTHOR))

    assert document.comments()[0]["anchor"] == "The pilot shipped on time."


def test_comment_rejects_an_anchor_that_is_not_there(plain_docx):
    with pytest.raises(ValueError, match="does not appear"):
        Document.open(plain_docx).comment("no such sentence", "x", AUTHOR)


# ── threads ──


def test_reply_threads_to_its_parent(plain_docx):
    document = Document.open(plain_docx)
    parent = document.comment(1, "Is this the title?", AUTHOR)
    child = document.reply(parent, "Yes, it is final.", "Another Reviewer")
    document.save()

    assert child != parent
    threads = {entry["id"]: entry["parent"] for entry in Document.open(plain_docx).comments()}
    assert threads == {parent: None, child: parent}


def test_reply_anchors_itself_where_its_parent_is(plain_docx):
    document = Document.open(plain_docx)
    parent = document.comment("Quarterly review", "Is this the title?", AUTHOR)
    child = document.reply(parent, "Yes.", "Another Reviewer")
    document.save()

    anchors = {entry["id"]: entry["anchor"] for entry in Document.open(plain_docx).comments()}
    assert anchors == {parent: "Quarterly review", child: "Quarterly review"}


def test_resolve_marks_the_thread_done(plain_docx):
    document = Document.open(plain_docx)
    comment_id = document.comment(1, "Is this the title?", AUTHOR)
    document.resolve(comment_id)
    document.save()

    assert Document.open(plain_docx).comments()[0]["resolved"] is True
    extended = ET.fromstring(part(plain_docx, COMMENTS_EXTENDED_PART))
    assert [entry.get(DONE) for entry in extended] == ["1"]


def test_an_unresolved_comment_reads_back_unresolved(plain_docx):
    document = commented(plain_docx, lambda d: d.comment(1, "Is this the title?", AUTHOR))

    assert document.comments()[0]["resolved"] is False


def test_reply_and_resolve_reject_an_unknown_comment(plain_docx):
    document = commented(plain_docx, lambda d: d.comment(1, "A note.", AUTHOR))

    with pytest.raises(ValueError, match="no comment"):
        document.reply("404", "x", AUTHOR)
    with pytest.raises(ValueError, match="no comment"):
        document.resolve("404")


def test_comments_come_back_in_the_order_the_document_meets_them(shuffled_comment_docx):
    """The editor stamps local time with a Z, so dates cannot be what orders a read."""
    reading = [(entry["anchor"], entry["text"]) for entry in Document.open(shuffled_comment_docx).comments()]

    assert reading == [
        ("Quarterly review", "On the first paragraph."),
        ("The pilot shipped on time.", "On the second paragraph."),
    ]


def test_a_comment_anchored_at_a_point_reads_where_its_reference_sits(point_comment_docx):
    """A comment left with nothing selected covers no text, and still has a place in it."""
    reading = [(entry["anchor"], entry["text"]) for entry in Document.open(point_comment_docx).comments()]

    assert reading == [
        ("", "Left at a point in the first."),
        ("The pilot shipped on time.", "On the second paragraph."),
    ]


# ── ids that survive a save ──
#
# These build the two comments in the opposite order to the one the document
# anchors them in, so a save's positional renumbering moves numbers between the
# two comments rather than handing each one back what it had.


def test_a_comment_id_still_names_the_same_comment_after_a_save_renumbers(plain_docx):
    """Both number families are positional, so an agent that cached one addresses a neighbour."""
    document = Document.open(plain_docx)
    second = document.comment(2, "Which pilot?", AUTHOR)
    first = document.comment(1, "Is this the title?", AUTHOR)
    document.save()
    renumber_comment_ids(plain_docx)

    reread = {entry["id"]: entry["text"] for entry in Document.open(plain_docx).comments()}
    assert reread == {first: "Is this the title?", second: "Which pilot?"}


def test_a_comment_id_is_not_a_number_the_package_carries(plain_docx):
    """Handing back a `w:id` or a paraId lookalike invites a caller to trust one."""
    document = Document.open(plain_docx)
    comment_id = document.comment(1, "Is this the title?", AUTHOR)
    document.save()

    assert re.fullmatch(r"[0-9a-f]{12}", comment_id), comment_id
    assert comment_id not in part(plain_docx, COMMENTS_PART).decode("utf-8")


def test_reply_and_resolve_reach_the_same_thread_after_a_save_renumbers(plain_docx):
    document = Document.open(plain_docx)
    document.comment(2, "Which pilot?", AUTHOR)
    first = document.comment(1, "Is this the title?", AUTHOR)
    document.save()
    renumber_comment_ids(plain_docx)

    document = Document.open(plain_docx)
    reply = document.reply(first, "Yes, it is final.", "Another Reviewer")
    document.resolve(first)
    document.save()

    threads = {entry["id"]: entry for entry in Document.open(plain_docx).comments()}
    assert threads[first]["text"] == "Is this the title?"
    assert threads[first]["resolved"] is True
    assert threads[reply]["parent"] == first


def test_an_id_whose_comment_changed_names_nothing_rather_than_its_neighbour(plain_docx):
    """A stale id has to fail loudly. Silently landing on the comment that inherited the
    number resolves somebody else's thread, and the agent is never told."""
    document = Document.open(plain_docx)
    stale = document.comment(2, "Which pilot?", AUTHOR)
    document.comment(1, "Is this the title?", AUTHOR)
    document.save()
    edit_comment_text(plain_docx, "Which pilot?", "Which pilot exactly?")
    renumber_comment_ids(plain_docx)

    document = Document.open(plain_docx)
    with pytest.raises(ValueError, match="no comment"):
        document.resolve(stale)
    with pytest.raises(ValueError, match="no comment"):
        document.reply(stale, "Not this one.", AUTHOR)
    assert [entry["resolved"] for entry in document.comments()] == [False, False]


def test_two_identical_comments_are_still_addressable_one_at_a_time(twin_comment_docx):
    """Nothing in the content separates them, so the second takes a numbered suffix."""
    document = Document.open(twin_comment_docx)
    first, second = document.comments()
    assert second["id"] == first["id"] + "-2"

    reply = document.reply(second["id"], "Answering the second one.", AUTHOR)
    document.save()

    threads = {entry["id"]: entry["parent"] for entry in Document.open(twin_comment_docx).comments()}
    assert threads[reply] == second["id"]


def test_a_comment_with_no_thread_entry_has_no_id_to_address_it_by(legacy_comment_docx):
    """There is no handle a save would not renumber, so the read offers none."""
    (entry,) = Document.open(legacy_comment_docx).comments()

    assert entry["id"] is None
    assert entry["text"] == "Pre-2013 note."
    assert entry["anchor"] == "Quarterly review"


def test_reading_a_thread_less_comment_does_not_give_it_one(legacy_comment_docx):
    """Minting a paraId to fill the gap would be a write dressed up as a read."""
    before = part(legacy_comment_docx, COMMENTS_PART)

    document = Document.open(legacy_comment_docx)
    document.comments()
    document.save()

    assert part(legacy_comment_docx, COMMENTS_PART) == before


def test_para_ids_are_unique_and_eight_hex_digits(plain_docx):
    document = Document.open(plain_docx)
    first = document.comment(1, "One.", AUTHOR)
    document.reply(first, "Two.", AUTHOR)
    document.comment(2, "Three.", AUTHOR)
    document.save()

    bodies = ET.fromstring(part(plain_docx, COMMENTS_PART))
    para_ids = [paragraph.get(PARA_ID) for paragraph in bodies.iter(w("p"))]
    assert len(para_ids) == 3
    assert len(set(para_ids)) == 3
    assert all(len(value) == 8 and int(value, 16) > 0 for value in para_ids)
    threaded = [entry.get(EX_PARA_ID) for entry in ET.fromstring(part(plain_docx, COMMENTS_EXTENDED_PART))]
    assert sorted(threaded) == sorted(para_ids)


def test_para_ids_are_derived_the_same_way_every_run(plain_docx):
    document = Document.open(plain_docx)
    first = document.comment(1, "One.", AUTHOR)
    document.reply(first, "Two.", AUTHOR)
    document.save()

    bodies = ET.fromstring(part(plain_docx, COMMENTS_PART))
    assert [paragraph.get(PARA_ID) for paragraph in bodies.iter(w("p"))] == ["00000001", "00000002"]
    threads = ET.fromstring(part(plain_docx, COMMENTS_EXTENDED_PART))
    assert [(e.get(EX_PARA_ID), e.get(PARA_ID_PARENT)) for e in threads] == [
        ("00000001", None),
        ("00000002", "00000001"),
    ]


# ── everything the comment did not touch ──


def test_commenting_rewrites_only_the_parts_it_has_to(styled_docx):
    before = entries(styled_docx)

    commented(styled_docx, lambda d: d.comment("kept sentence", "Kept why?", AUTHOR))

    after = entries(styled_docx)
    rewritten = {name for name in before if after[name] != before[name]}
    assert rewritten == {DOCUMENT_PART, CONTENT_TYPES_PART, DOCUMENT_RELS_PART}
    assert set(after) - set(before) == {COMMENTS_PART, COMMENTS_EXTENDED_PART}


def test_the_rewritten_parts_keep_their_own_namespaces(styled_docx):
    commented(styled_docx, lambda d: d.comment("kept sentence", "Kept why?", AUTHOR))

    body = part(styled_docx, DOCUMENT_PART).decode("utf-8")
    assert 'mc:Ignorable="w14 w15"' in body
    assert 'w14:paraId="1A2B3C4D"' in body
    types = part(styled_docx, CONTENT_TYPES_PART).decode("utf-8")
    assert 'xmlns="http://schemas.openxmlformats.org/package/2006/content-types"' in types
    assert '<Default Extension="rels"' in types
