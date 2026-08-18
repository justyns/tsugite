"""Opening a docx, reading it, and putting it back the way it came.

ElementTree rewrites namespace prefixes and drops declarations it thinks are
unused, so a save that re-serializes parts it did not need to touch quietly
strips `w14`, `w15` and `mc:Ignorable` off a Word document. Hence the
byte-identity assertion on every entry.
"""

import pytest
from onlyoffice_helpers import zip_entries as entries
from onlyoffice_helpers import zip_part as part
from tsugite_onlyoffice.documents import replacing
from tsugite_onlyoffice.docx import Document, XmlPart

# ── round tripping ──


def test_saving_an_untouched_document_reproduces_every_entry(styled_docx):
    before = entries(styled_docx)

    Document.open(styled_docx).save()

    after = entries(styled_docx)
    assert list(after) == list(before)
    assert after == before


def test_reserializing_a_part_keeps_its_namespace_declarations(styled_docx):
    original = part(styled_docx, "word/document.xml")

    assert XmlPart(original).to_bytes() == original


def test_reserializing_a_part_keeps_a_self_closing_root(orphan_comment_rels_docx):
    original = part(orphan_comment_rels_docx, "word/_rels/comments.xml.rels")

    assert XmlPart(original).to_bytes() == original


# ── the text rendering ──


def test_text_numbers_every_paragraph(plain_docx):
    assert Document.open(plain_docx).text() == (
        "1. Quarterly review\n2. The pilot shipped on time.\n3. Costs held flat."
    )


def test_text_keeps_each_paragraph_on_a_single_line(styled_docx):
    assert Document.open(styled_docx).text() == "\n".join(
        [
            "1. Bold start and italic tail",
            "2. Second\tafter a tab",
            "3. Line one line two",
            "4. kept sentence",
            "5.",
        ]
    )


def test_text_leaves_out_tracked_deletions(styled_docx):
    assert "removed sentence" not in Document.open(styled_docx).text()


def test_two_writers_on_one_document_do_not_share_a_temp_file(tmp_path):
    """A save from the document server does not hold the turn's lock, so two writes to
    one document overlap. A temp name they share means one renames the other's file away."""
    target = tmp_path / "notes.docx"
    target.write_bytes(b"before")
    with replacing(target) as first, replacing(target) as second:
        assert first != second, "overlapping writes need their own temp files"
        first.write_bytes(b"one")
        second.write_bytes(b"two")


def test_a_failed_write_leaves_neither_the_document_nor_a_temp_file_behind(tmp_path):
    """The temp file lives beside the document, so one left per failure accumulates in
    the documents directory, where nothing lists it and nothing cleans it up."""
    target = tmp_path / "notes.docx"
    target.write_bytes(b"before")

    with pytest.raises(RuntimeError):
        with replacing(target) as tmp:
            tmp.write_bytes(b"half a document")
            raise RuntimeError("the write failed")

    assert target.read_bytes() == b"before"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["notes.docx"]
