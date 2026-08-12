"""Inserting and replacing text without flattening the runs it lands in.

Text that reads as one sentence is often several `w:r` elements with different
`w:rPr` on each, so an edit that lands mid-run has to split the run and carry the
formatting onto both halves. These tests read the runs back out of the saved
package rather than trusting the rendering, which cannot see formatting.
"""

import pytest
from onlyoffice_helpers import runs
from onlyoffice_helpers import zip_entries as entries
from onlyoffice_helpers import zip_part as part
from tsugite_onlyoffice.docx import DOCUMENT_PART, Document


def edited(path, mutate):
    """Apply an edit, save in place, and hand back the reopened rendering."""
    document = Document.open(path)
    mutate(document)
    document.save()
    return Document.open(path).text()


# ── replace ──


def test_replace_inside_a_single_run(plain_docx):
    assert "2. The rollout shipped on time." in edited(plain_docx, lambda d: d.replace("pilot", "rollout"))


def test_replace_spanning_two_runs(styled_docx):
    text = edited(styled_docx, lambda d: d.replace("start and", "middle"))

    assert text.startswith("1. Bold middle italic tail")
    # The italic run is only half covered, so its uncovered half has to survive.
    assert runs(styled_docx, 1) == [("Bold ", ["b"]), ("middle", ["b"]), (" italic tail", ["i"])]


def test_replace_at_a_run_boundary(styled_docx):
    text = edited(styled_docx, lambda d: d.replace("and italic tail", "plain tail"))

    assert text.startswith("1. Bold start plain tail")
    assert runs(styled_docx, 1) == [("Bold start ", ["b"]), ("plain tail", ["i"])]


def test_replace_keeps_formatting_on_both_halves_of_a_split(styled_docx):
    text = edited(styled_docx, lambda d: d.replace("start", "finish"))

    assert text.startswith("1. Bold finish and italic tail")
    assert runs(styled_docx, 1) == [
        ("Bold ", ["b"]),
        ("finish", ["b"]),
        (" ", ["b"]),
        ("and italic tail", ["i"]),
    ]


def test_replace_keeps_the_space_a_split_left_at_an_edge(styled_docx):
    edited(styled_docx, lambda d: d.replace("start", "finish"))

    assert '<w:t xml:space="preserve"> </w:t>' in part(styled_docx, DOCUMENT_PART).decode("utf-8")


def test_replace_handles_every_occurrence(repeated_docx):
    document = Document.open(repeated_docx)

    assert document.replace("draft", "revision") == 3

    document.save()
    assert Document.open(repeated_docx).text() == (
        "1. The revision went out on Monday and the revision came back on Friday.\n"
        "2. A third revision follows next week."
    )


def test_replace_with_nothing_deletes(plain_docx):
    assert "2. The shipped on time." in edited(plain_docx, lambda d: d.replace("pilot ", ""))


def test_replace_rejects_a_target_that_is_not_there(plain_docx):
    with pytest.raises(ValueError, match="does not appear"):
        Document.open(plain_docx).replace("quarterly review", "annual review")


# ── insert ──


def test_insert_at_the_end_of_a_paragraph(plain_docx):
    assert edited(plain_docx, lambda d: d.insert(3, " Signed off.")).endswith("3. Costs held flat. Signed off.")


def test_insert_after_a_string_anchor(plain_docx):
    assert edited(plain_docx, lambda d: d.insert("Quarterly review", " (draft)")).startswith(
        "1. Quarterly review (draft)"
    )


def test_insert_takes_the_formatting_of_the_run_it_lands_in(styled_docx):
    edited(styled_docx, lambda d: d.insert("Bold start", " and bolder"))

    assert runs(styled_docx, 1)[0] == ("Bold start and bolder ", ["b"])


def test_insert_fills_an_empty_paragraph(styled_docx):
    assert edited(styled_docx, lambda d: d.insert(5, "A closing thought.")).endswith("5. A closing thought.")


def test_insert_rejects_an_anchor_that_is_not_there(plain_docx):
    with pytest.raises(ValueError, match="does not appear"):
        Document.open(plain_docx).insert("no such sentence", "x")


def test_insert_rejects_a_paragraph_that_is_not_there(plain_docx):
    with pytest.raises(ValueError, match="no paragraph 9"):
        Document.open(plain_docx).insert(9, "x")


# ── everything the edit did not touch ──


def test_an_edit_rewrites_the_document_part_and_nothing_else(styled_docx):
    before = entries(styled_docx)

    edited(styled_docx, lambda d: d.replace("kept sentence", "revised sentence"))

    after = entries(styled_docx)
    assert list(after) == list(before)
    rewritten = {name for name in before if after[name] != before[name]}
    assert rewritten == {DOCUMENT_PART}


def test_the_rewritten_document_part_keeps_its_namespace_declarations(styled_docx):
    edited(styled_docx, lambda d: d.replace("kept sentence", "revised sentence"))

    body = part(styled_docx, DOCUMENT_PART).decode("utf-8")
    assert body.startswith('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n')
    assert 'mc:Ignorable="w14 w15"' in body
    assert 'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml"' in body
    assert 'xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas"' in body
    assert 'w14:paraId="1A2B3C4D"' in body


def test_an_edit_leaves_tracked_deletions_in_place(styled_docx):
    edited(styled_docx, lambda d: d.replace("kept sentence", "revised sentence"))

    assert "<w:delText>removed sentence </w:delText>" in part(styled_docx, DOCUMENT_PART).decode("utf-8")
