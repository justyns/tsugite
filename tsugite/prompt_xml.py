"""Element builder for the XML-tagged text embedded in LLM prompts.

Blocks like ``<context>``, ``<tsugite_execution_result>`` and ``<message_context>``
are built here rather than by string concatenation, so escaping is one decision
made in one place.

A plain string child is escaped; wrap it in :class:`Raw` to pass it through
verbatim. Verbatim is the right call for file and attachment bodies - a model
reads ``<div>`` better than ``&lt;div&gt;`` - and ``Raw`` makes that a stated
choice instead of a missing call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Union
from xml.sax.saxutils import escape, quoteattr

__all__ = ["El", "Raw", "render_fragments"]


@dataclass(slots=True, frozen=True)
class Raw:
    """Body inserted verbatim, without escaping."""

    text: str


Child = Union["El", Raw, str, None]


@dataclass(slots=True)
class El:
    """One element. ``render`` returns it as text, children nested inside.

    Attributes with a ``None`` value are omitted, so optional attributes can be
    passed unconditionally. ``inline`` keeps the body on the opening line;
    ``void`` renders ``<tag />`` with no body at all.
    """

    tag: str
    children: list[Child] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    inline: bool = False
    void: bool = False

    def render(self, indent: str = "", level: int = 0) -> str:
        out: list[str] = []
        self._write(out, indent, level)
        return "".join(out)

    def _write(self, out: list[str], indent: str, level: int) -> None:
        """Append this element's fragments to `out`.

        Fragments accumulate in one flat list and are joined once, so a large
        attachment body is copied once rather than once per enclosing element.
        """
        pad = indent * level
        out.append(f"{pad}<{self.tag}")
        if self.attrs:
            for key, value in self.attrs.items():
                if value is not None:
                    out.append(f" {key}={_quote(value)}")

        if self.void:
            out.append(" />")
            return

        children = [c for c in self.children if c is not None]
        if not children:
            out.append(f"></{self.tag}>")
            return

        out.append(">")
        if self.inline:
            for child in children:
                _write_child(child, out, indent, 0)
            out.append(f"</{self.tag}>")
            return

        for child in children:
            out.append("\n")
            _write_child(child, out, indent, level + 1)
        out.append(f"\n{pad}</{self.tag}>")


# quoteattr walks the string several times; most attribute values are clean
# identifiers, paths and counts, where it returns the value in plain quotes.
# Newlines, tabs and carriage returns belong here too - quoteattr turns those
# into numeric entities.
_needs_attr_escape = re.compile('[&<>"\n\r\t]').search


def _quote(value: Any) -> str:
    text = str(value)
    return quoteattr(text) if _needs_attr_escape(text) else f'"{text}"'


def _write_child(child: Child, out: list[str], indent: str, level: int) -> None:
    if isinstance(child, El):
        child._write(out, indent, level)
    elif isinstance(child, Raw):
        out.append(child.text)
    else:
        out.append(escape(str(child)))


def render_fragments(items: Iterable[Optional[El]]) -> str:
    """Join sibling elements. Some blocks are a run of siblings, not one root."""
    return "\n".join(el.render() for el in items if el is not None)
