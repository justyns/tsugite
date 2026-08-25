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

from dataclasses import dataclass, field
from typing import Any, Iterable, Union
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
        pad = indent * level
        attrs = "".join(f" {k}={quoteattr(str(v))}" for k, v in self.attrs.items() if v is not None)

        if self.void:
            return f"{pad}<{self.tag}{attrs} />"

        parts = [_render_child(c, indent, level + 1, self.inline) for c in self.children if c is not None]
        if not parts:
            return f"{pad}<{self.tag}{attrs}></{self.tag}>"

        if self.inline:
            return f"{pad}<{self.tag}{attrs}>{''.join(parts)}</{self.tag}>"
        body = "\n".join(parts)
        return f"{pad}<{self.tag}{attrs}>\n{body}\n{pad}</{self.tag}>"


def _render_child(child: Child, indent: str, level: int, inline: bool) -> str:
    if isinstance(child, El):
        return child.render(indent, 0 if inline else level)
    if isinstance(child, Raw):
        return child.text
    return escape(str(child))


def render_fragments(items: Iterable[Union[El, Raw, str, None]], sep: str = "\n") -> str:
    """Join top-level items. Observations are several sibling blocks, not one root."""
    parts = []
    for item in items:
        if item is None:
            continue
        text = item.render() if isinstance(item, El) else (item.text if isinstance(item, Raw) else str(item))
        if text:
            parts.append(text)
    return sep.join(parts)
