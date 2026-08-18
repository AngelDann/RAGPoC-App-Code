from __future__ import annotations

from html.parser import HTMLParser

from markdown_it import MarkdownIt

_md = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])

# Tags that become TipTap marks applied to whatever text they wrap, rather than
# nodes of their own.
_MARK_TAGS = {
    "strong": "bold",
    "b": "bold",
    "em": "italic",
    "i": "italic",
    "s": "strike",
    "del": "strike",
    "code": "code",
}

_HEADING_LEVELS = {f"h{i}": i for i in range(1, 7)}

# Tags whose TipTap node only accepts block-level children (paragraph, list, etc.),
# never bare text/marks directly — matches each node's ProseMirror `content` spec.
_BLOCK_CONTAINER_TAGS = {"blockquote", "li", "ul", "ol", "table", "tr", "th", "td"}


class _TiptapBuilder(HTMLParser):
    """Walks markdown-it's HTML output into the TipTap/ProseMirror JSON doc shape
    the editor expects (StarterKit's default schema, plus the Table extensions
    already registered in PAGE_EDITOR_EXTENSIONS: paragraph, heading, bulletList/
    orderedList/listItem, blockquote, codeBlock, horizontalRule, table/tableRow/
    tableHeader/tableCell, text + marks).

    A full HTML parser (rather than walking markdown-it's token stream directly) keeps
    this in lockstep with the same CommonMark rendering already used for PDF export,
    and gives one place that understands the handful of tags an LLM's markdown output
    actually produces.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.doc: list[dict] = []
        # Parallel stacks: `_stack` is the list new child nodes get appended to;
        # `_wants_block` says whether that container's schema requires block-level
        # children (True) or accepts inline text/marks directly (False); `_implicit`
        # marks frames this parser synthesized itself (see _ensure_inline) so they can
        # be closed automatically rather than needing a matching HTML end tag.
        self._stack: list[list[dict]] = [self.doc]
        self._wants_block: list[bool] = [True]
        self._implicit: list[bool] = [False]

        self._marks: list[dict] = []
        self._link_stack: list[str] = []
        self._in_code_block = False
        self._code_block_buf: list[str] = []
        self._code_block_lang = ""

    def _top(self) -> list[dict]:
        return self._stack[-1]

    def _push_frame(self, node: dict, content: list[dict], wants_block: bool, implicit: bool = False):
        self._top().append(node)
        self._stack.append(content)
        self._wants_block.append(wants_block)
        self._implicit.append(implicit)

    def _pop_frame(self):
        self._stack.pop()
        self._wants_block.pop()
        self._implicit.pop()

    def _close_implicit(self):
        """Closes a dangling auto-inserted paragraph (e.g. bare text directly inside a
        tight list's <li>, which CommonMark renders without a wrapping <p>) before a
        real block-level tag opens or closes at this level.
        """
        while self._implicit[-1]:
            self._pop_frame()

    def _ensure_inline(self):
        """Before appending inline content (text, or opening a mark/link/br tag), make
        sure the current frame actually accepts inline children — auto-opening an
        implicit paragraph if it doesn't (see _close_implicit).
        """
        if self._wants_block[-1]:
            content: list[dict] = []
            self._push_frame({"type": "paragraph", "content": content}, content, wants_block=False, implicit=True)

    def _append_text(self, text: str):
        if not text:
            return
        node = {"type": "text", "text": text}
        marks = list(self._marks)
        if self._link_stack:
            marks = marks + [{"type": "link", "attrs": {"href": self._link_stack[-1]}}]
        if marks:
            node["marks"] = marks
        self._top().append(node)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        if tag == "pre":
            self._close_implicit()
            self._in_code_block = True
            self._code_block_buf = []
            self._code_block_lang = ""
            return
        if tag == "code" and self._in_code_block:
            cls = attrs.get("class", "")
            if cls.startswith("language-"):
                self._code_block_lang = cls[len("language-"):]
            return

        if tag in _HEADING_LEVELS:
            self._close_implicit()
            content: list[dict] = []
            self._push_frame({"type": "heading", "attrs": {"level": _HEADING_LEVELS[tag]}, "content": content}, content, wants_block=False)
            return
        if tag == "p":
            self._close_implicit()
            content = []
            self._push_frame({"type": "paragraph", "content": content}, content, wants_block=False)
            return
        if tag == "blockquote":
            self._close_implicit()
            content = []
            self._push_frame({"type": "blockquote", "content": content}, content, wants_block=True)
            return
        if tag == "ul":
            self._close_implicit()
            content = []
            self._push_frame({"type": "bulletList", "content": content}, content, wants_block=True)
            return
        if tag == "ol":
            self._close_implicit()
            content = []
            node = {"type": "orderedList", "content": content}
            start = attrs.get("start")
            if start and start.isdigit() and int(start) != 1:
                node["attrs"] = {"start": int(start)}
            self._push_frame(node, content, wants_block=True)
            return
        if tag == "li":
            self._close_implicit()
            content = []
            self._push_frame({"type": "listItem", "content": content}, content, wants_block=True)
            return
        if tag == "table":
            self._close_implicit()
            content = []
            self._push_frame({"type": "table", "content": content}, content, wants_block=True)
            return
        if tag == "tr":
            self._close_implicit()
            content = []
            self._push_frame({"type": "tableRow", "content": content}, content, wants_block=True)
            return
        if tag in ("th", "td"):
            self._close_implicit()
            content = []
            node_type = "tableHeader" if tag == "th" else "tableCell"
            self._push_frame({"type": node_type, "content": content}, content, wants_block=True)
            return
        if tag == "hr":
            self._close_implicit()
            self._top().append({"type": "horizontalRule"})
            return
        if tag == "br":
            self._ensure_inline()
            self._top().append({"type": "hardBreak"})
            return
        if tag == "a":
            self._ensure_inline()
            self._link_stack.append(attrs.get("href", ""))
            return
        if tag in _MARK_TAGS:
            self._ensure_inline()
            self._marks.append({"type": _MARK_TAGS[tag]})
            return
        # Unknown/structural tags (e.g. <thead>/<tbody>) — ignore the tag itself but
        # still walk its children against the current container.

    def handle_endtag(self, tag):
        if tag == "pre":
            text = "".join(self._code_block_buf).rstrip("\n")
            attrs = {"language": self._code_block_lang} if self._code_block_lang else {}
            node = {"type": "codeBlock", "attrs": attrs}
            if text:
                node["content"] = [{"type": "text", "text": text}]
            self._top().append(node)
            self._in_code_block = False
            self._code_block_buf = []
            return
        if tag == "code" and self._in_code_block:
            return  # the code-block's own <code>, actually closed via </pre> above

        if tag in _HEADING_LEVELS or tag == "p":
            self._pop_frame()
            return
        if tag in _BLOCK_CONTAINER_TAGS:
            self._close_implicit()
            self._pop_frame()
            return
        if tag == "a":
            if self._link_stack:
                self._link_stack.pop()
            return
        if tag in _MARK_TAGS:
            for i in range(len(self._marks) - 1, -1, -1):
                if self._marks[i]["type"] == _MARK_TAGS[tag]:
                    del self._marks[i]
                    break
            return

    def handle_data(self, data):
        if self._in_code_block:
            self._code_block_buf.append(data)
            return
        if self._wants_block[-1] and not data.strip():
            # Pretty-printing whitespace markdown-it inserts between block tags — not
            # real content, and the current container's schema wouldn't accept a bare
            # text node here anyway.
            return
        self._ensure_inline()
        self._append_text(data)


def markdown_to_tiptap_json(markdown_text: str) -> dict:
    """Converts a markdown string into a TipTap-compatible ProseMirror JSON doc,
    mirroring how the editor itself turns pasted/generated markdown into real
    formatted blocks (headings, lists, bold, code, etc.) instead of a literal
    text node full of `**`/`#`/`-` characters.
    """
    html = _md.render((markdown_text or "").strip())
    builder = _TiptapBuilder()
    builder.feed(html)
    builder.close()
    content = builder.doc
    if not content:
        content = [{"type": "paragraph"}]
    return {"type": "doc", "content": content}
