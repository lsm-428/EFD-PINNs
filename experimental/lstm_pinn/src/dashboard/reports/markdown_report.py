from pathlib import Path
from typing import List, Optional


class MarkdownReport:
    """Simple Markdown format report generator."""

    def __init__(self, title: Optional[str] = None) -> None:
        """Initialize report with optional title.

        Args:
            title: Optional report title
        """
        self._lines: List[str] = []
        if title:
            self.add_title(title)

    def add_title(self, text: str, level: int = 1) -> "MarkdownReport":
        """Add a title/header.

        Args:
            text: Title text
            level: Header level (1-6)

        Returns:
            Self for method chaining
        """
        self._lines.append(f"{'#' * level} {text}")
        return self

    def add_paragraph(self, text: str) -> "MarkdownReport":
        """Add a paragraph.

        Args:
            text: Paragraph text

        Returns:
            Self for method chaining
        """
        self._lines.append(text)
        return self

    def add_list(self, items: List[str], ordered: bool = False) -> "MarkdownReport":
        """Add a list.

        Args:
            items: List items
            ordered: Whether to use ordered list

        Returns:
            Self for method chaining
        """
        for i, item in enumerate(items, 1):
            prefix = f"{i}. " if ordered else "- "
            self._lines.append(f"{prefix}{item}")
        return self

    def add_table(self, headers: List[str], rows: List[List[str]]) -> "MarkdownReport":
        """Add a table.

        Args:
            headers: Column headers
            rows: Table rows (each row is a list of cell values)

        Returns:
            Self for method chaining
        """
        # Header row
        self._lines.append(f"| {' | '.join(headers)} |")
        # Separator row
        self._lines.append(f"| {' | '.join(['---'] * len(headers))} |")
        # Data rows
        for row in rows:
            self._lines.append(f"| {' | '.join(row)} |")
        return self

    def add_code(self, code: str, language: str = "python") -> "MarkdownReport":
        """Add a code block.

        Args:
            code: Code content
            language: Programming language for syntax highlighting

        Returns:
            Self for method chaining
        """
        self._lines.append(f"```{language}")
        self._lines.append(code)
        self._lines.append("```")
        return self

    def add_image(self, path: str, alt: str = "") -> "MarkdownReport":
        """Add an image link (relative path).

        Args:
            path: Relative path to image file
            alt: Alternative text

        Returns:
            Self for method chaining
        """
        self._lines.append(f"![{alt}]({path})")
        return self

    def add_blank_line(self) -> "MarkdownReport":
        """Add a blank line.

        Returns:
            Self for method chaining
        """
        self._lines.append("")
        return self

    def add_section(self, title: str, content: str) -> "MarkdownReport":
        """Add a section with title and content.

        Args:
            title: Section title
            content: Section content

        Returns:
            Self for method chaining
        """
        self.add_title(title, level=2)
        self.add_paragraph(content)
        return self

    def to_markdown(self) -> str:
        """Convert to Markdown string.

        Returns:
            Complete Markdown content
        """
        # Remove trailing blank lines
        while self._lines and self._lines[-1] == "":
            self._lines.pop()
        return "\n".join(self._lines)

    def save(self, filepath: str) -> "MarkdownReport":
        """Save to file.

        Args:
            filepath: Output file path

        Returns:
            Self for method chaining
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        return self
