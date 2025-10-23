"""Replacement strategies for smart file editing.

This module provides various strategies for finding and replacing text in files,
with progressive fallback from exact matching to more flexible approaches.
"""

from abc import ABC, abstractmethod
from typing import Generator, List


class ReplacementStrategy(ABC):
    """Base class for text replacement strategies."""

    @abstractmethod
    def find_matches(self, content: str, search: str) -> Generator[str, None, None]:
        """Find all matches of search string in content.

        Args:
            content: The full file content to search in
            search: The text pattern to find

        Yields:
            Exact strings from content that match the search pattern
        """
        pass


class ExactStrategy(ReplacementStrategy):
    """Exact string matching - the most precise strategy."""

    def find_matches(self, content: str, search: str) -> Generator[str, None, None]:
        """Find exact string matches."""
        if not search:
            return

        start_index = 0
        while True:
            index = content.find(search, start_index)
            if index == -1:
                break
            yield search
            start_index = index + len(search)


class LineTrimmedStrategy(ReplacementStrategy):
    """Match lines with trimmed whitespace.

    Ignores leading/trailing whitespace on each line while preserving
    the original whitespace in the matched content.
    """

    def find_matches(self, content: str, search: str) -> Generator[str, None, None]:
        """Find matches with trimmed line comparison."""
        if not search:
            return

        content_lines = content.split("\n")
        search_lines = search.split("\n")

        # Remove trailing empty line if present
        if search_lines and search_lines[-1] == "":
            search_lines.pop()

        if not search_lines:
            return

        # Search for matching blocks
        for i in range(len(content_lines) - len(search_lines) + 1):
            # Check if all lines match when trimmed
            matches = True
            for j in range(len(search_lines)):
                if content_lines[i + j].strip() != search_lines[j].strip():
                    matches = False
                    break

            if matches:
                # Yield the original content with its whitespace preserved
                match_start = sum(len(line) + 1 for line in content_lines[:i])
                match_end = match_start + sum(
                    len(content_lines[i + j]) + (1 if j < len(search_lines) - 1 else 0)
                    for j in range(len(search_lines))
                )
                yield content[match_start:match_end]


class BlockAnchorStrategy(ReplacementStrategy):
    """Match blocks using first and last lines as anchors.

    Uses Levenshtein distance for fuzzy matching of middle content.
    Requires at least 3 lines (first anchor, middle content, last anchor).
    """

    # Similarity thresholds
    SINGLE_CANDIDATE_THRESHOLD = 0.0
    MULTIPLE_CANDIDATES_THRESHOLD = 0.3

    def find_matches(self, content: str, search: str) -> Generator[str, None, None]:
        """Find matches using anchor lines with fuzzy middle content."""
        search_lines = search.split("\n")

        if search_lines and search_lines[-1] == "":
            search_lines.pop()

        if len(search_lines) < 3:
            return

        content_lines = content.split("\n")
        first_line_search = search_lines[0].strip()
        last_line_search = search_lines[-1].strip()
        search_block_size = len(search_lines)

        # Find all candidate blocks with matching anchors
        candidates: List[tuple[int, int]] = []
        for i in range(len(content_lines)):
            if content_lines[i].strip() != first_line_search:
                continue

            # Look for matching last line
            for j in range(i + 2, len(content_lines)):
                if content_lines[j].strip() == last_line_search:
                    candidates.append((i, j))
                    break

        if not candidates:
            return

        # Handle single candidate with relaxed threshold
        if len(candidates) == 1:
            start_line, end_line = candidates[0]
            actual_block_size = end_line - start_line + 1
            similarity = self._calculate_similarity(
                content_lines, search_lines, start_line, search_block_size, actual_block_size
            )

            if similarity >= self.SINGLE_CANDIDATE_THRESHOLD:
                yield self._extract_block(content, content_lines, start_line, end_line)
            return

        # Handle multiple candidates - find best match
        best_match = None
        max_similarity = -1

        for start_line, end_line in candidates:
            actual_block_size = end_line - start_line + 1
            similarity = self._calculate_similarity(
                content_lines, search_lines, start_line, search_block_size, actual_block_size
            )

            if similarity > max_similarity:
                max_similarity = similarity
                best_match = (start_line, end_line)

        if max_similarity >= self.MULTIPLE_CANDIDATES_THRESHOLD and best_match:
            start_line, end_line = best_match
            yield self._extract_block(content, content_lines, start_line, end_line)

    def _calculate_similarity(
        self,
        content_lines: List[str],
        search_lines: List[str],
        start_line: int,
        search_block_size: int,
        actual_block_size: int,
    ) -> float:
        """Calculate similarity between middle lines using Levenshtein distance."""
        lines_to_check = min(search_block_size - 2, actual_block_size - 2)

        if lines_to_check <= 0:
            return 1.0

        similarity = 0.0
        for j in range(1, min(search_block_size - 1, actual_block_size - 1)):
            content_line = content_lines[start_line + j].strip()
            search_line = search_lines[j].strip()
            max_len = max(len(content_line), len(search_line))

            if max_len == 0:
                continue

            distance = self._levenshtein_distance(content_line, search_line)
            similarity += (1 - distance / max_len) / lines_to_check

            # Early exit if we've already hit threshold
            if similarity >= self.SINGLE_CANDIDATE_THRESHOLD:
                break

        return similarity

    def _extract_block(self, content: str, content_lines: List[str], start_line: int, end_line: int) -> str:
        """Extract block of text from content."""
        match_start = sum(len(line) + 1 for line in content_lines[:start_line])
        match_end = match_start + sum(
            len(content_lines[i]) + (1 if i < end_line else 0) for i in range(start_line, end_line + 1)
        )
        return content[match_start:match_end]

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if not s1 or not s2:
            return max(len(s1), len(s2))

        # Create distance matrix
        m, n = len(s1), len(s2)
        matrix = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize first row and column
        for i in range(m + 1):
            matrix[i][0] = i
        for j in range(n + 1):
            matrix[0][j] = j

        # Fill matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + cost)

        return matrix[m][n]


class WhitespaceNormalizedStrategy(ReplacementStrategy):
    """Match with normalized whitespace.

    Normalizes all whitespace to single spaces and trims before comparison.
    """

    def find_matches(self, content: str, search: str) -> Generator[str, None, None]:
        """Find matches with normalized whitespace."""

        if not search:
            return

        normalized_search = self._normalize_whitespace(search)
        lines = content.split("\n")

        # Check single line matches
        for line in lines:
            if self._normalize_whitespace(line) == normalized_search:
                yield line

        # Check multi-line matches
        search_lines = search.split("\n")
        if len(search_lines) <= 1:
            return

        for i in range(len(lines) - len(search_lines) + 1):
            block = "\n".join(lines[i : i + len(search_lines)])
            if self._normalize_whitespace(block) == normalized_search:
                yield block

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace to single spaces and trim."""
        import re

        return re.sub(r"\s+", " ", text).strip()


class IndentationFlexibleStrategy(ReplacementStrategy):
    """Match ignoring indentation differences.

    Strips minimum common indentation from both search and content blocks
    before comparison.
    """

    def find_matches(self, content: str, search: str) -> Generator[str, None, None]:
        """Find matches ignoring indentation."""
        if not search:
            return

        normalized_search = self._remove_indentation(search)
        content_lines = content.split("\n")
        search_lines = search.split("\n")

        for i in range(len(content_lines) - len(search_lines) + 1):
            block = "\n".join(content_lines[i : i + len(search_lines)])
            if self._remove_indentation(block) == normalized_search:
                yield block

    @staticmethod
    def _remove_indentation(text: str) -> str:
        """Remove minimum common indentation from text."""
        lines = text.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        if not non_empty_lines:
            return text

        # Find minimum indentation
        min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)

        # Remove minimum indentation from all lines
        dedented_lines = [line[min_indent:] if line.strip() else line for line in lines]

        return "\n".join(dedented_lines)


def apply_replacement(content: str, search: str, replace: str, expected_count: int = 1) -> tuple[str, int, str | None]:
    """Apply replacement using progressive fallback strategies.

    Args:
        content: Full file content
        search: Text to find
        replace: Text to replace with
        expected_count: Expected number of matches (default: 1)

    Returns:
        Tuple of (new_content, match_count, error_message)
        error_message is None on success
    """
    if not search:
        return content, 0, "Search string cannot be empty"

    if search == replace:
        return content, 0, "Search and replace strings must be different"

    # Try strategies in order of precision
    strategies = [
        ("exact", ExactStrategy()),
        ("line-trimmed", LineTrimmedStrategy()),
        ("block-anchor", BlockAnchorStrategy()),
        ("whitespace-normalized", WhitespaceNormalizedStrategy()),
        ("indentation-flexible", IndentationFlexibleStrategy()),
    ]

    for strategy_name, strategy in strategies:
        matches = list(strategy.find_matches(content, search))

        if not matches:
            continue

        match_count = len(matches)

        # Check if match count matches expectation
        if match_count != expected_count:
            if expected_count == 1:
                error = (
                    f"Found {match_count} matches but expected 1. "
                    "Either add more context to make the match unique or use expected_replacements parameter."
                )
            else:
                error = f"Found {match_count} matches but expected {expected_count}."
            return content, match_count, error

        # Apply replacement
        new_content = content
        for match in matches:
            new_content = new_content.replace(match, replace, 1)

        return new_content, match_count, None

    return (
        content,
        0,
        "No matches found. Ensure old_string matches file content exactly (including whitespace/indentation). Use read_file_lines to verify.",
    )


def detect_line_ending(content: str) -> str:
    """Detect line ending style in content.

    Args:
        content: File content to analyze

    Returns:
        '\r\n' for Windows-style or '\n' for Unix-style
    """
    return "\r\n" if "\r\n" in content else "\n"


def preserve_line_ending(original_content: str, modified_content: str) -> str:
    """Preserve original line endings in modified content.

    Args:
        original_content: Original file content
        modified_content: Modified content (with \n line endings)

    Returns:
        Modified content with original line ending style
    """
    original_ending = detect_line_ending(original_content)

    # If original had CRLF but modified has only LF, convert
    if original_ending == "\r\n" and "\r\n" not in modified_content:
        return modified_content.replace("\n", "\r\n")

    # If original had LF but modified has CRLF, convert
    if original_ending == "\n" and "\r\n" in modified_content:
        return modified_content.replace("\r\n", "\n")

    return modified_content
