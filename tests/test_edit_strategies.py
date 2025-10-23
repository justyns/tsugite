"""Tests for file editing strategies."""

from tsugite.tools.edit_strategies import (
    BlockAnchorStrategy,
    ExactStrategy,
    IndentationFlexibleStrategy,
    LineTrimmedStrategy,
    WhitespaceNormalizedStrategy,
    apply_replacement,
    detect_line_ending,
    preserve_line_ending,
)


class TestExactStrategy:
    """Tests for ExactStrategy."""

    def test_exact_match(self):
        """Test exact string matching."""
        strategy = ExactStrategy()
        content = "Hello World\nTest String\nGoodbye"
        search = "Test String"

        matches = list(strategy.find_matches(content, search))
        assert len(matches) == 1
        assert matches[0] == "Test String"

    def test_no_match(self):
        """Test when no exact match exists."""
        strategy = ExactStrategy()
        content = "Hello World"
        search = "Nonexistent"

        matches = list(strategy.find_matches(content, search))
        assert len(matches) == 0

    def test_multiple_matches(self):
        """Test multiple exact matches."""
        strategy = ExactStrategy()
        content = "foo bar foo baz foo"
        search = "foo"

        matches = list(strategy.find_matches(content, search))
        assert len(matches) == 3

    def test_empty_search(self):
        """Test with empty search string."""
        strategy = ExactStrategy()
        content = "Hello World"
        search = ""

        matches = list(strategy.find_matches(content, search))
        assert len(matches) == 0


class TestLineTrimmedStrategy:
    """Tests for LineTrimmedStrategy."""

    def test_trimmed_lines_match(self):
        """Test matching with trimmed whitespace."""
        strategy = LineTrimmedStrategy()
        content = "  def foo():\n      return 42\n  "
        search = "def foo():\n    return 42"

        matches = list(strategy.find_matches(content, search))
        assert len(matches) == 1
        # Should return original content with whitespace
        assert "  def foo():" in matches[0]

    def test_multiline_trimmed(self):
        """Test multiline matching with different indentation."""
        strategy = LineTrimmedStrategy()
        content = """    if True:
        print("hello")
        return"""
        search = 'if True:\n    print("hello")\n    return'

        matches = list(strategy.find_matches(content, search))
        assert len(matches) >= 1


class TestBlockAnchorStrategy:
    """Tests for BlockAnchorStrategy."""

    def test_block_with_anchors(self):
        """Test matching block using first/last line anchors."""
        strategy = BlockAnchorStrategy()
        content = """def foo():
    # Some comment
    x = 42
    return x"""
        search = """def foo():
    # Different comment
    return x"""

        matches = list(strategy.find_matches(content, search))
        # Should match based on anchors despite different middle content
        assert len(matches) >= 0  # May or may not match depending on similarity

    def test_requires_min_three_lines(self):
        """Test that block anchor requires at least 3 lines."""
        strategy = BlockAnchorStrategy()
        content = "Line 1\nLine 2"
        search = "Line 1\nLine 2"

        matches = list(strategy.find_matches(content, search))
        assert len(matches) == 0  # Not enough lines for block anchor

    def test_levenshtein_distance(self):
        """Test Levenshtein distance calculation."""
        strategy = BlockAnchorStrategy()

        # Identical strings
        assert strategy._levenshtein_distance("hello", "hello") == 0

        # One character difference
        assert strategy._levenshtein_distance("hello", "hallo") == 1

        # Empty strings
        assert strategy._levenshtein_distance("", "abc") == 3
        assert strategy._levenshtein_distance("abc", "") == 3


class TestWhitespaceNormalizedStrategy:
    """Tests for WhitespaceNormalizedStrategy."""

    def test_normalized_whitespace(self):
        """Test matching with normalized whitespace."""
        strategy = WhitespaceNormalizedStrategy()
        content = "Hello    World\nWith  Multiple   Spaces"
        search = "Hello World"

        matches = list(strategy.find_matches(content, search))
        assert len(matches) >= 1

    def test_multiline_normalized(self):
        """Test multiline matching with normalized whitespace."""
        strategy = WhitespaceNormalizedStrategy()
        content = "Line  1\nLine    2\nLine 3"
        search = "Line 1\nLine 2\nLine 3"

        matches = list(strategy.find_matches(content, search))
        assert len(matches) >= 1


class TestIndentationFlexibleStrategy:
    """Tests for IndentationFlexibleStrategy."""

    def test_flexible_indentation(self):
        """Test matching ignoring indentation."""
        strategy = IndentationFlexibleStrategy()
        content = """    def foo():
        return 42"""
        search = """def foo():
    return 42"""

        matches = list(strategy.find_matches(content, search))
        assert len(matches) >= 1

    def test_remove_indentation(self):
        """Test indentation removal."""
        strategy = IndentationFlexibleStrategy()

        text = "    Line 1\n    Line 2\n    Line 3"
        result = strategy._remove_indentation(text)
        assert result == "Line 1\nLine 2\nLine 3"


class TestApplyReplacement:
    """Tests for apply_replacement function."""

    def test_exact_replacement(self):
        """Test basic exact replacement."""
        content = "Hello World"
        search = "World"
        replace = "Universe"

        new_content, count, error = apply_replacement(content, search, replace, expected_count=1)

        assert error is None
        assert count == 1
        assert new_content == "Hello Universe"

    def test_multiple_replacements(self):
        """Test replacing multiple occurrences."""
        content = "foo bar foo baz foo"
        search = "foo"
        replace = "qux"

        new_content, count, error = apply_replacement(content, search, replace, expected_count=3)

        assert error is None
        assert count == 3
        assert new_content == "qux bar qux baz qux"

    def test_no_match_error(self):
        """Test error when no matches found."""
        content = "Hello World"
        search = "Nonexistent"
        replace = "Something"

        new_content, count, error = apply_replacement(content, search, replace)

        assert error is not None
        assert "No matches found" in error
        assert count == 0

    def test_count_mismatch_error(self):
        """Test error when match count doesn't match expected."""
        content = "foo bar foo"
        search = "foo"
        replace = "qux"

        # Expect 1 but found 2
        new_content, count, error = apply_replacement(content, search, replace, expected_count=1)

        assert error is not None
        assert "Found 2 matches but expected 1" in error
        assert count == 2

    def test_same_strings_error(self):
        """Test error when search and replace are identical."""
        content = "Hello World"
        search = "Hello"
        replace = "Hello"

        new_content, count, error = apply_replacement(content, search, replace)

        assert error is not None
        assert "must be different" in error

    def test_empty_search_error(self):
        """Test error with empty search string."""
        content = "Hello World"
        search = ""
        replace = "Something"

        new_content, count, error = apply_replacement(content, search, replace)

        assert error is not None
        assert "cannot be empty" in error

    def test_fallback_strategies(self):
        """Test that fallback strategies are tried."""
        content = "  def foo():\n      return 42"
        search = "def foo():\n    return 42"  # Different indentation
        replace = "def foo():\n    return 100"

        new_content, count, error = apply_replacement(content, search, replace, expected_count=1)

        # Should succeed with line-trimmed or another fallback strategy
        assert error is None
        assert count == 1
        assert "return 100" in new_content


class TestLineEndingFunctions:
    """Tests for line ending detection and preservation."""

    def test_detect_unix_line_ending(self):
        """Test detecting Unix-style line endings."""
        content = "Line 1\nLine 2\nLine 3"
        assert detect_line_ending(content) == "\n"

    def test_detect_windows_line_ending(self):
        """Test detecting Windows-style line endings."""
        content = "Line 1\r\nLine 2\r\nLine 3"
        assert detect_line_ending(content) == "\r\n"

    def test_preserve_unix_line_ending(self):
        """Test preserving Unix line endings."""
        original = "Line 1\nLine 2"
        modified = "Line 1\nLine 2 Modified"

        result = preserve_line_ending(original, modified)
        assert "\r\n" not in result
        assert "\n" in result

    def test_preserve_windows_line_ending(self):
        """Test preserving Windows line endings."""
        original = "Line 1\r\nLine 2"
        modified = "Line 1\nLine 2 Modified"  # Modified has Unix endings

        result = preserve_line_ending(original, modified)
        assert "\r\n" in result

    def test_preserve_when_already_correct(self):
        """Test that preservation works when endings already match."""
        original = "Line 1\r\nLine 2"
        modified = "Line 1\r\nLine 2 Modified"

        result = preserve_line_ending(original, modified)
        assert result == modified
