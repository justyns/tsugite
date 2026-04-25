"""If an HTML-entity-escaped XML observation (`&lt;tsugite_execution_result...`)
ever makes it into `exec()` as source code, that's a bug — the escape is
meant for LLM-facing XML, not for local execution. A narrow tripwire catches
this regression loudly instead of producing a confusing SyntaxError at
compile time.
"""

import pytest

from tsugite.core.executor import LocalExecutor


@pytest.mark.asyncio
async def test_escaped_observation_is_rejected():
    executor = LocalExecutor()
    code = (
        "&lt;tsugite_execution_result status=&quot;success&quot;&gt;\n"
        "&lt;output&gt;ok&lt;/output&gt;\n"
        "&lt;/tsugite_execution_result&gt;"
    )
    result = await executor.execute(code)
    assert result.error is not None
    assert "escape" in result.error.lower() or "html" in result.error.lower()


@pytest.mark.asyncio
async def test_legitimate_code_with_ampersand_literal_runs():
    """User code that happens to contain `&lt;` as a string literal is not
    rejected — the tripwire only fires when the *source itself* looks escaped.
    """
    executor = LocalExecutor()
    code = "x = '&lt;b&gt;hello&lt;/b&gt;'\nprint(x)"
    result = await executor.execute(code)
    assert result.error is None, f"unexpected rejection: {result.error}"
    assert "&lt;b&gt;hello&lt;/b&gt;" in result.output
