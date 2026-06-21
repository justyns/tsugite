"""Phase 1b: rich context blocks (pre_context_build) and pre_llm_call message mutation."""

import pytest

from tsugite.hooks import Block, HookRule, HooksConfig, collect_context_blocks, render_blocks


def test_render_blocks_xml():
    out = render_blocks([Block(tag="memory", body="remember this", attributes={"source": "USER.md"})])
    assert out == '<memory source="USER.md">\nremember this\n</memory>'


def test_collect_context_blocks_sorts_by_priority_and_aliases_rag():
    blocks = [Block(tag="low", priority=1), Block(tag="high", priority=10)]
    result = collect_context_blocks(blocks, rag_context="legacy rag")
    tags = [b.tag for b in result]
    assert tags[0] == "high"  # highest priority first
    assert "context" in tags  # rag_context aliased to a <context> block
    rag_block = next(b for b in result if b.tag == "context")
    assert rag_block.body == "legacy rag"


def test_collect_context_blocks_empty():
    assert collect_context_blocks(None) == []
    assert collect_context_blocks([]) == []


@pytest.mark.asyncio
async def test_make_pre_llm_call_callback_mutates_messages(monkeypatch, tmp_path):
    """A python pre_llm_call hook mutates the outgoing messages list in place."""
    import tsugite.hooks as hooks_mod
    from tsugite.agent_runner import runner

    def inject(ctx):
        ctx["messages"].append({"role": "system", "content": "late-injected"})

    cfg = HooksConfig(pre_llm_call=[HookRule(type="python", hook_callable=inject, name="inj")])
    monkeypatch.setattr(hooks_mod, "load_hooks_config", lambda wd: cfg)

    callback = runner._make_pre_llm_call_callback(tmp_path, "agent")
    assert callback is not None

    messages = [{"role": "user", "content": "hi"}]
    await callback(messages, "openai:gpt-4o-mini")
    assert any(m["content"] == "late-injected" for m in messages)


def test_make_pre_llm_call_callback_none_without_hooks(monkeypatch, tmp_path):
    import tsugite.hooks as hooks_mod
    from tsugite.agent_runner import runner

    monkeypatch.setattr(hooks_mod, "load_hooks_config", lambda wd: HooksConfig())
    assert runner._make_pre_llm_call_callback(tmp_path, "agent") is None


@pytest.mark.asyncio
async def test_pre_context_build_hook_block_renders(monkeypatch, tmp_path):
    """A python pre_context_build hook can append a Block that ends up in the prompt."""
    import tsugite.hooks as hooks_mod
    from tsugite.hooks import fire_hooks

    def mem_hook(ctx):
        ctx["blocks"].append(Block(tag="memory", body="user likes tea", attributes={"source": "USER.md"}))

    cfg = HooksConfig(pre_context_build=[HookRule(type="python", hook_callable=mem_hook, name="mem")])
    monkeypatch.setattr(hooks_mod, "load_hooks_config", lambda wd: cfg)

    blocks: list = []
    await fire_hooks(tmp_path, "pre_context_build", {"blocks": blocks})
    rendered = render_blocks(collect_context_blocks(blocks))
    assert '<memory source="USER.md">' in rendered
    assert "user likes tea" in rendered


@pytest.mark.asyncio
async def test_pre_llm_call_callback_invoked_by_agent():
    """The agent fires pre_llm_call before the provider call; mutations reach the provider."""
    from tsugite.core.agent import TsugiteAgent
    from tsugite.providers.base import CompletionResponse, Usage

    agent = TsugiteAgent(model_string="openai:gpt-4o-mini", tools=[])

    async def cb(messages, model):
        messages.append({"role": "system", "content": "from-pre-llm-call"})

    agent._pre_llm_call = cb

    seen = {}

    async def capture_acompletion(messages, model, stream, **kwargs):
        seen["messages"] = list(messages)
        return CompletionResponse(content="Final answer: ok", usage=Usage())

    agent._provider.acompletion = capture_acompletion
    await agent._provider_turn(messages=[{"role": "user", "content": "hi"}], turn_num=0, stream=False)

    assert any(m["content"] == "from-pre-llm-call" for m in seen["messages"])


def test_context_blocks_template_renders_under_strict_undefined():
    """The render guard emits blocks when present and nothing (no error) when absent."""
    from tsugite.renderer import AgentRenderer

    snippet = "{% if context_blocks is defined and context_blocks %}{{ context_blocks }}{% endif %}"
    renderer = AgentRenderer()
    assert "<memory>tea</memory>" in renderer.render(snippet, {"context_blocks": "<memory>tea</memory>"})
    assert renderer.render(snippet, {}).strip() == ""


def test_default_md_uses_context_blocks_slot():
    """default.md renders the blocks region and no longer hard-codes the rag_context slot."""
    from tsugite.agent_inheritance import get_builtin_agents_path

    content = (get_builtin_agents_path() / "default.md").read_text()
    assert "context_blocks" in content
    assert "rag_context" not in content
