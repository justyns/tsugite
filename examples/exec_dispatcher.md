---
name: exec_dispatcher
model: openai:gpt-4o-mini
tools: [read_file, list_files, return_value]
---

<!-- tsu:ignore -->
# Exec Dispatcher Demo

Demonstrates the `<!-- tsu:exec -->` directive: arbitrary Python at agent render
time, with the return value bound to a Jinja variable.

The pattern below scans an `inbox/` directory of JSON webhook payloads and
returns a list of `{path, source}` entries. The LLM step then chooses how to
react based on what was found - no LLM round-trip needed for the inventory.

Run:
    mkdir -p inbox
    echo '{"source":"github"}' > inbox/alpha.json
    echo '{"source":"slack"}'  > inbox/beta.json
    tsu render exec_dispatcher.md "process inbox"
    tsu run    exec_dispatcher.md "process inbox"
<!-- /tsu:ignore -->

<!-- tsu:exec name="inventory" assign="targets" -->
import json
import os

out = []
inbox = "inbox"
if os.path.isdir(inbox):
    for path in sorted(os.listdir(inbox)):
        if path.endswith(".json"):
            payload = json.loads(open(f"{inbox}/{path}").read())
            out.append({"path": path, "source": payload.get("source", "unknown")})
out
<!-- /tsu:exec -->

# Inbox Dispatcher

Task: {{ user_prompt }}

Inventory ({{ targets | length }} entries):
{% for t in targets %}- `{{ t.path }}` from `{{ t.source }}`
{% endfor %}

For each entry, briefly note what handler an autonomous loop would dispatch to.
Call `return_value` with a one-line summary.
