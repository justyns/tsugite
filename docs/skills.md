# Skill Frontmatter: disable-model-invocation

`disable-model-invocation: true` in a skill's `SKILL.md` frontmatter keeps a model from loading the skill on its own initiative.

```yaml
---
name: refine
description: Deep code quality pass
disable-model-invocation: true
---
```

Only the hyphenated key is read. The frontmatter loader is plain YAML with no key normalisation, so a skill that spells it `disable_model_invocation` is not gated.

| Path | Gated skill |
|------|-------------|
| `list_available_skills()` | Omitted |
| `<available_skills>` block (`get_skills_for_template`) | Omitted |
| `load_skill("name")` | Returns an error saying the skill is invocation-gated |
| `auto_load_skills:` in agent frontmatter | Loads normally |
| Daemon sticky skills already loaded | Keep renewing |
| `triggers:` keyword match | Not loaded |

A gated skill is also left out of the "Available skills" list in the not-found error from `load_skill`.
