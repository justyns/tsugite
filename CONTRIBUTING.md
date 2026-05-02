# Contributing

Quick guide for contributors. See [AGENTS.md](AGENTS.md) for architecture and dev commands.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/). Format:

```
<type>(<scope>): <subject>
```

Keep the subject on one line, lowercase, no trailing period. Body is optional and rarely needed.

### Allowed types

These match the groups in `cliff.toml` so release notes generate cleanly:

| Type       | Use for                                                      |
|------------|--------------------------------------------------------------|
| `feat`     | User-visible new capability                                  |
| `fix`      | Bug fix                                                      |
| `refactor` | Code restructure with no behavior change                     |
| `docs`     | Documentation only                                           |
| `test`     | Tests only                                                   |
| `chore`    | Deps, version bumps, formatting, tooling, lockfile           |
| `ci`       | CI/release pipeline only                                     |
| `revert`   | Reverts a prior commit                                       |

Don't use other prefixes (`wip:`, `bump:`, `lint:`, `style:`, etc.). Squash WIP commits before merging. Roll lint/format/version bumps into `chore:`.

### Breaking changes

Append `!` to the type or add a `BREAKING CHANGE:` footer:

```
feat!: drop Python 3.10 support
```

### Scopes

Scopes are optional, but should be limited to one of these:

`webui`, `daemon`, `agent`, `cli`, `skills`, `history`, `sandbox`

### Examples

```
feat(webui): show session topic inline in sidebar
fix(agent): use ast.parse to locate code block close fence
refactor(history): per-event JSONL log replaces Turn aggregate
chore: bump lxml to 6.1.0 (CVE-2026-41066)
docs: add plugin docs
```
