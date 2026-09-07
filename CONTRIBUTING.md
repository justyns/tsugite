# Contributing

Quick guide for contributors. See [AGENTS.md](AGENTS.md) for architecture and dev commands.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/). Format:

```
<type>(<scope>): <subject>
```

Keep the subject on one line, lowercase, no trailing period. Body is optional; a change someone
using tsugite would notice carries a `Release-Note:` trailer (below).

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

### Release notes

git-cliff (`cliff.toml`) generates the release notes from the commits, and the line it prints for a
commit comes from a `Release-Note:` trailer:

```
fix(webui): scope agent artifact panes by session

Release-Note: Opening an artifact from one chat keeps every other chat's pane as it was.
```

One line, present tense, stating what the software does now. No "instead of", no because-clause. A
commit with nothing to tell a user (a test fix, an internal rename) carries no trailer and prints its
subject instead. `chore`, `docs`, `test` and `ci` commits never appear.

The trailer has to be the last paragraph of the message. When squashing, keep one `Release-Note:`
line at the end of the squashed message; a trailer buried mid-body is not parsed.

### Examples

```
feat(webui): show session topic inline in sidebar
fix(agent): use ast.parse to locate code block close fence
refactor(history): per-event JSONL log replaces Turn aggregate
chore: bump lxml to 6.1.0 (CVE-2026-41066)
docs: add plugin docs
```
