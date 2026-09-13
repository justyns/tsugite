# Contributing

Quick guide for contributors. See [AGENTS.md](AGENTS.md) for architecture and dev commands.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/). Format:

```
<type>(<scope>): <subject>
```

Keep the subject on one line, lowercase, no trailing period, and write it as the line a user would
read in the release notes. No body, no trailers.

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

git-cliff (`cliff.toml`) generates the release notes from the commit subjects, so the subject is the
release line. Write it as what the software does now for the person using it, and it needs nothing
else. `chore`, `docs`, `test` and `ci` commits never appear.

`cliff.toml` also prints a `Release-Note:` trailer in place of the subject when one is present. That
is an escape hatch for the rare commit whose subject cannot say what a user sees, not something to
add by default; almost every commit should have none. When one is needed it has to be the last
paragraph of the message, and a squash keeps at most one.

### Examples

```
feat(webui): show session topic inline in sidebar
fix(agent): use ast.parse to locate code block close fence
refactor(history): per-event JSONL log replaces Turn aggregate
chore: bump lxml to 6.1.0 (CVE-2026-41066)
docs: add plugin docs
```
