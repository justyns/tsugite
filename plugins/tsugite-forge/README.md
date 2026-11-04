# tsugite-forge

Read open pull requests from Forgejo and GitHub in the tsugite daemon.

## Setup

- Forgejo: create an API token with `read:repository`, `read:organization` and `read:user`, store
  it with `tsu secrets set forgejo-token`, and put that name in `token_secret`.
- GitHub: install the [`gh` CLI](https://cli.github.com) for the daemon's user and run
  `gh auth login`.

```yaml
plugins:
  forge:
    poll_minutes: 5
    forges:
      - name: home
        kind: forgejo
        url: https://forge.example.test
        token_secret: forgejo-token
        repos: [octo-org/demo, octo-org/*]
      - name: gh
        kind: github
        repos: [octo-org/*]
```
