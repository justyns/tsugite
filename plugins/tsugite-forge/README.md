# tsugite-forge

Read open pull requests from Forgejo, GitHub and Bitbucket Cloud in the tsugite daemon.

## Setup

- Forgejo: create an API token with `read:repository`, `read:organization` and `read:user`, store
  it with `tsu secrets set forgejo-token`, and put that name in `token_secret`.
- GitHub: install the [`gh` CLI](https://cli.github.com) for the daemon's user and run
  `gh auth login`.
- Bitbucket Cloud: create an Atlassian API token with the `read:repository:bitbucket`,
  `read:pullrequest:bitbucket` and `read:user:bitbucket` scopes, store it with
  `tsu secrets set bitbucket-token`, put that name in `token_secret`, and set `email` to
  your Atlassian account email. Repo patterns use the workspace as the owner. Only pull
  requests you opened are listed.

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
      - name: bb
        kind: bitbucket
        email: me@example.test
        token_secret: bitbucket-token
        repos: [myworkspace/*]
```
