# tsugite-onlyoffice

Edit `.docx` files from a configured directory in an embedded [ONLYOFFICE Docs](https://api.onlyoffice.com/docs/docs-api/get-started/basic-concepts/)
editor, and let an agent work on the same document.

The daemon is the document storage service: it serves a signed editor config, serves the file the
document server downloads, and handles the callback that writes saves back. The editor opens as a
web UI tab in the current tsugite theme. Agents get `@onlyoffice` tools that read the document and
its comment threads, edit it, and comment as themselves.

An agent edit to a document somebody has open force-saves their session, edits the file, and rotates
the document key, and the open editor swaps to the new version in place. A run of edits announces
once it settles, so a bulk operation costs the reader one swap rather than one per edit. Only the
parts that change are re-serialized, so existing comments, tracked changes and styles survive.

## Deployment: the document server has to reach you

The document server fetches `document.url` and POSTs `callbackUrl` **itself**, so both must be
reachable from it rather than from your browser. That is why `public_base_url` is required and has no
fallback to the bind address.

`services.CoAuthoring.request-filtering-agent.allowPrivateIPAddress` defaults to `false`, so a stock
document server refuses any RFC1918 address, including its own host, failing with a bare
`{"error": -4}` and no connection attempt. Front the daemon with a vhost it can resolve, or enable
that setting. `GET /api/plugins/onlyoffice/health` reports which of these you are hitting, and the
editor page shows the same message when a document fails to open.

`server_url` should be https, and the document server behind it is fully trusted: its `api.js` runs
same-origin in the editor surface and can read the daemon token the page holds, so a compromised or
intercepted document server gets daemon access.

## Config (`daemon.yaml` -> `plugins.onlyoffice`)

| key | default | meaning |
|-----|---------|---------|
| `enabled` | `false` | load the plugin |
| `server_url` | required | base URL of the ONLYOFFICE Docs server |
| `jwt_secret_name` | required | name of the shared JWT secret in the secrets backend |
| `public_base_url` | required | base URL the document server uses to reach this daemon |
| `documents_dir` | required | directory holding the editable documents |
| `agent_name` | `Tsugite` | the author on comments the tools write |

```yaml
plugins:
  onlyoffice:
    enabled: true
    server_url: https://onlyoffice.example.net
    jwt_secret_name: onlyoffice-jwt-secret
    public_base_url: https://tsugite.example.net
    documents_dir: /srv/tsugite/documents
```

`plugins` is boot-only, so a change here needs a daemon restart. A missing required key, or one this
table does not list, fails the plugin's load and the daemon runs without it.

Every path from HTTP or a tool must resolve inside `documents_dir` and name a `.docx`, on the write
path as well as the read one. Every inbound request from the document server is JWT-verified, and
each token names both the document and the route it was minted for.

## Tools (`@onlyoffice`)

`doc_read`, `doc_insert`, `doc_replace`, `doc_comment`, `doc_reply`, `doc_resolve`.

`doc_read` returns the document as numbered paragraphs plus every comment thread. Those paragraph
numbers are what the editing and commenting tools take as anchors.

## Limitations

- docx only.
- Single writer: two people plus an agent editing at once is not a supported shape.
- Anything typed into the editor after a turn force-saves it is lost, for the length of the turn or
  of a batch.
- Turn state is in memory. After a daemon restart mid-turn, reload the tab.
- Retired document keys are in memory too, so the first tab opened after a restart can land on a key
  the document server already closed. Editing the document clears it.
- A captured callback can be replayed onto the document it was minted for, re-applying a state that
  document was previously saved in.
- The agent edits server-side rather than holding an editor seat, so its changes arrive as a new file
  version, not another cursor.
- Rewording a comment retires its id, because the id is a digest of its author, date and text. Every
  number the format carries is renumbered on save, so none of them can be held across one. A retired
  id is an error, never a silent hit on the comment beside it.
- Comment dates are not comparable between authors: the editor stamps local time with a `Z`, the
  tools stamp UTC. `doc_read` orders comments by where the document anchors them instead.
- The side panels and status bar cannot be hidden; `customization.layout` needs the Developer Edition
  white-label licence.
