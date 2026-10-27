# Solo chat window

`#solo` renders one chat and nothing else: the transcript, the composer and a minimal header. There is no nav rail, space bar, context rail or tab strip. An `open_artifact` call docks a second pane beside the conversation, and no rails appear. At 700px wide or less, a chat/artifact toggle shows one pane at a time.

## URLs

- `<daemon-url>/#solo?sessionId=<id>` pins one conversation. If it is compacted into a successor, the window says so and links to the successor.
- `<daemon-url>/#solo?alias=<alias>` shows whichever session currently holds the alias, and follows it across compaction. If no session holds the alias, the window shows an empty state and opens the session once one claims it.
- With both parameters, `sessionId` is used and `alias` is ignored.

A session's menu in the full UI has **Open in solo window** and **Copy solo link**. Both use the alias form when the session holds an alias, and the `sessionId` form otherwise.

## Launching

```sh
chromium --app="<daemon-url>/#solo?alias=assistant"
```

The installed PWA opens a standalone window; open `#solo?...` in it. The window stays usable at about 400x600: the transcript scrolls and the composer stays visible.

## Pinning on Hyprland

Always-on-top is set by the window manager. Float and pin the window with a rule matched on its class (see `hyprctl clients`). The Hyprland wiki has the syntax: <https://wiki.hypr.land/Configuring/Window-Rules/>.
