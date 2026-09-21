# pi-btw

**English** | [简体中文](README.zh-CN.md)

A [Pi](https://www.npmjs.com/package/@earendil-works/pi-coding-agent) extension that adds `/btw`: ask a question about the conversation you are already in, get one answer, and leave the main task running.

```text
/btw what is WBC in this diff
```

The main agent does not stop, is not steered, and never sees the question. The answer appears in a panel where the editor was and goes nowhere else — not into the transcript, not into the session file.

## Why not just ask

Three things already in Pi almost do this and each one costs something:

| | What it does |
| --- | --- |
| Enter while streaming | Steers the main task. Your question becomes part of the work. |
| Alt+Enter | Queues a follow-up. You get the answer after the task finishes. |
| A second session | Answers with no idea what "this diff" refers to. |

`/btw` is the fourth option: the same context, a separate one-shot agent, and nothing written back.

## What it does

- **Answers from the conversation's own context** — the system prompt, the tool schemas and the messages the main loop last sent, so "this diff" and "that function" mean what you think they mean
- **Reuses the main request's prefix**, byte for byte, so a side question reads from the provider's prompt cache instead of paying to write a second copy of the context
- **Cannot act.** No tools run. The schemas are in the request for the cache, and nothing feeds a tool result back. A model that writes a tool call as prose gets a line under it saying none of it happened
- **Remembers up to 20 exchanges per session**, in memory only. Earlier side questions can see each other; the main task still cannot see any of them
- **An empty `/btw` reopens the last one**, or reattaches to one still in flight

## Keys

| Key | |
| --- | --- |
| `Esc` | Cancel the request and close |
| `Enter` / `Space` / `Ctrl+]` | Close. An open request keeps running — an empty `/btw` picks it back up |
| `↑` / `↓` | Scroll the answer |
| `[` / `]`, or `Shift+←` / `Shift+→` | Step through recent side questions |
| `c` | Copy the answer as markdown |
| `x` | Clear the list above the panel. The answer on screen stays |

Escape is the only key that cancels. In Claude Code the panel can be put away and dismissed by separate paths, and only one of them aborts; here the panel owns the keyboard while it is up, so closing it has to be free.

## Install

```bash
pi package add @blueocean223/pi-btw
```

Or point Pi at a checkout:

```bash
pi --extension ./index.ts
```

## Notes

- **Interactive only.** In `print`, `json` and `rpc` modes `/btw` says so and sends nothing.
- **A side question sees the main loop's last outbound request**, not the response to it. A tool result recorded since then shows up in the next side question, after the main loop sends again.
- **While the panel is up it owns the keyboard**, so the main task's interrupt key is not reachable. Close the panel first.
- **The exchange list lives for as long as the process does.** Switching sessions and back keeps it; quitting or `/reload` drops it, and an empty `/btw` is back to the usage line.
- **`PI_BTW_DEBUG=1`** prints `cacheRead`/`cacheWrite` after each answer. A warm second question reads most of the context and writes almost nothing; a zero read next to a large write means the prefix diverged from the main request's.

## Not in this version

- **Promoting an answer into a real branch.** Claude Code's `f` splices the exchange into a parallel fork while the main session keeps going. Pi's `ctx.fork` tears the current runtime down and switches to the fork, which stops the main task — the thing `/btw` exists to avoid.
- **Tools for side questions.** That is a different feature; the permission model here is that schemas exist for the cache and calls are never executed.
- **`/btw` in the middle of a sentence.** Pi dispatches commands from the start of the input only.
- **Images.** The question is text. Images already in the shared prefix come along with it.

## Development

```bash
bun test ./*.test.ts   # pure functions: replay rewrite, extraction, cache-control placement
bun x tsc --noEmit
```

`node_modules/@earendil-works/*` is symlinked to the installed Pi, so the types check against the version you actually run.

## License

MIT
