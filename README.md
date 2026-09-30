# interview-stories-mcp

An MCP server for storing interview stories in STAR-L format (situation, task,
action, result, learning) and retrieving them by keyword or scenario during
interview prep.

## Storage

Stories live in `data/stories.json`, a flat JSON array, gitignored since it's
personal content. `data/stories.example.json` shows the schema — copy it to
`data/stories.json` to start, or just call `add_story` once the server is
running and it'll create the file.

Set the `STORIES_PATH` environment variable to store stories somewhere else.

It's safe to run more than one server against the same file (for example Claude
Code and Claude Desktop at once): every read and write holds a lock
(`stories.json.lock`), and saves replace the file atomically, so a crash never
leaves a half-written file. Hand edits are fine. Unknown keys are kept, and
missing STAR-L parts load as empty. If the file isn't valid JSON, tools report
where the problem is and never overwrite it.

## Setup

```
uv sync
```

## Running

```
uv run server.py
```

This starts the server on stdio, which is how Claude Code / Claude Desktop
talk to it — you won't see output, it just waits for a client to connect.

## Registering with Claude Code

```
claude mcp add interview-stories -- uv run --project "C:\Users\benji\OneDrive\Documents\Brami\Projects\interview-stories-mcp" server.py
```

## Registering with Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "interview-stories": {
      "command": "uv",
      "args": [
        "run",
        "--project",
        "C:\\Users\\benji\\OneDrive\\Documents\\Brami\\Projects\\interview-stories-mcp",
        "server.py"
      ]
    }
  }
}
```

## Tools

- `list_stories()` — id, title, tags, and whether the learning is missing, for every story
- `get_story(story_id)` — full STAR-L text for one story
- `search_stories(query)` — keyword search across title, tags, and STAR-L text
- `add_story(title, tags, situation, task, action, result, learning)`
- `update_story(story_id, ...)` — update any subset of fields
- `delete_story(story_id)`

The title and every STAR-L part are required and can't be blank. Surrounding
whitespace is trimmed. Tags are lowercased and hyphenated, so
"Conflict Resolution" and "conflict_resolution" are both stored as
`conflict-resolution`, and every story needs at least one.

## Development

```
uv run pytest        # unit tests plus end-to-end tests that drive server.py over stdio
uv run ruff check .
uv run ruff format .
uv run pyright
uv run python -m evals.retrieval --verbose   # search quality report
```

Tests always point `STORIES_PATH` at a temp file, so they never touch your real stories.
