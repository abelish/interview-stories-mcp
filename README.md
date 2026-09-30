# interview-stories-mcp

An MCP server for storing interview stories in STAR format (situation, task,
action, result) and retrieving them by keyword or scenario during interview
prep.

## Storage

Stories live in `data/stories.json`, a flat JSON array, gitignored since it's
personal content. `data/stories.example.json` shows the schema — copy it to
`data/stories.json` to start, or just call `add_story` once the server is
running and it'll create the file.

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

- `list_stories()` — id, title, tags for every story
- `get_story(story_id)` — full STAR text for one story
- `search_stories(query)` — keyword search across title, tags, and STAR text
- `add_story(title, tags, situation, task, action, result)`
- `update_story(story_id, ...)` — update any subset of fields
- `delete_story(story_id)`
