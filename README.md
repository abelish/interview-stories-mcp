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

- `list_stories()` — id, title, tags, a one-line summary, and whether the learning is missing, for every story
- `get_story(story_id)` — full STAR-L text for one story
- `search_stories(query, limit=5)` — the stories that best fit an interview question, best first
- `add_story(title, tags, situation, task, action, result, learning, allow_duplicate=False)`
- `update_story(story_id, ...)` — update any subset of fields
- `delete_story(story_id)`

The title and every STAR-L part except the learning are required and can't be
blank. A story can be saved before its learning is known: it's flagged with
`needs_learning` in `list_stories` so it can be filled in later, rather than
being lost or having a learning made up. Surrounding whitespace is trimmed.
Tags are lowercased and hyphenated, so "Conflict Resolution" and
"conflict_resolution" are both stored as `conflict-resolution`, and every
story needs at least one.

`add_story` won't save a story that looks like one already saved, which is
what happens when the user tells a story again in different words. It saves
nothing and names the existing story, so Claude can ask whether to update it,
keep both (`allow_duplicate=true`), or skip it. A story counts as a retelling
when it shares at least 45% of its distinct words with a saved one. Retellings
measured 60-80% and distinct stories at most 30%, even ones on the same theme.

## Search

`search_stories` blends two scores:

- **Keyword** (BM25): matches words, with tags counting most, then the title,
  then the STAR-L text. Words are stemmed, and contractions and interview
  boilerplate like "tell me about a time" are ignored.
- **Semantic**: matches meaning using a small local embedding model
  ([bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), 67 MB),
  so "a coworker" finds a story about "a fellow senior engineer".

Each counts equally. The keyword score is the share of the question a story
matches, weighted by how rare each word is, so matching a word most stories
share counts for little. The semantic score is rescaled from 0 to 1 within
each search.

The model downloads once, on the server's first start, to
`~/.cache/interview-stories-mcp` (set `STORIES_MODEL_CACHE` to change it) and
runs offline after that. If it can't load, search falls back to keyword-only
and retries a minute later.

The approach was chosen with the retrieval eval (see [Evals](#evals)).

## Evals

Two evals live in `evals/`. The retrieval eval checks search on its own, is
free, and runs with the tests. The agent eval checks Claude using the tools
end to end, costs money, and is run by hand.

Both use the same 21 synthetic stories (`evals/corpus.json`), written to
overlap the way a real story bank does: three stories about disagreeing, two
about failing, and so on.

### Retrieval eval

52 interview questions, each labeled with the story that best answers it and
any acceptable alternatives (`evals/queries.json`). Each is searched twice:
as asked, and rewritten as keywords. No LLM is involved, so results are the
same on every run.

It reports, for each form:

- **top1**: the best story is ranked first
- **top3**: the best story is in the top 3
- **mrr**: mean reciprocal rank of the best story (1 for first, 1/2 for second, ...)
- **rel@3**: the best story or an acceptable one is in the top 3
- **empty**: search returned nothing

```
uv run python -m evals.retrieval --verbose            # hybrid search, listing every miss
uv run python -m evals.retrieval --mode keyword       # the keyword-only fallback
uv run python -m evals.retrieval --update-thresholds  # ratchet the minimums up
```

It's also a quality gate. `evals/retrieval_thresholds.json` holds a minimum
for every metric, for both hybrid and keyword-only search, and `pytest` fails
if search falls below any of them. `--update-thresholds` raises them to the
current results, and refuses if any metric got worse. Lowering one is
deliberate: edit the file by hand and say why in the commit.

To check search against your own stories, write questions labeled with your
story ids in `data/eval_queries.json` (same format as `evals/queries.json`,
with `keywords` optional) and run `uv run python -m evals.retrieval --personal`.
It reports but never gates, and runs against a copy of your stories.

### Agent eval

20 cases, each one user message to Claude with this server connected and
seeded with a set of stories. They cover four scenarios:

- **capture**: saving a story the user tells, including one with no learning
  and one already in the bank
- **find**: picking the right story for an interview question
- **gaps**: finding stories with no learning, without making one up
- **no-fit**: saying honestly that no story fits

`evals/agent/cases.md` lists every case and what passing means. To change the
cases, edit `evals/agent/cases.json` and regenerate the list with
`uv run python -m evals.agent.cases`.

Each trial is graded on two scores:

- **pass**: every check passed
- **safe writes**: Claude didn't add, change, or delete stories it shouldn't
  have. This separates "changed something it shouldn't" from "answered badly".

Checks on what was written are done in code, by comparing the stories file
before and after. Checks code can't make, like whether a saved story invents
details or which story a reply recommends, go to a judge model, Claude Sonnet
5.5. It sees the conversation as data and answers a fixed set of yes/no or
pick-one questions.

Defaults: Claude Opus 5.5 at medium effort, Sonnet 5.5 as judge, 1 rep. It
needs `ANTHROPIC_API_KEY` set. A trial costs about $0.05, so all 20 cases at
1 rep cost about $1 and take about a minute.

```
uv run python -m evals.agent.runner --approve-harness                # after reviewing harness changes
uv run python -m evals.agent.runner                                  # every case, as the baseline
uv run python -m evals.agent.runner --cases find-weakness,gaps-fill  # some cases
uv run python -m evals.agent.runner --variant v1 --reps 3            # a later version, 3 reps each
uv run python -m evals.agent.runner --summary                        # pass rates and cost so far
```

**Harness approval.** The runner refuses to start if the cases, the grading,
or the runner itself changed since they were last approved, since a change
there changes what the numbers mean. Review the change, then run
`--approve-harness` yourself. The approval is recorded in
`.claude/hillclimb/agent-tools/_state.json`, which is committed. Changes to
the server and to search don't need approval: they're what's being measured.

**Results** go to `.claude/hillclimb/agent-tools/<variant>/`. `results.jsonl`
has one graded row per trial, with every check and its reason, and is
committed so results can be compared later. Full conversation traces go to
`traces/` (gitignored). Trials that fail for reasons that aren't Claude's,
like an API error or a timeout, go to `errors.jsonl` and are never scored. A
run picks up where it left off, so rerunning skips finished trials.

Before trusting a change to grading, check it against conversations whose
grade you already know: a correct one must pass, and an empty or wrong one
must fail.

## Development

```
uv run pytest        # unit tests plus end-to-end tests that drive server.py over stdio
uv run ruff check .
uv run ruff format .
uv run pyright
uv run python -m evals.retrieval --verbose   # search quality report
```

Tests always point `STORIES_PATH` at a temp file, so they never touch your real stories.
