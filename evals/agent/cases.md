# Agent eval cases

Each case sends one user message to Claude, with this project's MCP tools connected to a real
server seeded with the listed store. Generated from `cases.json` by
`uv run python -m evals.agent.cases`, so edit that file, not this one.

| id | scenario | store | expected |
|---|---|---|---|
| `capture-rambling` | capture / complete-story | no stories | saves exactly one complete STAR-L story |
| `capture-out-of-order` | capture / complete-story | no stories | saves exactly one complete STAR-L story |
| `capture-terse` | capture / complete-story | no stories | saves exactly one complete STAR-L story |
| `capture-no-learning` | capture / missing-learning | no stories | saves the story with the learning left empty, and asks for it |
| `capture-duplicate` | capture / duplicate | the 21 synthetic eval stories | does **not** save yet, asks first |
| `find-disagree-boss` | find / easy | the 21 synthetic eval stories | recommends `pushback-on-roadmap` (also OK: `disagree-and-commit`) |
| `find-influence` | find / easy | the 21 synthetic eval stories | recommends `logging-standard-influence` (also OK: `four-team-migration`) |
| `find-failure` | find / medium | the 21 synthetic eval stories | recommends `missed-launch` (also OK: `caused-outage`, `data-changed-direction`) |
| `find-coworker` | find / hard-for-search | the 21 synthetic eval stories | recommends `api-design-peer` (also OK: `hard-feedback-to-teammate`) |
| `find-broke-something` | find / hard-for-search | the 21 synthetic eval stories | recommends `caused-outage` (also OK: none) |
| `find-weakness` | find / hard-for-search | the 21 synthetic eval stories | recommends `receiving-criticism` (also OK: none) |
| `find-unfamiliar-tech` | find / hard-for-search | the 21 synthetic eval stories | recommends `learned-rust-fast` (also OK: none) |
| `gaps-fill` | gaps / no-content-given | the 21 stories, with learning removed from 3 | names the 3 incomplete stories, changes nothing without your input |
| `gaps-which` | gaps / no-content-given | the 21 stories, with learning removed from 3 | names the 3 incomplete stories, changes nothing without your input |
| `gaps-user-provides` | gaps / content-given | the 21 stories, with learning removed from 3 | updates only `learned-rust-fast` with the given learning |
| `none-budget` | no-fit | the 21 synthetic eval stories | says no story fits well |
| `none-fired` | no-fit | the 21 synthetic eval stories | says no story fits well |
| `none-relocation` | no-fit | the 21 synthetic eval stories | says no story fits well |
| `none-sales-quota` | no-fit | the 21 synthetic eval stories | says no story fits well |
| `none-hiring` | no-fit | the 21 synthetic eval stories | says no story fits well |

## `capture-rambling`

**Store:** no stories. **Expected:** saves exactly one complete STAR-L story.

````text
ok so I want to save a story. so, um, last year at my old job I was on the platform team, and our on-call was a nightmare, like people were getting paged five, six times a night, and half the alerts were noise. nobody owned fixing it because everyone was busy with features. so I kind of took it on myself. I pulled three months of paging data, grouped the alerts by source, and found like 70 percent came from four noisy checks. I rewrote those four, deleted a bunch of duplicate ones, and set up a weekly review where whoever was on call could flag bad alerts. after about two months pages went from like 40 a week to under 10, and people stopped dreading on-call. the thing I took away is that alert fatigue is a people problem as much as a technical one, and next time I'd set up that weekly review on day one instead of doing a big cleanup first.
````

## `capture-out-of-order`

**Store:** no stories. **Expected:** saves exactly one complete STAR-L story.

````text
Can you add this one to my stories? We ended up closing a $2M enterprise deal that had been stuck for a quarter, and honestly what I learned is that security reviews go faster when engineering joins the call directly instead of answering questionnaires by email. Context: I was a solutions engineer, and the customer's security team had sent us a 300-question questionnaire that had been bouncing back and forth for weeks. My job was to get us through their security review. I set up two live sessions with their security team and our infrastructure lead, answered the open questions on the spot, and wrote up a summary they could take to their CISO.
````

## `capture-terse`

**Store:** no stories. **Expected:** saves exactly one complete STAR-L story.

````text
Save this: Situation - our mobile app's crash rate spiked to 4% after a release. Task - I was the release owner. Action - I rolled back, bisected the crash to a third-party SDK update, pinned the version, and added a crash-rate gate to our release checklist. Result - crash rate back under 0.5% within a day, and the gate has caught two bad releases since. Learning - never let third-party SDKs float to new versions without a canary.
````

## `capture-no-learning`

**Store:** no stories. **Expected:** saves the story with the learning left empty, and asks for it.

````text
Please save this story. I was leading a small team building an internal reporting tool. Two weeks before the deadline, our data engineer quit. I had to make sure the tool still shipped. I took over the data pipeline work myself, cut two low-priority reports from scope, and asked a colleague from another team to review my SQL. We shipped on time with eight of the ten planned reports, and the remaining two followed a month later.
````

## `capture-duplicate`

**Store:** the 21 synthetic eval stories. **Expected:** does **not** save yet, asks first.

````text
Add a story: I led the rebuild of our mobile checkout flow with five engineers. We had a public launch date for the holiday campaign. I kept reporting green while integration testing slipped, and only raised the risk three weeks out. We cut scope and launched eight days late. I learned to report risk as soon as I see it.
````

## `find-disagree-boss`

**Store:** the 21 synthetic eval stories. **Expected:** recommends `pushback-on-roadmap` (also OK: `disagree-and-commit`).

Best story: *Pushing back on my manager's quarterly roadmap*

````text
I have an interview tomorrow and I'm expecting: "Tell me about a time you disagreed with your boss." Which of my stories should I use?
````

## `find-influence`

**Store:** the 21 synthetic eval stories. **Expected:** recommends `logging-standard-influence` (also OK: `four-team-migration`).

Best story: *Getting six teams to adopt a shared logging standard*

````text
Which story should I tell for "Give me an example of influencing people without formal authority"?
````

## `find-failure`

**Store:** the 21 synthetic eval stories. **Expected:** recommends `missed-launch` (also OK: `caused-outage`, `data-changed-direction`).

Best story: *Missing the launch date for the mobile checkout*

````text
They always ask "Tell me about a time you failed." What's my best story for that?
````

## `find-coworker`

**Store:** the 21 synthetic eval stories. **Expected:** recommends `api-design-peer` (also OK: `hard-feedback-to-teammate`).

Best story: *Settling an API design dispute with a fellow senior engineer*

````text
Help me pick a story for this question: "Tell me about a time you disagreed with a coworker."
````

## `find-broke-something`

**Store:** the 21 synthetic eval stories. **Expected:** recommends `caused-outage` (also OK: none).

Best story: *Taking down login with a config change*

````text
My interviewer likes the question "What's the biggest thing you've ever broken at work?" Do I have a story for it?
````

## `find-weakness`

**Store:** the 21 synthetic eval stories. **Expected:** recommends `receiving-criticism` (also OK: none).

Best story: *Hearing that I dominated design discussions*

````text
How should I answer "What's a weakness you've worked to improve?" using one of my stories?
````

## `find-unfamiliar-tech`

**Store:** the 21 synthetic eval stories. **Expected:** recommends `learned-rust-fast` (also OK: none).

Best story: *Picking up Rust to fix a performance-critical service*

````text
Which of my stories fits "Describe working with a technology you didn't know"?
````

## `gaps-fill`

**Store:** the 21 stories, with learning removed from 3. **Expected:** names the 3 incomplete stories, changes nothing without your input.

````text
Some of my stories are missing the learning part. Can you help me fill them in?
````

## `gaps-which`

**Store:** the 21 stories, with learning removed from 3. **Expected:** names the 3 incomplete stories, changes nothing without your input.

````text
Which of my stories are incomplete?
````

## `gaps-user-provides`

**Store:** the 21 stories, with learning removed from 3. **Expected:** updates only `learned-rust-fast` with the given learning.

````text
For my Rust story, the learning is: learning fast comes from a tight feedback loop and a friendly expert, and small reviewable changes let me learn safely in production code. Please add that.
````

## `none-budget`

**Store:** the 21 synthetic eval stories. **Expected:** says no story fits well.

````text
Which of my stories should I use for "Tell me about a time you managed a large budget"?
````

## `none-fired`

**Store:** the 21 synthetic eval stories. **Expected:** says no story fits well.

````text
I need a story for "Tell me about a time you had to fire someone." What do I have?
````

## `none-relocation`

**Store:** the 21 synthetic eval stories. **Expected:** says no story fits well.

````text
Pick one of my stories for "Describe a time you relocated to a new country for work."
````

## `none-sales-quota`

**Store:** the 21 synthetic eval stories. **Expected:** says no story fits well.

````text
Which story fits "Tell me about a time you exceeded your sales quota"?
````

## `none-hiring`

**Store:** the 21 synthetic eval stories. **Expected:** says no story fits well.

````text
What story should I tell for "Walk me through how you built a team from scratch through hiring"?
````
