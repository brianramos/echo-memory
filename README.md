# echo-memory

> **Your coding agent has a map. This is its memory.**

Exact, scoped, self-summarizing storage for AI coding agents — a hierarchical KV where
**every write echoes a summary into every ancestor prefix**, so aggregates exist at every
zoom level as an *invariant*, not a sync job. Competence that lives in this index transfers
to repositories the model has never seen; competence baked into weights does not. This repo
contains the data structure, a real-repo benchmark harness, and the experiment that shows
the difference.

---

## The result in one table

Train a model on **Flask**'s real symbol table, then evaluate on **Requests** (never seen),
plus a probe where the presented evidence is shuffled. Reproduced by this repo's demo on
every run (chance ≈ 0.031):

| evaluation                      | in-weights baseline | echo memory |
|---------------------------------|:-------------------:|:-----------:|
| FROM repo (fixed facts)         | 0.963               | **1.000**   |
| TO repo (never seen)            | 0.033               | **1.000**   |
| FROM repo, evidence **shuffled**| 0.032               | **1.000**   |

The in-weights model isn't *degraded* on the new repo — it's at chance, and the shuffle
probe shows why: it was never reading the context at all, only reciting. Echo memory
answers from the evidence in front of it, so repo identity, scale, and fact-shuffles are
irrelevant by construction.

## What it is

Code already has a natural hierarchical keyspace — `repo / module / class / method` — and
a parser emits it for free. Echo memory stores every object as an exact leaf at its path,
and the one unusual move is the write path:

```python
def put(path, vec, payload):
    exact[path] = payload
    for prefix in prefixes(path):      # module, module/class, ...
        sums[prefix]   += vec          # O(depth) — a handful of dict updates
        counts[prefix] += 1
```

That single write is then visible at **four resolutions**, each a dictionary lookup:

| read                     | answers                                    | typical use          |
|--------------------------|--------------------------------------------|----------------------|
| `get(path)`              | the exact symbol                           | precision            |
| `prefix_summary(path)`   | live aggregate over a subtree              | orientation          |
| `child_summaries(path)`  | one line per direct child                  | ranked navigation    |
| `resolve(scope, name)`   | nearest-enclosing binding (walk-up)        | shadowing / overrides|

The resolver is five lines and *is* the semantics of variable shadowing, config-override
precedence, and permission inheritance — including post-pop restore of an outer binding,
the exact operation where trained neural memories break at unseen depths:

```python
def resolve(scope, name):
    while True:
        if (hit := exact.get(scope + (name,))) is not None: return hit
        if not scope: return None
        scope = scope[:-1]
```

None of the ingredients are exotic (tries, rollups, lexical scoping). The point is the
combination, applied as an *agent's memory* — and then measured.

## Quickstart

```bash
pip install numpy pandas matplotlib seaborn      # plus git on PATH
python3 echo_memory_demo.py \
    --from-repo https://github.com/pallets/flask \
    --to-repo   https://github.com/psf/requests \
    --outdir results/
```

Local paths and bare clones work too; `--quick` gives a ~30 s smoke run. Real output from
the run that produced the figures in this repo:

```
  INCREMENTAL REPLAY — flask: last 25 commits, one apply each
──────────────────────────────────────────────────────────────
    base build @HEAD~25                              232 ms
    per-commit apply  (median / p90 / max)           2.1 ms   p90 61.2 / max 93.0
    full rebuild @HEAD (comparison)                  232 ms   speedup ×109 vs median apply
    [PASS] replayed state digest == fresh HEAD build — 6dab05dc… vs 6dab05dc…
    [PASS] invariant holds after adds/replaces/deletes — 0 bad prefixes
...
  DONE — 9/9 verdicts passed, 65 metrics recorded.
```

Outputs: a long-format `echo_benchmarks.csv`, four seaborn figures
(`fig_transfer / fig_incremental / fig_read_latency / fig_scope`), and the console report.

## What gets benchmarked

Everything runs against **real repositories** you point it at, and every suite ends in a
machine-checked `[PASS]/[FAIL]` verdict rather than a vibe:

- **Index build** — AST symbol extraction + echo writes (Flask: 415 symbols, ~36 µs/put,
  ~250 KiB), followed by the consistency check: every stored aggregate recomputed from
  leaves and compared. Aggregates that can drift would fail here.
- **Read latency** — leaf get, prefix summary, child summaries, scope resolve:
  0.4–1.4 µs medians, because there is nothing to compute at read time.
- **Incremental replay** — the last *N* real commits applied one at a time
  (adds/edits/deletes/renames), median ~2 ms per commit vs a full rebuild, and the
  strongest verdict in the suite: the replayed state's **digest must equal a fresh build
  at HEAD**. Correctness, not just speed.
- **Scope resolution** — synthetic bind/shadow/pop streams over the repo's own identifier
  names, depths 2–12, checked against an independent environment stack: exact at every
  depth, ~1 µs per resolve.
- **Determinism** — two independent builds must produce identical digests.
- **Transfer + shuffle probe** — the table above. The in-weights baseline is a numpy
  softmax memorizer, documented in-file as a distilled stand-in for what full neural
  models were shown to become under fixed facts.

## CLI

| flag           | default | meaning                                            |
|----------------|---------|----------------------------------------------------|
| `--from-repo`  | —       | path **or URL** of the known repo (required)       |
| `--to-repo`    | none    | unseen repo; omitting skips the transfer section   |
| `--from-src` / `--to-src` | auto | package prefix override, e.g. `src/flask/` |
| `--commits`    | 25      | commits for the incremental replay                 |
| `--iters`      | 3000    | timed calls per read op                            |
| `--seqs`       | 200     | transfer sequences per evaluation cell             |
| `--quick`      | off     | small everything, ~30 s                            |
| `--no-plots`   | off     | CSV + console only                                 |

## Alongside your code graph, not instead of it

Graph tools (roam-code, repowise, Sourcegraph-style indexes) are **spatial**: callers,
inheritance, blast radius. Vector search is **associative**: similar things when you don't
know the name. Echo memory is **mnemonic**: what is *known* at each identity and scope,
exactly, with receipts, cheap at every zoom level, updated the moment anything is written.
An agent needs all three; today it usually has only the first and last.

The integration recipe: **parse with tree-sitter, connect with the graph, remember with
echo** — route any signal that accumulates over identities (analysis results, ownership,
coverage, the agent's own decisions) through echo writes, and every summary the agent reads
between graph re-syncs is a microsecond lookup that cannot be stale. Two things fall out
that nothing else in the current stack offers: prefix-granular access shapes ("summaries of
`src/payments/**`, never the leaves" — a design property of the addressing, not yet
enforced in this demo) and a store the agent can *write to* with the same guarantees it
reads by.

## Limitations

Python-only AST extraction (tree-sitter frontends are the obvious next step); a heuristic
this-is-not-a-call-graph resolver; merge commits are diffed against their first parent;
the transfer task's values (`lineno % 32`) are real and verifiable but semantically
arbitrary; the neural evidence behind the headline comes from small models (~75k–110k
params, single-seed, pre-registered protocol) — the *probes*, however, are black-box and
run against any model you can call; timings are single-machine medians. 

## License & citation

MIT. If this is useful in research, please cite the repo
