#!/usr/bin/env python3
"""echo_memory_demo.py — standalone echo-memory demonstration & benchmark suite.

Consolidates the echo-memory line of work into one dependency-light file:

  * EchoMemory  — hierarchical exact KV where every write echoes a summary into
                  every ancestor prefix. O(depth) writes; O(1) aggregate reads;
                  native lexical (nearest-enclosing) scope resolution; clean
                  replace/delete; consistency as an invariant.
  * Real-repo harness — clone/point at any two git repos ("from" = known,
                  "to" = unseen), extract symbol tables at HEAD via AST.
  * Benchmarks  — index build, read latency (leaf / prefix summary / subtree /
                  resolve), incremental single-commit replay with a correctness
                  digest against a fresh build, scope-resolution vs depth,
                  consistency + determinism verdicts.
  * Transfer    — the three-memories experiment: an in-weights memorizer
                  baseline (numpy softmax regression; a distilled stand-in for
                  what trained neural models became under fixed facts — see the
                  accompanying report's shuffle-probe result) vs echo memory,
                  evaluated on the FROM repo, the unseen TO repo, and the
                  shuffled-evidence probe.
  * Reporting   — polished stdout, results CSV (pandas), seaborn figures.

Usage:
  python3 echo_memory_demo.py \
      --from-repo https://github.com/pallets/flask \
      --to-repo   https://github.com/psf/requests \
      --outdir results/

Local paths (including bare clones) work too. `--quick` shrinks iteration
counts for a fast smoke run. Requires: git, numpy, pandas, matplotlib, seaborn.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ============================================================================
# 1. EchoMemory — the standalone core
# ============================================================================

def sketch(text: str, dim: int = 8) -> List[float]:
    """Tiny signed hashed bag-of-tokens vector; the payload we echo upward."""
    v = [0.0] * dim
    for tok in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", text):
        h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
        v[h % dim] += 1.0 if (h >> 8) % 2 else -1.0
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


class EchoMemory:
    """Exact hierarchical KV with write-time ancestor summaries ("echoes").

    put(path, vec, payload):
        exact[path] = payload
        for every proper prefix of path: sums[prefix] += vec; counts[prefix] += 1

    Reads:
        get(path)              exact leaf
        prefix_summary(path)   count + mean vector over all descendant leaves
        child_summaries(path)  one-line summary per direct child
        subtree(path)          descendant leaf paths
        resolve(scope, name)   nearest-enclosing lexical lookup (walk-up)
    """

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.exact: Dict[Tuple[str, ...], Any] = {}
        self.leaf_vec: Dict[Tuple[str, ...], List[float]] = {}
        self.sums: Dict[Tuple[str, ...], List[float]] = defaultdict(lambda: [0.0] * dim)
        self.counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.children: Dict[Tuple[str, ...], set] = defaultdict(set)
        self.n_puts = self.n_deletes = 0

    # -- write path ----------------------------------------------------------
    def _apply(self, path: Tuple[str, ...], vec: Sequence[float], sign: int) -> None:
        for i in range(len(path)):
            pre = path[:i]
            s = self.sums[pre]
            for j in range(self.dim):
                s[j] += sign * vec[j]
            self.counts[pre] += sign
            if sign > 0:
                self.children[pre].add(path[i])

    def put(self, path: Sequence[str], vec: Sequence[float], payload: Any) -> None:
        path = tuple(path)
        if path in self.exact:                       # replace: retract old echo
            self._apply(path, self.leaf_vec[path], -1)
        self.exact[path] = payload
        self.leaf_vec[path] = list(vec)
        self._apply(path, vec, +1)
        self.n_puts += 1

    def delete(self, path: Sequence[str]) -> bool:
        path = tuple(path)
        if path not in self.exact:
            return False
        self._apply(path, self.leaf_vec.pop(path), -1)
        del self.exact[path]
        self.n_deletes += 1
        return True

    # -- read path -----------------------------------------------------------
    def get(self, path: Sequence[str]) -> Any:
        return self.exact.get(tuple(path))

    def prefix_summary(self, path: Sequence[str]) -> Dict[str, Any]:
        path = tuple(path)
        n = self.counts.get(path, 0)
        s = self.sums.get(path)
        mean = [x / n for x in s] if (s and n) else [0.0] * self.dim
        return {"path": "/".join(path) or "<root>", "leaves": n, "mean_vec": mean}

    def child_summaries(self, path: Sequence[str]) -> List[Dict[str, Any]]:
        path = tuple(path)
        out = []
        for c in sorted(self.children.get(path, ())):
            p = path + (c,)
            n = self.counts.get(p, 0)
            here = p in self.exact
            if n == 0 and not here:
                continue                              # fully deleted branch
            out.append({"name": c, "descendant_leaves": n, "is_leaf": here})
        return out

    def subtree(self, path: Sequence[str]) -> List[Tuple[str, ...]]:
        path = tuple(path)
        L = len(path)
        return [k for k in self.exact if k[:L] == path]

    def resolve(self, scope: Sequence[str], name: str) -> Any:
        """Nearest-enclosing lexical lookup: innermost binding of `name` wins."""
        cur = tuple(scope)
        while True:
            hit = self.exact.get(cur + (name,))
            if hit is not None:
                return hit
            if not cur:
                return None
            cur = cur[:-1]

    # -- verification --------------------------------------------------------
    def consistency_check(self) -> Tuple[bool, int]:
        """Recompute every prefix aggregate from leaves; compare to stored echoes."""
        rs: Dict[Tuple[str, ...], List[float]] = defaultdict(lambda: [0.0] * self.dim)
        rc: Dict[Tuple[str, ...], int] = defaultdict(int)
        for path, vec in self.leaf_vec.items():
            for i in range(len(path)):
                pre = path[:i]
                rc[pre] += 1
                s = rs[pre]
                for j in range(self.dim):
                    s[j] += vec[j]
        bad = 0
        keys = set(rc) | {k for k, v in self.counts.items() if v}
        for k in keys:
            if rc.get(k, 0) != self.counts.get(k, 0):
                bad += 1
                continue
            a, b = rs.get(k, [0.0] * self.dim), self.sums.get(k, [0.0] * self.dim)
            if any(abs(x - y) > 1e-6 for x, y in zip(a, b)):
                bad += 1
        return bad == 0, bad

    def digest(self) -> str:
        h = hashlib.sha256()
        for path in sorted(self.exact):
            pay = self.exact[path]
            h.update(("/".join(path) + "|" + json.dumps(pay, sort_keys=True, default=str)).encode())
        return h.hexdigest()[:16]


# ============================================================================
# 2. Git plumbing + symbol extraction
# ============================================================================

def sh(args: List[str], cwd: Optional[str] = None) -> str:
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(args)}\n{r.stderr.strip()[:400]}")
    return r.stdout


def git(repo: str, *args: str) -> str:
    return sh(["git", "-C", repo, *args])


def clone_if_url(spec: str, cache_dir: Path) -> str:
    if re.match(r"^(https?://|git@|ssh://)", spec):
        name = re.sub(r"\.git$", "", spec.rstrip("/").split("/")[-1]) + ".git"
        dest = cache_dir / name
        if not dest.exists():
            print(f"    cloning {spec} -> {dest} ...", flush=True)
            sh(["git", "clone", "--bare", "--quiet", spec, str(dest)])
        return str(dest)
    p = Path(spec).expanduser()
    if not p.exists():
        raise SystemExit(f"repo not found: {spec}")
    return str(p)


def detect_src_prefix(repo: str) -> str:
    """Pick the top-level python package with the most .py files."""
    paths = [p for p in git(repo, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
             if p.endswith(".py")]
    buckets: Dict[str, int] = defaultdict(int)
    inits = set(p for p in paths if p.endswith("__init__.py"))
    for p in paths:
        parts = p.split("/")
        cand = "/".join(parts[:2]) + "/" if parts[0] == "src" and len(parts) > 2 else parts[0] + "/"
        buckets[cand] += 1
    ranked = sorted(buckets.items(), key=lambda kv: -kv[1])
    for cand, _ in ranked:
        if cand + "__init__.py" in inits:
            return cand
    return ranked[0][0] if ranked else ""


def blob(repo: str, rev: str, path: str) -> str:
    try:
        return git(repo, "show", f"{rev}:{path}")
    except RuntimeError:
        return ""


def file_defs(src: str) -> Dict[str, Tuple[int, str, str]]:
    """qualname -> (lineno, kind, body-hash) for classes, methods, functions."""
    try:
        tree = ast.parse(src)
    except Exception:
        return {}
    out: Dict[str, Tuple[int, str, str]] = {}

    class V(ast.NodeVisitor):
        def __init__(self):
            self.stack: List[str] = []

        def _body_hash(self, node) -> str:
            try:
                seg = ast.get_source_segment(src, node) or ""
            except Exception:
                seg = ""
            return hashlib.blake2b(seg.encode(), digest_size=6).hexdigest()

        def visit_ClassDef(self, n):
            q = ".".join(self.stack + [n.name])
            out[q] = (n.lineno, "class", self._body_hash(n))
            self.stack.append(n.name); self.generic_visit(n); self.stack.pop()

        def visit_FunctionDef(self, n): self._fn(n)
        def visit_AsyncFunctionDef(self, n): self._fn(n)

        def _fn(self, n):
            q = ".".join(self.stack + [n.name])
            out[q] = (n.lineno, "method" if self.stack else "function", self._body_hash(n))
            self.stack.append(n.name); self.generic_visit(n); self.stack.pop()

    V().visit(tree)
    return out


def module_path(rel: str, prefix: str) -> Tuple[str, ...]:
    return tuple(rel[len(prefix):-3].split("/"))


def repo_symbols(repo: str, prefix: str, rev: str = "HEAD"
                 ) -> List[Tuple[Tuple[str, ...], Dict[str, Any]]]:
    files = [p for p in git(repo, "ls-tree", "-r", "--name-only", rev).splitlines()
             if p.startswith(prefix) and p.endswith(".py")]
    out = []
    for f in files:
        mp = module_path(f, prefix)
        for q, (ln, kind, bh) in file_defs(blob(repo, rev, f)).items():
            path = mp + tuple(q.split("."))
            out.append((path, {"kind": kind, "lineno": ln, "file": f, "hash": bh}))
    return out


def build_memory(symbols) -> EchoMemory:
    m = EchoMemory()
    for path, pay in symbols:
        m.put(path, sketch(pay["kind"] + " " + path[-1]), pay)
    return m


# ============================================================================
# 3. Reporting helpers
# ============================================================================

RESULTS: List[Dict[str, Any]] = []


def record(section: str, benchmark: str, metric: str, value: float, unit: str,
           repo: str = "-", detail: str = "") -> None:
    RESULTS.append(dict(section=section, benchmark=benchmark, repo=repo,
                        metric=metric, value=round(float(value), 6),
                        unit=unit, detail=detail))


W = 78
def hr(ch: str = "─") -> None: print(ch * W)
def banner(t: str) -> None: hr("═"); print(f"  {t}"); hr("═")
def section(t: str) -> None: print(); hr(); print(f"  {t}"); hr()
def row(label: str, value: str, note: str = "") -> None:
    print(f"    {label:<44} {value:>16}  {note}")
def verdict(label: str, ok: bool, note: str = "") -> None:
    print(f"    [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {note}" if note else ""))
    record("verdicts", label, "pass", 1.0 if ok else 0.0, "bool", detail=note)


def timeit_ns(fn, iters: int, warmup: int = 50) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter_ns(); fn(); ts.append(time.perf_counter_ns() - t0)
    return statistics.median(ts)


# ============================================================================
# 4. Benchmark suites
# ============================================================================

def bench_index(repo: str, prefix: str, tag: str):
    section(f"INDEX BUILD — {tag} ({prefix})")
    t0 = time.perf_counter()
    symbols = repo_symbols(repo, prefix)
    t_extract = time.perf_counter() - t0
    tracemalloc.start()
    t0 = time.perf_counter()
    m = build_memory(symbols)
    t_build = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    ok, bad = m.consistency_check()
    row("symbols extracted (classes+methods+functions)", f"{len(symbols):,}")
    row("AST extraction", f"{t_extract*1000:,.1f} ms")
    row("echo writes (put, incl. ancestor echoes)", f"{t_build*1000:,.1f} ms",
        f"{t_build/max(1,len(symbols))*1e6:,.1f} µs/put")
    row("index footprint (tracemalloc peak)", f"{peak/1024:,.0f} KiB")
    verdict("aggregates == recomputed-from-leaves (invariant)", ok, f"{bad} bad prefixes")
    record("index", "extract", "wall", t_extract * 1000, "ms", tag, f"{len(symbols)} symbols")
    record("index", "build", "wall", t_build * 1000, "ms", tag)
    record("index", "build", "per_put", t_build / max(1, len(symbols)) * 1e6, "us", tag)
    record("index", "footprint", "peak", peak / 1024, "KiB", tag)
    return m, symbols


def bench_reads(m: EchoMemory, symbols, tag: str, iters: int):
    section(f"READ LATENCY — {tag}  (median of {iters:,} timed calls)")
    rng = random.Random(0)
    leafs = [p for p, _ in symbols]
    files = sorted({p[:-1] for p in leafs if len(p) > 1})
    classes = [p for p, pay in symbols if pay["kind"] == "class"]
    ops = {
        "leaf get(path)": lambda: m.get(rng.choice(leafs)),
        "prefix_summary(module)": lambda: m.prefix_summary(rng.choice(files)),
        "child_summaries(class)": (lambda: m.child_summaries(rng.choice(classes))) if classes else None,
        "resolve(scope,name) depth-3": lambda: m.resolve(rng.choice(files) + ("X",), rng.choice(leafs)[-1]),
    }
    for name, fn in ops.items():
        if fn is None:
            continue
        ns = timeit_ns(fn, iters)
        row(name, f"{ns/1000:,.2f} µs")
        record("read_latency", name, "median", ns / 1000, "us", tag)


def bench_incremental(repo: str, prefix: str, tag: str, k_commits: int):
    section(f"INCREMENTAL REPLAY — {tag}: last {k_commits} commits, one apply each")
    hs = [h for h in git(repo, "log", "--format=%H", f"-n{k_commits+1}",
                         "--first-parent", "HEAD").splitlines() if h]
    if len(hs) < 2:
        print("    (history too short — skipped)"); return
    base = hs[-1]
    t0 = time.perf_counter()
    m = build_memory(repo_symbols(repo, prefix, rev=base))
    t_base = time.perf_counter() - t0
    applies: List[float] = []
    touched_total = 0
    for c in reversed(hs[:-1]):                       # oldest -> newest
        t0 = time.perf_counter()
        try:
            status = git(repo, "diff", "--name-status", "-M", f"{c}~1", c, "--", prefix)
        except RuntimeError:
            status = ""
        for line in status.splitlines():
            parts = line.split("\t")
            code = parts[0][0]
            if code == "R" and len(parts) == 3:
                oldp, newp = parts[1], parts[2]
                if oldp.endswith(".py"):
                    for leaf in m.subtree(module_path(oldp, prefix)):
                        m.delete(leaf)
                if newp.endswith(".py"):
                    mp = module_path(newp, prefix)
                    for q, (ln, kind, bh) in file_defs(blob(repo, c, newp)).items():
                        m.put(mp + tuple(q.split(".")), sketch(kind + " " + q.split(".")[-1]),
                              {"kind": kind, "lineno": ln, "file": newp, "hash": bh})
                touched_total += 1
                continue
            path = parts[-1]
            if not path.endswith(".py"):
                continue
            touched_total += 1
            mp = module_path(path, prefix)
            if code == "D":
                for leaf in m.subtree(mp):
                    m.delete(leaf)
                continue
            new = file_defs(blob(repo, c, path))
            old_leaves = {p: m.exact[p] for p in m.subtree(mp)}
            newpaths = set()
            for q, (ln, kind, bh) in new.items():
                p = mp + tuple(q.split("."))
                newpaths.add(p)
                prev = old_leaves.get(p)
                if prev is None or prev.get("hash") != bh or prev.get("lineno") != ln:
                    m.put(p, sketch(kind + " " + q.split(".")[-1]),
                          {"kind": kind, "lineno": ln, "file": path, "hash": bh})
            for p in old_leaves:
                if p not in newpaths:
                    m.delete(p)
        applies.append((time.perf_counter() - t0) * 1000)

    fresh = build_memory(repo_symbols(repo, prefix, rev=hs[0]))
    same = m.digest() == fresh.digest()
    ok, bad = m.consistency_check()
    med = statistics.median(applies)
    row(f"base build @HEAD~{k_commits}", f"{t_base*1000:,.0f} ms")
    row("per-commit apply  (median / p90 / max)",
        f"{med:,.1f} ms",
        f"p90 {np.percentile(applies,90):,.1f} / max {max(applies):,.1f}  ({touched_total} file-touches)")
    row("full rebuild @HEAD (comparison)", f"{t_base*1000:,.0f} ms",
        f"speedup ×{(t_base*1000)/max(1e-9,med):,.0f} vs median apply")
    verdict("replayed state digest == fresh HEAD build", same,
            f"{m.digest()} vs {fresh.digest()}")
    verdict("invariant holds after adds/replaces/deletes", ok, f"{bad} bad prefixes")
    for a in applies:
        record("incremental", "apply_commit", "wall", a, "ms", tag)
    record("incremental", "full_rebuild", "wall", t_base * 1000, "ms", tag)


def bench_scope(m_names: List[str], tag: str, trials: int):
    section(f"SCOPE RESOLUTION — synthetic streams over {tag} identifiers")
    rng = random.Random(7)
    rows_out = []
    for depth in (2, 4, 8, 12):
        lat, correct, n = [], 0, 0
        for _ in range(trials):
            mem = EchoMemory(dim=2)
            envs: List[Dict[str, int]] = [{}]
            scopes: List[str] = ["g"]
            names = rng.sample(m_names, min(6, len(m_names)))
            for d in range(depth):
                if d:
                    scopes.append(f"s{d}"); envs.append({})
                if d == 0 or rng.random() < 0.6:
                    nm, v = rng.choice(names), rng.randrange(1000)
                    envs[-1][nm] = v
                    mem.put(tuple(scopes) + (nm,), [1, 0], v)
            for _ in range(4):                        # queries incl. post-pop restores
                if len(scopes) > 1 and rng.random() < 0.5:
                    scopes.pop(); envs.pop()
                nm = rng.choice(names)
                gt = next((e[nm] for e in reversed(envs) if nm in e), None)
                t0 = time.perf_counter_ns()
                got = mem.resolve(tuple(scopes), nm)
                lat.append(time.perf_counter_ns() - t0)
                correct += int(got == gt); n += 1
        acc = correct / n
        med = statistics.median(lat) / 1000
        rows_out.append((depth, acc, med))
        record("scope", f"depth_{depth}", "accuracy", acc, "frac", tag)
        record("scope", f"depth_{depth}", "resolve_median", med, "us", tag)
    for depth, acc, med in rows_out:
        row(f"depth {depth:>2}: resolve == env-stack ground truth",
            f"{acc*100:5.1f} %", f"median {med:,.2f} µs")
    verdict("shadow / rebind / post-pop restore all exact",
            all(a == 1.0 for _, a, _ in rows_out))


# ============================================================================
# 5. Transfer experiment — three memories, two repos, one probe
# ============================================================================

def hash_feat(path: Tuple[str, ...], F: int = 512) -> np.ndarray:
    x = np.zeros(F, np.float32)
    for i, c in enumerate(path):
        h = int.from_bytes(hashlib.blake2b(f"{i}:{c}".encode(), digest_size=8).digest(), "little")
        x[h % F] += 1.0
    return x


def train_memorizer(pool, epochs: int, lr: float = 0.5, F: int = 512):
    """Softmax regression path->value: the in-weights baseline. A distilled
    stand-in for what full neural models became under fixed facts (their
    shuffle-probe scores matched this baseline's mechanism; see report)."""
    W = np.zeros((F, 32), np.float32)
    rng = random.Random(0)
    items = list(pool)
    for _ in range(epochs):
        rng.shuffle(items)
        for path, val in items:
            x = hash_feat(path, F)
            z = x @ W
            p = np.exp(z - z.max()); p /= p.sum()
            p[val] -= 1.0
            W -= lr * np.outer(x, p)
    return W


def eval_models(W, pool_from, pool_to, n_seq: int, N: int = 16, seed: int = 1):
    rng = random.Random(seed)
    def run(pool, shuffle_vals: bool):
        acc_w = acc_e = n = 0
        items = list(pool)
        for _ in range(n_seq):
            chosen = rng.sample(items, min(N, len(items)))
            mem = EchoMemory(dim=2)                    # in-structure: reads the evidence
            presented = {}
            for path, true_v in chosen:
                v = rng.randrange(32) if shuffle_vals else true_v
                presented[path] = v
                mem.put(path, [1, 0], v)
            for path, _ in rng.sample(chosen, min(8, len(chosen))):
                target = presented[path]
                pw = int(np.argmax(hash_feat(path) @ W))   # in-weights: recites
                pe = mem.get(path)
                acc_w += int(pw == target); acc_e += int(pe == target); n += 1
        return acc_w / n, acc_e / n
    out = {}
    out["FROM repo (fixed facts)"] = run(pool_from, False)
    out["TO repo (never seen)"] = run(pool_to, False)
    out["FROM repo, evidence SHUFFLED"] = run(pool_from, True)
    return out


def bench_transfer(sym_from, sym_to, tag_from: str, tag_to: str,
                   n_seq: int, epochs: int):
    section(f"TRANSFER — train on {tag_from}, evaluate on {tag_to}  (+ shuffle probe)")
    pool_f = [(p, pay["lineno"] % 32) for p, pay in sym_from]
    pool_t = [(p, pay["lineno"] % 32) for p, pay in sym_to]
    t0 = time.perf_counter()
    W = train_memorizer(pool_f, epochs=epochs)
    row("memorizer training (softmax reg, numpy)",
        f"{(time.perf_counter()-t0)*1000:,.0f} ms", f"{len(pool_f)} facts × {epochs} epochs")
    res = eval_models(W, pool_f, pool_t, n_seq=n_seq)
    print()
    print(f"    {'evaluation':<36} {'in-weights':>12} {'echo memory':>12}   chance ≈ {1/32:.3f}")
    print(f"    {'-'*36} {'-'*12} {'-'*12}")
    for k, (aw, ae) in res.items():
        print(f"    {k:<36} {aw:>12.3f} {ae:>12.3f}")
        record("transfer", k, "in_weights", aw, "acc", f"{tag_from}->{tag_to}")
        record("transfer", k, "echo_memory", ae, "acc", f"{tag_from}->{tag_to}")
    aw_to = res["TO repo (never seen)"][0]
    aw_sh = res["FROM repo, evidence SHUFFLED"][0]
    verdict("echo memory == 1.000 on all three evaluations",
            all(ae == 1.0 for _, ae in res.values()))
    verdict("memorizer fails shuffle probe (≈chance) → it recites, not reads",
            aw_sh < 0.12, f"shuffled acc {aw_sh:.3f}")
    verdict("memorizer ≈ chance on unseen repo (competence not portable)",
            aw_to < 0.12, f"TO-repo acc {aw_to:.3f}")
    return res


# ============================================================================
# 6. Figures + CSV
# ============================================================================

def write_outputs(outdir: Path, no_plots: bool):
    import pandas as pd
    df = pd.DataFrame(RESULTS)
    csv_path = outdir / "echo_benchmarks.csv"
    df.to_csv(csv_path, index=False)
    figs = []
    if not no_plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="talk", palette="deep")

        d = df[df.section == "read_latency"]
        if len(d):
            plt.figure(figsize=(9, 4.6))
            ax = sns.barplot(d, x="value", y="benchmark", hue="repo", errorbar=None)
            ax.set(xlabel="median latency (µs, log scale)", ylabel="", xscale="log",
                   title="EchoMemory read latency — aggregates precomputed at write")
            plt.tight_layout(); p = outdir / "fig_read_latency.png"
            plt.savefig(p, dpi=140); plt.close(); figs.append(p)

        d = df[(df.section == "incremental") & (df.benchmark == "apply_commit")]
        f = df[(df.section == "incremental") & (df.benchmark == "full_rebuild")]
        if len(d):
            plt.figure(figsize=(9, 4.6))
            ax = sns.stripplot(d, x="repo", y="value", size=8, alpha=.8)
            for _, r in f.iterrows():
                ax.axhline(r.value, ls="--", c="crimson", lw=1.5)
            ax.text(0.02, f.value.max() * 0.92, "full rebuild", color="crimson",
                    transform=ax.get_yaxis_transform())
            ax.set(ylabel="ms per commit (log)", xlabel="", yscale="log",
                   title="Incremental replay: one real commit applied to the echo index")
            plt.tight_layout(); p = outdir / "fig_incremental.png"
            plt.savefig(p, dpi=140); plt.close(); figs.append(p)

        d = df[df.section == "transfer"].copy()
        if len(d):
            plt.figure(figsize=(10.5, 4.8))
            ax = sns.barplot(d, x="benchmark", y="value", hue="metric", errorbar=None)
            ax.axhline(1 / 32, ls=":", c="gray"); ax.text(1.7, 1/32 + .02, "chance", c="gray")
            ax.set(ylim=(0, 1.05), ylabel="answer accuracy", xlabel="",
                   title="Three memories: in-weights recites; echo memory reads")
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels([t.get_text().replace(" (", "\n(") for t in ax.get_xticklabels()])
            plt.legend(title="")
            plt.tight_layout(); p = outdir / "fig_transfer.png"
            plt.savefig(p, dpi=140); plt.close(); figs.append(p)

        d = df[(df.section == "scope") & (df.metric == "resolve_median")].copy()
        if len(d):
            d["depth"] = d.benchmark.str.replace("depth_", "").astype(int)
            plt.figure(figsize=(8, 4.2))
            ax = sns.lineplot(d, x="depth", y="value", hue="repo", marker="o")
            ax.set(xlabel="nesting depth", ylabel="resolve median (µs)",
                   title="Lexical resolution cost vs depth (exact at every depth)")
            plt.tight_layout(); p = outdir / "fig_scope.png"
            plt.savefig(p, dpi=140); plt.close(); figs.append(p)
    return csv_path, figs


# ============================================================================
# 7. Main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-repo", required=True, help="path or URL of the KNOWN repo")
    ap.add_argument("--to-repo", default=None, help="path or URL of the UNSEEN repo")
    ap.add_argument("--from-src", default=None, help="package prefix override, e.g. src/flask/")
    ap.add_argument("--to-src", default=None)
    ap.add_argument("--commits", type=int, default=25, help="commits for incremental replay")
    ap.add_argument("--iters", type=int, default=3000, help="timed calls per read op")
    ap.add_argument("--seqs", type=int, default=200, help="transfer eval sequences per cell")
    ap.add_argument("--epochs", type=int, default=30, help="memorizer training epochs")
    ap.add_argument("--outdir", default="echo_demo_results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.commits, a.iters, a.seqs, a.epochs = 8, 500, 50, 10

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    cache = outdir / "repo_cache"; cache.mkdir(exist_ok=True)

    banner("ECHO MEMORY — standalone demonstration & benchmark suite")
    print(f"  from-repo : {a.from_repo}")
    print(f"  to-repo   : {a.to_repo or '(none — transfer section skipped)'}")
    print(f"  outdir    : {outdir.resolve()}")

    repo_f = clone_if_url(a.from_repo, cache)
    pref_f = a.from_src or detect_src_prefix(repo_f)
    tag_f = Path(repo_f).name.replace(".git", "")
    print(f"  {tag_f}: prefix={pref_f}  HEAD={git(repo_f,'rev-parse','--short','HEAD').strip()}")

    m_f, sym_f = bench_index(repo_f, pref_f, tag_f)
    bench_reads(m_f, sym_f, tag_f, a.iters)
    bench_incremental(repo_f, pref_f, tag_f, a.commits)
    names = sorted({p[-1] for p, _ in sym_f})[:200]
    bench_scope(names, tag_f, trials=60 if not a.quick else 20)

    d1 = build_memory(sym_f).digest(); d2 = build_memory(sym_f).digest()
    section("DETERMINISM")
    verdict("two independent builds produce identical digests", d1 == d2, d1)

    if a.to_repo:
        repo_t = clone_if_url(a.to_repo, cache)
        pref_t = a.to_src or detect_src_prefix(repo_t)
        tag_t = Path(repo_t).name.replace(".git", "")
        print(f"\n  {tag_t}: prefix={pref_t}  HEAD={git(repo_t,'rev-parse','--short','HEAD').strip()}")
        m_t, sym_t = bench_index(repo_t, pref_t, tag_t)
        bench_reads(m_t, sym_t, tag_t, a.iters)
        bench_transfer(sym_f, sym_t, tag_f, tag_t, n_seq=a.seqs, epochs=a.epochs)

    csv_path, figs = write_outputs(outdir, a.no_plots)
    section("OUTPUTS")
    row("results CSV", str(csv_path))
    for f in figs:
        row("figure", str(f))
    print()
    hr("═")
    n_pass = sum(1 for r in RESULTS if r["section"] == "verdicts" and r["value"] == 1.0)
    n_all = sum(1 for r in RESULTS if r["section"] == "verdicts")
    print(f"  DONE — {n_pass}/{n_all} verdicts passed, {len(RESULTS)} metrics recorded.")
    hr("═")


if __name__ == "__main__":
    main()
