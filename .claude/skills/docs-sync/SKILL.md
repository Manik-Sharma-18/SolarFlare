---
name: docs-sync
description: Audit V5_JEPA docs for staleness, line-cap breaches, undocumented run.jsonl entries, evidence-tag gaps, INDEX.md drift. Read-only — propose edits, don't apply. Invoke at session end or after any completed experiment.
---

# docs-sync — V5_JEPA doc drift detector

Read-only audit. Propose edits, user reviews diffs. Output: terse grouped report (≤30 lines).

## When to invoke

- Session end (after appending new experiment rows).
- After any experiment marked CONFIRMED.
- Before opening a PR that touches `docs/V5_JEPA/**`.
- If suspect staleness during session.

## Severity grouping

- **BLOCKING** — 200-line cap breach; untagged hard-claim in `_findings.md` or `09_progress.md`.
- **WARN** — stale-claim leak; run.jsonl entry not in `12_experiments.md`; INDEX best-results table missing CONFIRMED finding.
- **INFO** — date-anchor drift >7d; line count >180 approaching cap.

## Checks

Run each in sequence. Aggregate output into report at end. All shell from repo root.

### 1. Line-cap audit (BLOCKING if >200)

```bash
wc -l docs/V5_JEPA/*.md docs/V5_JEPA/concepts/*.md CLAUDE.md | awk '$1 > 200 {print "BLOCKING: " $0} $1 > 180 && $1 <= 200 {print "INFO: approaching cap: " $0}'
```

### 2. Stale-claim grep (WARN)

```bash
# Path A reintroduction risk — any live mention not flagged abandoned/killed/deprecated/ABANDONED
grep -rni "path a" docs/V5_JEPA/ | grep -vi "abandon\|killed\|deprecate\|kill"

# Stale data/clip claims
grep -rn "BZ_CLIP\|1e5 clip\|10 cubes\|14 legacy\|14 AR\|14 cubes" docs/V5_JEPA/ | grep -v "concepts/\|archive\|history\|RESOLVED\|prior\|old"

# Reintroduction risk for abandoned deps
grep -rn "transformers\|peft\|huggingface_hub" docs/V5_JEPA/ | grep -vi "don't\|abandon\|killed\|drop"

# OUTLIERS.md ref (file moved to concepts/wind_flux_clipping.md)
grep -rn "OUTLIERS\.md\|03_masks_and_pathak\.md" docs/V5_JEPA/ CLAUDE.md
```

### 3. Run.jsonl freshness (WARN)

```bash
for d in outputs_v5*/; do
  jsonl="${d}run.jsonl"
  [ -f "$jsonl" ] || continue
  last_ep=$(tail -1 "$jsonl" | python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('epoch','?'))" 2>/dev/null)
  echo "$d → last_ep=$last_ep"
done
```

Cross-check each `outputs_v5*` dir against an E-number in `12_experiments.md`. Dirs with ≥3 epochs but no row → WARN.

If `outputs_v5_*` dirs don't exist (fresh clone), skip silently.

### 4. INDEX best-results drift (WARN)

```bash
# Extract CONFIRMED entries from findings
grep -E "^\*\*Tag:\*\* CONFIRMED" docs/V5_JEPA/12_experiments_findings.md -B1 | head -40
```

Compare against rows in `docs/V5_JEPA/INDEX.md` best-results table. Any CONFIRMED finding with a quantitative result missing from INDEX → WARN.

### 5. Evidence-tag check (BLOCKING)

```bash
# Findings file: every claim block must end with **Tag:** CONFIRMED|HYPOTHESIS|STALE
python3 -c "
import re, pathlib
p = pathlib.Path('docs/V5_JEPA/12_experiments_findings.md').read_text()
sections = re.split(r'\n## F\d', p)
for i, s in enumerate(sections[1:], 1):
    if not re.search(r'\*\*Tag:\*\*\s+(CONFIRMED|HYPOTHESIS|STALE)', s):
        print(f'BLOCKING: F{i} missing evidence tag')
"
```

### 6. Date-anchor check (INFO)

```bash
today=$(date +%Y-%m-%d)
prog_date=$(grep -m1 "^\*\*Date:\*\*" docs/V5_JEPA/09_progress.md | grep -oE "20[0-9]{2}-[0-9]{2}-[0-9]{2}")
claude_date=$(grep -m1 "^## Where we are" CLAUDE.md | grep -oE "20[0-9]{2}-[0-9]{2}-[0-9]{2}")
index_date=$(grep -m1 "^Date:" docs/V5_JEPA/INDEX.md | grep -oE "20[0-9]{2}-[0-9]{2}-[0-9]{2}")
echo "today=$today | 09_progress=$prog_date | CLAUDE=$claude_date | INDEX=$index_date"
```

Flag INFO if any date >7 days behind `today`.

## Output format

```
docs-sync report (date)

BLOCKING (must fix):
- <file>:<line>: <claim>

WARN (review):
- <file>:<line>: <claim>

INFO:
- <metric>

Suggested edits:
- <file>: <one-line proposed change>
```

## Rules

- **Read-only.** Never apply edits. Propose in "Suggested edits" section. User reviews diffs.
- **Skip CLAUDE.md content audit** (project bootstrap; INDEX.md owns truth). Only check CLAUDE.md for line cap + OUTLIERS.md ref.
- **Skip `outputs_v5_*`** if dirs absent (fresh clone).
- **Don't flag historical/archival mentions** — pattern excludes match `concepts/`, `archive`, `RESOLVED`, `prior`, `old`.
- **Findings tag rule** — every F-section in `12_experiments_findings.md` MUST carry `**Tag:** CONFIRMED|HYPOTHESIS|STALE`. Untagged = BLOCKING.

## Examples

After E13 (tube+future) completes:
1. Append row to `12_experiments.md` summary table.
2. If CONFIRMED, add F-section to `_findings.md` with `**Tag:** CONFIRMED`.
3. Invoke `docs-sync` → expect 0 BLOCKING.
4. If `outputs_v5_e13_tube_future_cuda/run.jsonl` has ≥3 epochs but no table row → WARN "undocumented run."
5. If INDEX best-results table missing the new row → WARN "INDEX drift."

After CLAUDE.md edit:
1. `wc -l CLAUDE.md` — flag BLOCKING if >200.
2. Grep for `OUTLIERS.md` ref — flag WARN if present (file moved 2026-05-12).
