---
name: compact
description: Project-aware compaction for the SolarFlare V5 JEPA session. Gathers live experiment state, recent val curves, decisions made this session, then compacts with that context preserved. Invoke when context gets full or before ending a session. Replaces bare /compact for this project.
---

# Project-Aware Compact

Gather project state, build preserve-instructions message, then compact with it.

## Steps

### 1. Gather live state (parallel)

Read in parallel:

- `docs/V5_JEPA/INDEX.md` — entry-point hub (best results, active research, concept links).
- `docs/V5_JEPA/09_progress.md` — narrative source of truth.
- `docs/V5_JEPA/12_experiments.md` — live run log (table + active per-experiment sections).
- `docs/V5_JEPA/12_experiments_findings.md` — CONFIRMED/HYPOTHESIS findings.
- `scripts/slot_status.sh` output — which slots RUNNING, session names.

Tail latest events for each RUNNING slot:

- mini_mps: `tail -10 outputs_v5_mini_*/run.jsonl` (out_dir from active config).
- 5060ti_cuda: `ssh 5060ti "tail -10 /home/indra/solarflare/outputs_v5/run.jsonl"`.
- studio: equivalent path under studio working dir.

Extract per-job: latest epoch, latest val_loss, best_val so far, seconds-per-epoch.

### 2. Build preserve message

Terse, ≤80 lines:

```
SolarFlare V5 JEPA session state — preserve following:

RUNNING EXPERIMENTS:
<slot>: E<NN> <config_path> — ep<N>/<total>, val=<x>, best=<y> ep<z>, <s>s/ep, ETA <h>h
...

KEY FINDINGS THIS SESSION:
- E<NN>: <result with delta>  [CONFIRMED|HYPOTHESIS|STALE]
...

DECISIONS MADE:
- <what changed: config, curriculum knob, killed direction, etc.>

ACTIVE HYPOTHESES:
- <pending validations, follow-up experiments queued>

NEXT ACTIONS:
- <what was about to happen — next epoch milestone, next launch, doc update>

PROJECT STATE (stable facts, only repeat if changed this session):
- Active branch: v5-jepa-lora
- Active arch: Path B JEPA-from-scratch, ViT-Small + EMA target + block-causal predictor
- Path A (LoRA on Surya) KILLED — img_size lock
- Data: 21 AR cubes, 12-min cadence, wind flux clip 1e8
- Sanity scale: dim=192, 4 cubes (harp_17,83,45,51), t_in=4/t_out=2
- path_a scale: dim=384, 21 cubes, t_in=10/t_out=4, 33M trainable
- MPS gotcha: predictor routes around SDPA NaN under attn_mask + no_grad

PENDING DOC UPDATES:
- <experiments to append to 12_experiments.md on completion>
- <graphiti add_memory queued for when MCP online>
```

### 3. Compact with context

Say `Compacting with project context...`, show summary about to pass, then invoke built-in `/compact` with that summary as the preserve argument.

## Rules

- **Always** include latest val curve numbers for active jobs — they are non-recoverable from code.
- **Always** flag any in-progress edit to `12_experiments.md` or `09_progress.md` so post-compact session resumes the write.
- **Skip** static project facts already in CLAUDE.md or INDEX.md unless changed this session.
- **Tag** every finding with CONFIRMED/HYPOTHESIS/STALE per CLAUDE.md evidence rules.
- If graphiti MCP online, also queue `mcp__graphiti__add_memory` calls for newly CONFIRMED findings in the preserve message — execute post-compact.
