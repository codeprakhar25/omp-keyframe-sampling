# POD RUNBOOK — repeats=3 pilot on RunPod (egress-locked)

The final stretch: run repeats 1 + 2 for all 30 cells (60 new runs) on a
RunPod box, with GitHub blackholed so the agent **cannot** push/PR (the PR-#953
class of incident is physically impossible), DB persisted, live monitoring.

**You only do the steps in `> RUN:` blocks.** Everything else is context.

---

## 0. What runs, what it costs

- **Dataset:** 10 tasks × 3 strategies × 3 repeats = 90 cells. 30 (repeat-0) are
  already in the DB and **audit-clean** (no cell ever ran git push/commit/gh).
  The resumable driver **skips them** → **60 new cells**.
- **Cost:** ≈ **$40–45** (firebase ~$0.4, pdm ~$1.2, opshin ~$1–2 per cell).
- **Wall time:** ≈ **4–5 h** (opshin is the long pole; worst observed cell 77 min).
- **Pod size:** **8 vCPU / 16 GB+** (3 concurrent agents + pytest + uv). CPU only —
  no GPU needed.

---

## 1. Safety model (why this is safe to run unattended)

Five independent layers stop a push. Layer 5 (new) is the physical one.

1. `scrub_git_remotes` — workspace has no remote + a dead pushurl.
2. `strip_future_history` — agent can't read the gold commit from git log.
3. `--disallowedTools` — `git push/commit/remote`, `gh` denied at the agent.
4. `GH_TOKEN=""`/`GITHUB_TOKEN=""` — no GitHub creds in the agent env.
5. **GitHub blackholed in `/etc/hosts`** — even a slipped push has nowhere to
   resolve. Verified by `verify_lock.sh` before any agent starts.

PyPI + `api.anthropic.com` stay reachable (eval installs deps; agent calls LLM).

---

## 2. Create the pod

> **RUN (RunPod web):**
> - **Template:** a CPU pod, Ubuntu + Python 3.12 base (e.g. `runpod/base:...`
>   or any `python:3.12` image). No GPU.
> - **Disk:** container 20 GB+; **attach a Network Volume** (≥30 GB) mounted at
>   `/workspace` — this is what survives pod stop/restart.
> - **Expose SSH.** Copy the SSH command RunPod shows
>   (`ssh root@<ip> -p <port> -i <key>`).

Everything below lives under `/workspace` so it's on the persistent volume.

---

## 3. Get the code + key onto the pod

> **RUN (laptop):**
> ```bash
> # from the repo root on your laptop
> cd /home/prakh/ml-resarch
> # ship the experiment dir (excluding heavy/transient stuff) to the volume
> rsync -az -e "ssh -p <port> -i <key>" \
>   --exclude repos/ --exclude .uv-cache/ --exclude '*.html' \
>   experiment/ root@<ip>:/workspace/experiment/
> # ship the API key file (NEVER commit it; scp it)
> scp -P <port> -i <key> experiment/.env root@<ip>:/workspace/experiment/.env
> ```
>
> `.env` must contain **only** `ANTHROPIC_API_KEY=...`.
> **Do NOT** put `GH_TOKEN`/`GITHUB_TOKEN` on the pod — `verify_lock.sh` fails if found.

(Alternative: `git clone` your repo on the pod — but then GitHub auth lives on
the pod. The rsync path keeps GitHub creds off the box entirely. Prefer rsync.)

---

## 4. Setup phase (network OPEN, no agent)

> **RUN (pod):**
> ```bash
> ssh root@<ip> -p <port> -i <key>
> cd /workspace/experiment
> chmod +x pod/*.sh
> bash pod/setup.sh
> ```

Installs uv/node/claude CLI + deps, **full-clones all 3 repos**, and warms the
uv cache + builds firebase's venv via `calibrate_eval` (no API, no agent).
Takes ~10–20 min. Must end with `SETUP COMPLETE`.

---

## 5. Lock GitHub, then verify (HARD GATE)

> **RUN (pod):**
> ```bash
> bash pod/firewall.sh        # blackhole github in /etc/hosts (+ iptables if able)
> set -a; source .env; set +a # load ANTHROPIC_API_KEY into the shell
> bash pod/verify_lock.sh     # MUST print "LOCK VERIFIED" and exit 0
> ```

`verify_lock.sh` asserts: GitHub unreachable, PyPI+Anthropic reachable, repos
full-cloned, key present, no GitHub token. **If it fails, STOP — do not run the
pilot.** `run_repeats.sh` also re-checks this and refuses to launch if red.

---

## 6. Single-cell smoke test (~$0.40, proves the lock)

Before the $40 run, prove one agent cell completes end-to-end under lock and
**cannot** push.

> **RUN (pod):**
> ```bash
> set -a; source .env; set +a
> EXP_LIVE=1 python3 -m harness --task-file tasks/pilot.json \
>   --strategy always_on --agent claude \
>   --task-id firebase__firebase-admin-python__940
> ```

Watch for: `✅ result` (completed), an eval line `pass=.. fail=0`, and **no**
`🚨 PUSH/PR ATTEMPT`. If the agent tries git/gh you'll see it denied — that's
the layers working. Green → proceed.

---

## 7. Launch the full repeats=3 run

> **RUN (pod):**
> ```bash
> bash pod/run_repeats.sh
> ```

Launches tmux session `pilot` with 4 windows: `firebase`, `pdm`, `opshin`,
`sync`. Each repo runs `run_pilot.py --repeats 3 --repo <repo>` (skips the 30
done cells). Caps are set high (cost $6, abs 7200s, inactivity 1800s).

> **RUN (pod) — watch it:**
> ```bash
> tmux attach -t pilot       # Ctrl-b <0..3> switch windows; Ctrl-b d detach
> # or, glanceable status without attaching:
> bash pod/monitor.sh -w
> ```

---

## 8. Pull results to your laptop (durability + audit trail)

> **RUN (laptop, in a second terminal, leave running):**
> ```bash
> cd /home/prakh/ml-resarch/experiment
> POD='root@<ip> -p <port> -i <key>' bash pod/pull_results.sh
> ```

Pulls DB + JSON + logs + stream files every 120 s into `results_from_pod/`.
This is the off-pod copy that survives even if the RunPod volume is deleted.

---

## 9. When it finishes

`monitor.sh` should show **90 clean cells**, `by repeat_index {0:30, 1:30, 2:30}`,
**zero push alarms**.

> **RUN (pod):**
> ```bash
> python3 analyze.py --agent claude_code
> # final pull, then verify the DB locally on the laptop
> ```
> **RUN (laptop):**
> ```bash
> sqlite3 results_from_pod/experiment.db \
>   "SELECT task_id,strategy,repeat_index,total_turns,total_cache_read_tokens,task_passed FROM runs WHERE agent='claude_code' AND eval_method IS NOT NULL ORDER BY task_id,strategy,repeat_index;"
> ```

Merge `results_from_pod/experiment.db` back as the dataset of record, then run
the paired-Wilcoxon stats (turns + cache_read, per strategy) like Paper 1.

---

## 10. Teardown

> **RUN:** Stop the pod from the RunPod console. The **Network Volume keeps the
> data**; you can delete the pod. Keep the volume until the laptop copy is
> confirmed complete.

---

## 11. If something goes wrong

| Symptom | Action |
|---|---|
| `verify_lock.sh` says GitHub reachable | re-run `firewall.sh`; check `/etc/hosts` has the `EXP-EGRESS-LOCK` block |
| eval fails "could not resolve" / pip errors | PyPI got blocked — `verify_lock.sh` should catch; ensure only GitHub is blackholed |
| an opshin cell killed again | bump just that run: `EXP_ABSOLUTE_TIMEOUT=10800 EXP_MAX_COST_USD=10 python3 run_pilot.py --repeats 3 --repo opshin` |
| 🚨 push alarm in a log | the layers blocked it (no creds + no remote + blackhole); note the run_id, it's still safe — inspect for the paper |
| pod died mid-run | restart pod (volume persists), re-`source .env`, re-run `pod/run_repeats.sh` — resumable driver continues |
| DB locked errors | harmless under concurrency (busy_timeout=60s); driver retries |

**STOP everything:** `tmux kill-session -t pilot; pkill -9 -f run_pilot.py; pkill -9 -f 'claude --print'`

---

## 12. Codex arm (second agent — gap #1 fix)

Replicates the **same 11 tasks × 3 strategies × 3 repeats = 99 cells** on OpenAI
Codex (`gpt-5.5`), into the same `results/experiment.db` with `agent='codex'`
(the `claude_code` rows are untouched). Auth = **ChatGPT-plan login** (free quota,
no API key). GPT-5.5 needs ChatGPT-account auth — matches this path.

**Safety = the claude layers AND more:**
- `--sandbox workspace-write` (writes confined to the workspace, agent-command
  network disabled) — extra isolation the claude arm lacked.
- **PreToolUse deny-hook** (`pod/codex_home/deny_push.sh`) — the deny-side analog
  of claude's `--disallowedTools`; blocks `git push/commit/remote`, `gh`,
  `request-pull` and logs `🚨 …BLOCKED-BY-HOOK` to `/tmp/codex_deny.log`. (Codex's
  own command-rules are bypassable via `--ignore-rules`; a hook is not.)
- **gh PATH-shim** (`pod/codex_home/bin/gh`) — prepended to the agent's PATH so the
  real GitHub CLI is shadowed by a stub that exits 1 + logs. Zero-fragility,
  agent-agnostic belt-and-suspenders on top of the hook.
- Clean `CODEX_HOME` + `--ephemeral` = `--bare` analog (no global AGENTS.md/config
  leak; `--ephemeral` alone only skips session persistence — the clean home is what
  stops AGENTS.md/config auto-load).
- Reused: egress lock (`firewall.sh`), `scrub_git_remotes`/`strip_future_history`
  (runner), env credential scrub, watchdog timers + `EXP_MAX_TURNS` (codex has no
  native budget/turn flag).

> **RUN (one-time):**
> ```bash
> bash pod/codex_setup.sh                 # install codex CLI + render deny-hook
> CODEX_HOME=$PWD/pod/codex_home codex login   # device-code; or scp ~/.codex/auth.json into pod/codex_home/
> bash pod/verify_lock.sh codex           # HARD GATE (GitHub blocked, OpenAI reachable, authed, hook present)
> ```
> **RUN (smoke first, scratch db):**
> ```bash
> bash pod/run_codex_smoke.sh             # 3 tasks × none → results/codex_smoke.db + planted-push deny test
> ```
> **RUN (full arm, after smoke looks right):**
> ```bash
> bash pod/run_codex_repeats.sh           # 99 cells → experiment.db (agent='codex'); tmux session 'codex'
> ```

**Analyze:** `python3 analyze.py --agent codex`; cross-agent compare uses the
**portable** metric only (turns / wall-time / total tokens) — `cache_*` is
Claude-specific and has no Codex analog (`cached_input_tokens` is read-only; no
cache-creation concept).

**STOP the codex arm:** `tmux kill-session -t codex; pkill -9 -f 'run_pilot.py --agent codex'; pkill -9 -f 'codex exec'`
