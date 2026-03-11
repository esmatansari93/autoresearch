"""
Autonomous AI Research Agent for autoresearch.
Iteratively modifies train.py, runs experiments, and keeps improvements.

Supports Gemini, OpenAI, and Anthropic APIs (auto-detected from env vars).

Usage:
    set GEMINI_API_KEY=your-key-here       (or OPENAI_API_KEY or ANTHROPIC_API_KEY)
    uv run agent.py                        (starts autonomous loop)
    uv run agent.py --tag mar11            (custom branch tag)
    uv run agent.py --max-experiments 10   (limit number of experiments)
"""

import os
import re
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(REPO_DIR, "train.py")
PREPARE_FILE = os.path.join(REPO_DIR, "prepare.py")
RESULTS_FILE = os.path.join(REPO_DIR, "results.tsv")
RUN_LOG = os.path.join(REPO_DIR, "run.log")
TRAIN_TIMEOUT = 600  # 10 minutes max per run
MAX_CRASH_RETRIES = 2

# ---------------------------------------------------------------------------
# LLM Client — auto-detects API from environment variables
# ---------------------------------------------------------------------------

try:
    import requests as _requests
except ImportError:
    print("ERROR: 'requests' package not found. Run: uv sync")
    sys.exit(1)


def _detect_api():
    """Auto-detect which LLM API key is available."""
    for name, env_var in [("gemini", "GEMINI_API_KEY"), ("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")]:
        if os.environ.get(env_var):
            return name, os.environ[env_var]
    return None, None


def ask_llm(system_prompt, user_prompt, temperature=0.7):
    """Send a prompt to the LLM and return the response text."""
    api_name, api_key = _detect_api()
    if api_name is None:
        print("ERROR: No API key found. Set one of: GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY")
        sys.exit(1)

    if api_name == "gemini":
        return _call_gemini(api_key, system_prompt, user_prompt, temperature)
    elif api_name == "openai":
        return _call_openai(api_key, system_prompt, user_prompt, temperature)
    elif api_name == "anthropic":
        return _call_anthropic(api_key, system_prompt, user_prompt, temperature)


def _call_gemini(api_key, system_prompt, user_prompt, temperature):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent?key={api_key}"
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": 16384},
    }
    resp = _requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_openai(api_key, system_prompt, user_prompt, temperature):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": 16384,
    }
    resp = _requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _call_anthropic(api_key, system_prompt, user_prompt, temperature):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 16384,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
    }
    resp = _requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


# ---------------------------------------------------------------------------
# System prompt for the LLM researcher
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an autonomous ML researcher. Your goal is to achieve the lowest possible \
val_bpb (validation bits per byte) by modifying a GPT training script.

RULES:
1. You can ONLY modify train.py. Never modify prepare.py.
2. You cannot add new dependencies — only use what's in pyproject.toml.
3. Training runs for a fixed 5-minute time budget. You optimize WHAT trains, not HOW LONG.
4. Lower val_bpb is better. The metric is vocab-size-independent.
5. VRAM should not blow up dramatically. Small increases for meaningful gains are OK.
6. Simpler is better. Don't add complexity unless it clearly helps.
7. Be creative: change architecture, hyperparameters, optimizer settings, batch size, \
model size, activation functions, attention patterns — everything in train.py is fair game.

OUTPUT FORMAT:
Return the COMPLETE modified train.py inside a single ```python code block.
Do not return partial diffs or explanations outside the code block.
Before the code block, write ONE LINE describing what you changed and why.

IMPORTANT: The code must be syntactically valid Python that runs without errors.\
"""

# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(*args):
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git"] + list(args),
        cwd=REPO_DIR, capture_output=True, text=True, timeout=30,
    )
    return result.stdout.strip(), result.returncode


def git_current_branch():
    out, _ = git("rev-parse", "--abbrev-ref", "HEAD")
    return out


def git_short_hash():
    out, _ = git("rev-parse", "--short=7", "HEAD")
    return out


def git_commit(message):
    git("add", "train.py")
    git("commit", "-m", message)
    return git_short_hash()


def git_reset_hard(ref="HEAD~1"):
    git("reset", "--hard", ref)


def git_create_branch(branch_name):
    _, rc = git("checkout", "-b", branch_name)
    if rc != 0:
        # Branch already exists, just switch to it
        git("checkout", branch_name)


def git_has_changes():
    out, _ = git("diff", "--name-only")
    return bool(out.strip())

# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment():
    """Run train.py and return (val_bpb, peak_vram_mb, success)."""
    try:
        with open(RUN_LOG, "w") as log_file:
            proc = subprocess.run(
                ["uv", "run", "train.py"],
                cwd=REPO_DIR,
                stdout=log_file, stderr=subprocess.STDOUT,
                timeout=TRAIN_TIMEOUT,
            )
        return parse_results()
    except subprocess.TimeoutExpired:
        print("    ⏰ Run timed out (>10 min), treating as crash")
        return None, None, False
    except Exception as e:
        print(f"    ❌ Run error: {e}")
        return None, None, False


def parse_results():
    """Parse val_bpb and peak_vram_mb from run.log."""
    try:
        with open(RUN_LOG, "r") as f:
            content = f.read()
    except FileNotFoundError:
        return None, None, False

    val_bpb = None
    peak_vram = None

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("val_bpb:"):
            try:
                val_bpb = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("peak_vram_mb:"):
            try:
                peak_vram = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass

    if val_bpb is not None:
        return val_bpb, peak_vram, True
    else:
        return None, None, False


def get_crash_log():
    """Read the last 50 lines of run.log for crash diagnosis."""
    try:
        with open(RUN_LOG, "r") as f:
            lines = f.readlines()
        return "".join(lines[-50:])
    except FileNotFoundError:
        return "No run.log found"

# ---------------------------------------------------------------------------
# Results logger
# ---------------------------------------------------------------------------

TSV_HEADER = "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n"


def init_results():
    """Create results.tsv if it doesn't exist."""
    if not os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "w") as f:
            f.write(TSV_HEADER)


def log_result(commit_hash, val_bpb, memory_gb, status, description):
    """Append a result row to results.tsv."""
    val_str = f"{val_bpb:.6f}" if val_bpb else "0.000000"
    mem_str = f"{memory_gb:.1f}" if memory_gb else "0.0"
    # Sanitize description (no tabs or newlines)
    desc = description.replace("\t", " ").replace("\n", " ").strip()[:200]
    with open(RESULTS_FILE, "a") as f:
        f.write(f"{commit_hash}\t{val_str}\t{mem_str}\t{status}\t{desc}\n")


def load_results():
    """Load past results for context."""
    if not os.path.exists(RESULTS_FILE):
        return ""
    with open(RESULTS_FILE, "r") as f:
        return f.read()

# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code_block(response):
    """Extract Python code from LLM response (inside ```python ... ``` block)."""
    # Try ```python ... ``` first
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        # Return the longest match (in case there are multiple)
        return max(matches, key=len).strip()

    # Try generic ``` ... ```
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return max(matches, key=len).strip()

    return None


def extract_description(response):
    """Extract the one-line description before the code block."""
    lines = response.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line and not line.startswith("```"):
            return line[:200]
    return "no description"

# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def build_user_prompt(train_code, results_history, best_bpb, experiment_num):
    """Build the user prompt with current context."""
    prompt = f"""## Current train.py

```python
{train_code}
```

## Past Experiment Results

{results_history if results_history.strip() else "No experiments yet — this will be the first modification after baseline."}

## Current Best val_bpb: {best_bpb if best_bpb else "unknown (baseline not yet run)"}

## Experiment #{experiment_num}

Based on the code and past results, propose ONE targeted modification to improve val_bpb.
Think about what has worked, what hasn't, and try something new.
Return the COMPLETE modified train.py in a ```python code block.
"""
    return prompt


def build_fix_prompt(train_code, crash_log):
    """Build a prompt for the LLM to fix a crash."""
    return f"""## train.py (crashed)

```python
{train_code}
```

## Crash Log (last 50 lines)

```
{crash_log}
```

The above code crashed during training. Fix the bug and return the COMPLETE corrected \
train.py in a ```python code block. Only fix the crash — don't make other changes.
"""


def main():
    parser = argparse.ArgumentParser(description="Autonomous AI Research Agent")
    parser.add_argument("--tag", type=str, default=None,
                        help="Branch tag (default: today's date, e.g. mar11)")
    parser.add_argument("--max-experiments", type=int, default=0,
                        help="Max experiments to run (0 = unlimited)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip the baseline run")
    args = parser.parse_args()

    # Detect API
    api_name, _ = _detect_api()
    if api_name is None:
        print("❌ No LLM API key found!")
        print("   Set one of: GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY")
        sys.exit(1)
    print(f"🤖 Using {api_name.upper()} API" + (" (gemini-3-flash-preview free tier)" if api_name == "gemini" else ""))

    # Setup branch
    tag = args.tag or datetime.now().strftime("%b%d").lower()
    branch_name = f"autoresearch/{tag}"
    print(f"🌿 Branch: {branch_name}")
    git_create_branch(branch_name)

    # Verify data exists
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
    if not os.path.isdir(cache_dir):
        print(f"❌ Data not found at {cache_dir}")
        print("   Run: uv run prepare.py")
        sys.exit(1)
    print(f"📁 Data directory: {cache_dir}")

    # Init results
    init_results()
    best_bpb = None

    # Read existing results to find current best
    results = load_results()
    for line in results.strip().split("\n")[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) >= 4 and parts[3] == "keep":
            try:
                bpb = float(parts[1])
                if best_bpb is None or bpb < best_bpb:
                    best_bpb = bpb
            except ValueError:
                pass

    # -----------------------------------------------------------------------
    # Step 1: Baseline run
    # -----------------------------------------------------------------------
    if not args.skip_baseline and best_bpb is None:
        print("\n" + "=" * 60)
        print("📊 Running baseline experiment...")
        print("=" * 60)

        val_bpb, peak_vram, success = run_experiment()
        commit_hash = git_short_hash()

        if success:
            memory_gb = peak_vram / 1024 if peak_vram else 0
            best_bpb = val_bpb
            log_result(commit_hash, val_bpb, memory_gb, "keep", "baseline")
            print(f"✅ Baseline: val_bpb={val_bpb:.6f}, memory={memory_gb:.1f}GB")
        else:
            print("❌ Baseline failed! Fix train.py and try again.")
            log_result(commit_hash, None, None, "crash", "baseline (crashed)")
            sys.exit(1)

    # -----------------------------------------------------------------------
    # Step 2: Experiment loop
    # -----------------------------------------------------------------------
    experiment_num = 1
    max_exp = args.max_experiments

    print("\n" + "=" * 60)
    print("🔬 Starting autonomous experiment loop")
    print(f"   Best val_bpb so far: {best_bpb}")
    if max_exp > 0:
        print(f"   Max experiments: {max_exp}")
    else:
        print("   Running indefinitely (Ctrl+C to stop)")
    print("=" * 60)

    try:
        while True:
            if max_exp > 0 and experiment_num > max_exp:
                print(f"\n🏁 Reached max experiments ({max_exp}). Stopping.")
                break

            print(f"\n{'─' * 60}")
            print(f"🧪 Experiment #{experiment_num}")
            print(f"{'─' * 60}")

            # Read current train.py
            with open(TRAIN_FILE, "r") as f:
                current_code = f.read()

            # Save a copy for rollback
            original_code = current_code

            # Ask LLM for an experiment idea
            print("  🤔 Asking LLM for experiment idea...")
            results_history = load_results()
            user_prompt = build_user_prompt(current_code, results_history, best_bpb, experiment_num)

            try:
                response = ask_llm(SYSTEM_PROMPT, user_prompt)
            except Exception as e:
                print(f"  ❌ LLM API error: {e}")
                print("  ⏳ Waiting 30s before retry...")
                time.sleep(30)
                experiment_num += 1
                continue

            # Extract code and description
            new_code = extract_code_block(response)
            description = extract_description(response)
            print(f"  📝 Idea: {description}")

            if new_code is None:
                print("  ⚠️  LLM didn't return a code block, skipping")
                experiment_num += 1
                continue

            # Syntax check before writing
            try:
                compile(new_code, "train.py", "exec")
            except SyntaxError as e:
                print(f"  ⚠️  LLM returned invalid syntax: {e}")
                experiment_num += 1
                continue

            # Write modified train.py
            with open(TRAIN_FILE, "w") as f:
                f.write(new_code)

            # Git commit
            commit_hash = git_commit(f"experiment #{experiment_num}: {description[:80]}")
            print(f"  📌 Committed: {commit_hash}")

            # Run experiment (with crash retry)
            success = False
            for attempt in range(1 + MAX_CRASH_RETRIES):
                print(f"  🚀 Running training{'  (retry ' + str(attempt) + ')' if attempt > 0 else ''}...")
                val_bpb, peak_vram, success = run_experiment()

                if success:
                    break

                if attempt < MAX_CRASH_RETRIES:
                    print("  💥 Crashed! Asking LLM to fix...")
                    crash_log = get_crash_log()
                    with open(TRAIN_FILE, "r") as f:
                        crash_code = f.read()
                    fix_prompt = build_fix_prompt(crash_code, crash_log)
                    try:
                        fix_response = ask_llm(SYSTEM_PROMPT, fix_prompt)
                        fix_code = extract_code_block(fix_response)
                        if fix_code:
                            try:
                                compile(fix_code, "train.py", "exec")
                                with open(TRAIN_FILE, "w") as f:
                                    f.write(fix_code)
                                git("add", "train.py")
                                git("commit", "--amend", "-m", f"experiment #{experiment_num}: {description[:80]} (fix)")
                            except SyntaxError:
                                print("  ⚠️  Fix has syntax errors, giving up")
                                break
                        else:
                            print("  ⚠️  LLM couldn't produce a fix")
                            break
                    except Exception as e:
                        print(f"  ❌ LLM fix error: {e}")
                        break

            # Evaluate results
            memory_gb = peak_vram / 1024 if peak_vram else 0

            if not success:
                # Crash — revert
                print(f"  💥 CRASH — reverting")
                log_result(commit_hash, None, None, "crash", description)
                with open(TRAIN_FILE, "w") as f:
                    f.write(original_code)
                git_reset_hard("HEAD~1")

            elif val_bpb is not None and (best_bpb is None or val_bpb < best_bpb):
                # Improvement — keep!
                improvement = best_bpb - val_bpb if best_bpb else 0
                print(f"  ✅ KEEP — val_bpb={val_bpb:.6f} (improved by {improvement:.6f}), memory={memory_gb:.1f}GB")
                log_result(commit_hash, val_bpb, memory_gb, "keep", description)
                best_bpb = val_bpb

            else:
                # No improvement — discard
                print(f"  ❌ DISCARD — val_bpb={val_bpb:.6f} (best={best_bpb:.6f}), memory={memory_gb:.1f}GB")
                log_result(commit_hash, val_bpb, memory_gb, "discard", description)
                with open(TRAIN_FILE, "w") as f:
                    f.write(original_code)
                git_reset_hard("HEAD~1")

            experiment_num += 1

    except KeyboardInterrupt:
        print(f"\n\n🛑 Stopped by user after {experiment_num - 1} experiments")

    # Final summary
    print("\n" + "=" * 60)
    print("📋 Final Results")
    print("=" * 60)
    print(f"Best val_bpb: {best_bpb}")
    print(f"Experiments run: {experiment_num - 1}")
    print(f"Results log: {RESULTS_FILE}")
    print(load_results())


if __name__ == "__main__":
    main()
