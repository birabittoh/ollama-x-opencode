#!/usr/bin/env python3
import json
import pathlib
import re
import subprocess
import sys
import urllib.parse
import urllib.request

OPENCODE_CONFIG = pathlib.Path.home() / ".config" / "opencode" / "config.json"

CONTEXT_SIZES = [8192, 16384, 32768, 65536, 131072]
CONTEXT_LABELS = ["8k", "16k", "32k", "64k", "128k"]
VARIANT_SUFFIXES = {f"-{label}" for label in CONTEXT_LABELS}

WSL_MEMORY_HINT = """
  Hint: Docker is running inside WSL2, which caps available memory by default.
  If your physical RAM is larger than what ollama sees, add this to
  C:\\Users\\<you>\\.wslconfig:

    [wsl2]
    memory=24GB

  Then run: wsl --shutdown  (and restart Docker Desktop)
"""


# --- ollama helpers ---

def ollama(*args: str, stream: bool = False) -> subprocess.CompletedProcess:
    cmd = ["docker", "compose", "exec", "ollama", "ollama", *args]
    if stream:
        return subprocess.run(cmd)
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def ollama_bash(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", "exec", "ollama", "bash", "-c", script],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def is_wsl() -> bool:
    result = ollama_bash("cat /proc/version")
    return "microsoft" in result.stdout.lower()


def fetch_all_models() -> list[str]:
    result = ollama("list")
    result.check_returncode()
    return [line.split()[0] for line in result.stdout.strip().splitlines()[1:]]


def split_base_and_variants(all_models: list[str]) -> tuple[list[str], list[str], set[str]]:
    base = [m for m in all_models if not any(m.endswith(s) for s in VARIANT_SUFFIXES)]
    variants = [m for m in all_models if any(m.endswith(s) for s in VARIANT_SUFFIXES)]
    has_variants = {
        b for b in base
        if any(f"{b}-{label}" in variants for label in CONTEXT_LABELS)
    }
    return base, variants, has_variants


# --- ollama.com scraping ---

def http_fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "ollama-manager/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read().decode("utf-8")


def scrape_search(query: str) -> list[str]:
    html = http_fetch(f"https://ollama.com/search?q={urllib.parse.quote(query)}")
    nav = {"blog", "docs", "download", "pricing", "about", "signin", "signup", "public", "search"}
    seen: set[str] = set()
    results = []
    for m in re.finditer(r'href="/([^"#?]+)"', html):
        parts = m.group(1).split("/")
        if parts[0] == "library" and len(parts) == 2:
            name = parts[1]  # official model
        elif len(parts) == 2 and parts[0] not in nav:
            name = "/".join(parts)  # community model: user/model
        else:
            continue
        # skip file-like names (e.g. icon.png — not model names like qwen2.5-coder)
        if re.search(r'\.[a-zA-Z]{2,4}$', parts[-1]):
            continue
        if name not in seen:
            seen.add(name)
            results.append(name)
    return results


def scrape_tags(model: str) -> list[tuple[str, str]]:
    slug = f"library/{model}" if "/" not in model else model
    html = http_fetch(f"https://ollama.com/{slug}/tags")
    # Split on each hidden command input; groups capture the tag value
    parts = re.split(
        r'<input[^>]+class="command hidden"[^>]+value="([^"]+)"[^>]*/>', html
    )
    SIZE_RE = re.compile(
        r'<p class="col-span-2 text-neutral-500 text-\[13px\]">([^<]+)</p>'
    )
    results = []
    for i in range(1, len(parts), 2):
        tag = parts[i]
        size_m = SIZE_RE.search(parts[i + 1] if i + 1 < len(parts) else "")
        size = size_m.group(1).strip() if size_m else "?"
        results.append((tag, size))
    return results


# --- prompts ---

def pick(items: list[str], prompt: str = "Select") -> str | None:
    if not items:
        print("  (none)")
        return None
    for i, item in enumerate(items):
        print(f"  [{i + 1}] {item}")
    raw = input(f"\n{prompt} (number, or 0 to cancel): ").strip()
    if not raw.isdigit():
        print("Invalid input.")
        return None
    idx = int(raw)
    if idx == 0:
        return None
    if not (1 <= idx <= len(items)):
        print("Out of range.")
        return None
    return items[idx - 1]


def pick_multi(items: list[str], prompt: str = "Select", default: list[int] | None = None) -> list[str]:
    """Select multiple items using comma-separated numbers and/or ranges (e.g. 1,3,5-7).
    default is a list of 1-based indices pre-selected when the user presses Enter."""
    if not items:
        print("  (none)")
        return []
    for i, item in enumerate(items):
        mark = "*" if default and (i + 1) in default else " "
        print(f"  [{mark}{i + 1}] {item}")
    default_str = ",".join(str(d) for d in default) if default else ""
    hint = f", Enter for default ({default_str})" if default_str else ""
    raw = input(f"\n{prompt} (e.g. 1,3,5-7{hint} — or 0 to cancel): ").strip()
    if raw == "0":
        return []
    if not raw and default:
        return [items[i - 1] for i in sorted(default)]
    indices: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            bounds = part.split("-", 1)
            if not all(b.isdigit() for b in bounds):
                print(f"  Invalid range: {part}")
                return []
            lo, hi = int(bounds[0]), int(bounds[1])
            if not (1 <= lo <= hi <= len(items)):
                print(f"  Range out of bounds: {part}")
                return []
            indices.update(range(lo, hi + 1))
        elif part.isdigit():
            idx = int(part)
            if not (1 <= idx <= len(items)):
                print(f"  Out of range: {idx}")
                return []
            indices.add(idx)
        else:
            print(f"  Invalid input: {part}")
            return []
    return [items[i - 1] for i in sorted(indices)]


def confirm(msg: str) -> bool:
    return input(f"{msg} [y/N]: ").strip().lower() == "y"


# --- actions ---

def action_list() -> None:
    result = ollama("list")
    if result.returncode != 0:
        print(f"Error: {result.stderr.strip()}")
        return
    print()
    print(result.stdout.strip())


def action_pull() -> None:
    name = input("\nModel name to pull (e.g. qwen3:8b): ").strip()
    if not name:
        return
    print(f"\nPulling {name}...")
    ollama("pull", name, stream=True)
    action_create_variants(name)
    print("\nSyncing opencode config...")
    action_sync_opencode()


def action_remove() -> None:
    try:
        all_models = fetch_all_models()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return

    print("\nAll models:")
    targets = pick_multi(all_models, "Remove models")
    if not targets:
        return
    print(f"\n  About to remove {len(targets)} model(s):")
    for t in targets:
        print(f"    - {t}")
    if not confirm("  Confirm?"):
        return
    for t in targets:
        result = ollama("rm", t)
        if result.returncode == 0:
            print(f"  Removed {t}.")
        else:
            print(f"  Failed {t}: {result.stderr.strip()}")
    print("\nSyncing opencode config...")
    action_sync_opencode()


def action_search() -> None:
    query = input("\nSearch query: ").strip()
    if not query:
        return

    print("Searching...")
    try:
        models = scrape_search(query)
    except Exception as e:
        print(f"  Search failed: {e}")
        return

    if not models:
        print("  No results.")
        return

    print(f"\nResults for '{query}':")
    model = pick(models, "Select model")
    if not model:
        return

    print(f"\nFetching tags for {model}...")
    try:
        tag_entries = scrape_tags(model)
    except Exception as e:
        print(f"  Failed to fetch tags: {e}")
        return

    if not tag_entries:
        print("  No tags found.")
        return

    tags = [t for t, _ in tag_entries]
    labeled = [f"{t:<40} {s:>8}" for t, s in tag_entries]

    print(f"\nTags for {model}:")
    choice = pick(labeled, "Pull tag")
    if not choice:
        return

    full_name = tags[labeled.index(choice)]
    print(f"\nPulling {full_name}...")
    ollama("pull", full_name, stream=True)
    action_create_variants(full_name)
    print("\nSyncing opencode config...")
    action_sync_opencode()


def action_create_variants(base: str) -> None:
    try:
        all_models = fetch_all_models()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return

    _, _, has_variants = split_base_and_variants(all_models)

    action = "Recreating" if base in has_variants else "Creating"
    print(f"\n{action} context variants for {base}.")
    print("\nSelect context sizes to create:")
    chosen_labels = pick_multi(CONTEXT_LABELS, "Variants", default=[1, 2, 3])
    if not chosen_labels:
        return

    for label in chosen_labels:
        idx = CONTEXT_LABELS.index(label)
        _create_variant(base, CONTEXT_SIZES[idx], label)

    if is_wsl():
        print(WSL_MEMORY_HINT)


def action_create_variants_menu() -> None:
    try:
        all_models = fetch_all_models()
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return

    base_models, _, has_variants = split_base_and_variants(all_models)

    print("\nBase models:")
    labeled = [
        f"{m}  [has variants]" if m in has_variants else m
        for m in base_models
    ]
    choice = pick(labeled, "Create variants for")
    if not choice:
        return

    base = choice.replace("  [has variants]", "")
    action_create_variants(base)
    print("\nSyncing opencode config...")
    action_sync_opencode()


def _create_variant(base: str, ctx: int, label: str) -> None:
    name = f"{base}-{label}"
    cmd = (
        f"printf 'FROM {base}\\nPARAMETER num_ctx {ctx}\\n' > /tmp/Modelfile"
        f" && ollama create '{name}' -f /tmp/Modelfile"
    )
    result = ollama_bash(cmd)
    if result.returncode == 0:
        print(f"  created {name}")
    else:
        print(f"  failed {name}: {result.stderr.strip()}")


def action_sync_opencode() -> None:
    try:
        all_models = fetch_all_models()
    except subprocess.CalledProcessError as e:
        print(f"  Error fetching models: {e.stderr}")
        return

    if OPENCODE_CONFIG.exists():
        config = json.loads(OPENCODE_CONFIG.read_text(encoding="utf-8"))
    else:
        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "ollama": {
                    "npm": "@ai-sdk/openai-compatible",
                    "options": {"baseURL": "http://localhost:11434/v1"},
                    "models": {},
                }
            },
        }
        OPENCODE_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    local = set(all_models)
    models_cfg = config.setdefault("provider", {}).setdefault("ollama", {}).setdefault("models", {})
    registered = set(models_cfg.keys())

    to_add = [m for m in all_models if m not in registered]
    to_remove = [m for m in registered if m not in local]

    if not to_add and not to_remove:
        print("\n  opencode config is already in sync.")
        return

    for m in to_remove:
        del models_cfg[m]
        print(f"  - {m}")
    for m in to_add:
        models_cfg[m] = {"tools": True}
        print(f"  + {m}")

    OPENCODE_CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"\n  Saved to {OPENCODE_CONFIG}")


# --- main loop ---

MENU = [
    ("List models",                        action_list),
    ("Search & pull a model",              action_search),
    ("Pull a model by name",               action_pull),
    ("Remove a model",                     action_remove),
    ("Create context variants",            action_create_variants_menu),
    ("Sync missing models to opencode",    action_sync_opencode),
]


def main() -> None:
    while True:
        print("\n=== Ollama Manager ===")
        for i, (label, _) in enumerate(MENU):
            print(f"  [{i + 1}] {label}")
        print("  [0] Exit")

        raw = input("\nChoice: ").strip()
        if raw == "0":
            break
        if not raw.isdigit() or not (1 <= int(raw) <= len(MENU)):
            print("Invalid choice.")
            continue

        _, action = MENU[int(raw) - 1]
        action()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
        sys.exit(0)
