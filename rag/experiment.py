"""Load, merge, and compare experiment configs.

An experiment config names a parent and lists only what it changes. That makes
"only one variable moved" a property of the file rather than something you
have to remember - the other settings are not present to be edited by mistake.

Configs are TOML because tomllib is in the standard library and because a
config file's job is partly to record why an experiment exists, which needs
comments.

Also computes the corpus hash. The corpus is the one input no config mentions
and every metric depends on, so a run records what it was actually measuring.
"""

import hashlib
import tomllib
from pathlib import Path

CONFIG_DIR = Path("configs")

# Filled in for any key a config omits, so a run always records the full
# picture rather than leaving the reader to guess at defaults.
DEFAULTS = {
    "chunking": {"strategy": "fixed_size", "size": 512, "overlap": 128},
    "embedding": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
    "retrieval": {"k": 5},
    "generation": {"enabled": True, "model": "claude-haiku-4-5"},
    "judging": {"enabled": True, "model": "claude-sonnet-5"},
}


def _merge(base, override):
    """Deep-merge override onto base, one level of tables deep."""
    merged = {key: dict(value) if isinstance(value, dict) else value
              for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _read(name):
    path = name if str(name).endswith(".toml") else CONFIG_DIR / f"{name}.toml"
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no config at {path}")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_config(name):
    """Return (merged config, parent config or None).

    Follows `extends` chains, applying defaults at the bottom. The parent is
    returned as well so the caller can report exactly what this experiment
    changed.
    """
    chain = []
    seen = set()
    current = name

    while current is not None:
        if current in seen:
            raise ValueError(f"config {current!r} extends itself in a loop")
        seen.add(current)
        raw = _read(current)
        chain.append(raw)
        current = raw.get("extends")

    # Oldest ancestor first, so later files win.
    merged = _merge(DEFAULTS, {})
    for raw in reversed(chain):
        merged = _merge(merged, raw)
    merged.setdefault("name", str(name))

    parent = None
    if len(chain) > 1:
        parent = _merge(DEFAULTS, {})
        for raw in reversed(chain[1:]):
            parent = _merge(parent, raw)

    return merged, parent


# Keys that describe presentation rather than the experiment. Changing these
# alongside a real variable is not a second variable.
METADATA_KEYS = {"name", "note", "extends"}

# Switching a stage off changes which metrics exist, not how the pipeline is
# configured - there is nothing left to misattribute a result to. So these do
# not count toward the one-variable rule.
NON_VARIABLE_KEYS = {"generation.enabled", "judging.enabled"}


def diff_configs(parent, child, count_only_variables=False):
    """List (dotted key, old, new) for every setting that differs.

    With count_only_variables, the stage on/off switches are left out, which
    is what the one-variable rule is checked against.
    """
    changes = []
    for section in sorted(set(parent) | set(child)):
        if section in METADATA_KEYS:
            continue
        old, new = parent.get(section), child.get(section)
        if isinstance(old, dict) or isinstance(new, dict):
            old, new = old or {}, new or {}
            for key in sorted(set(old) | set(new)):
                dotted = f"{section}.{key}"
                if count_only_variables and dotted in NON_VARIABLE_KEYS:
                    continue
                if old.get(key) != new.get(key):
                    changes.append((dotted, old.get(key), new.get(key)))
        elif old != new:
            changes.append((section, old, new))
    return changes


def corpus_hash(documents):
    """A short digest of the corpus a run measured.

    Covers doc_id, revision, and text of every document, so re-fetching the
    corpus produces a different hash and the mismatch is visible rather than
    quietly making two runs incomparable.
    """
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda d: d.doc_id):
        digest.update(document.doc_id.encode())
        digest.update(str(document.revision_id).encode())
        digest.update(hashlib.sha256(document.text.encode()).digest())
    return digest.hexdigest()[:12]


def dump_toml(config):
    """Serialise a merged config back to TOML.

    tomllib only reads, and the shape here is small and known - scalars at the
    top, one level of tables below - so writing it out directly avoids a
    dependency. Anything outside that shape raises rather than being silently
    mangled.
    """

    def scalar(value):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
        raise TypeError(f"cannot serialise {type(value).__name__} to TOML: {value!r}")

    lines = []
    for key, value in config.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {scalar(value)}")
    for key, value in config.items():
        if isinstance(value, dict):
            lines.append(f"\n[{key}]")
            for subkey, subvalue in value.items():
                lines.append(f"{subkey} = {scalar(subvalue)}")
    return "\n".join(lines) + "\n"
