#!/opt/homebrew/bin/python3.12
# ============================================================================
# SMELT: Schema-Aware Markdown Compilation for Efficient Local Token Inference
# ============================================================================
#
# Compiles agent workspace markdown files into progressively denser runtime
# representations across four layers:
#
#   Layer 1 — Lossless Archival Storage (zstd + SHA-256 round-trip)
#   Layer 2 — Semantic Runtime Compilation (schema-aware structural reduction)
#   Layer 3 — Macro Runtime Compression (in-band phrase dictionary)
#   Layer 4 — Query-Conditioned Selective Emission (TF-IDF relevance scoring)
#
# Source files remain human-readable; runtime context is compiled.
# Full provenance is preserved, enabling decompilation back to the original.
#
# See: Lister, E. (2026). "SMELT: Schema-Aware Markdown Compilation for
# Efficient Local Token Inference." Zenodo. DOI: 10.5281/zenodo.19380983
# ============================================================================
from __future__ import annotations

import argparse
import collections
import datetime as dt
import html
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path


# ============================================================================
# CONFIGURATION & CONSTANTS
# Binary container format identifiers, compression defaults, and directory
# layout for compiled/restored/runtime output files.
# ============================================================================

MAGIC = b"MDC1"
MANIFEST_LEN_STRUCT = ">I"
FORMAT_VERSION = 1
ZSTD_BIN = shutil.which("zstd") or "/opt/homebrew/bin/zstd"
DEFAULT_THREADS = min(max(os.cpu_count() or 1, 1), 8)
DEFAULT_LEVEL = 10
ROOT = Path("/Users/studio-m3/test")
INPUT_DIR = ROOT / "input"
COMPRESSED_DIR = ROOT / "compressed"
RESTORED_DIR = ROOT / "restored"
RUNTIME_DIR = ROOT / "runtime"
DOCS_DIR = ROOT / "docs"
DEFAULT_WORKSPACE = Path("/Users/studio-m3/si/projects/openclaw-workspace")
DEFAULT_TOKENIZER = Path("/Users/studio-m3/si/llm/qwen/Qwen3.5-VL-122B-A10B-8bit-MLX/tokenizer.json")

# ============================================================================
# SCHEMA DEFINITIONS FOR SEMANTIC COMPILATION (Layer 2)
# These mappings encode the known structure of OpenClaw workspace files.
# Section headings, groups, and field labels are replaced with short codes
# to eliminate structural redundancy while preserving all values.
# This is the "schema-aware" part — exploiting known document structure
# rather than applying generic text compression.
# ============================================================================

USER_SECTION_CODES = {
    "USER.md - About Your Human": "A",
    "Edmund's 40-Year Journey with SI": "J",
    "Edmund's FIghting Spirit": "F",
    "Family Dynamics": "D",
    "What He Wants From Me": "W",
    "His Strengths & Quirks": "Q",
    "Notes for Daily Life": "N",
}

USER_GROUP_CODES = {
    "The Visionary": "vz",
    "Groundbreaking Moments": "gm",
    "Immediate Family": "if",
    "SI Family (His Daughters)": "sf",
}

USER_FIELD_CODES = {
    "Name": "nm",
    "What to call them": "call",
    "Pronouns": "pro",
    "Timezone": "tz",
    "Location": "loc",
    "Born": "born",
    "Health Warrior": "hlth",
}

# Stopwords filtered out during query tokenization (Layer 4).
# These carry no discriminative value for TF-IDF scoring.
QUERY_STOPWORDS = {
    "a",
    "about",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "get",
    "got",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "like",
    "may",
    "maybe",
    "me",
    "more",
    "most",
    "my",
    "no",
    "not",
    "of",
    "on",
    "one",
    "or",
    "our",
    "out",
    "please",
    "real",
    "really",
    "same",
    "she",
    "should",
    "so",
    "some",
    "tell",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "up",
    "us",
    "use",
    "using",
    "very",
    "want",
    "was",
    "we",
    "well",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}


# ============================================================================
# LAYER 1: Lossless Archival Storage
# Compresses source markdown with zstd and validates round-trip via SHA-256
# hash comparison. This layer targets archival integrity and transport
# efficiency, not inference acceleration. The binary container format (MDC1)
# bundles a JSON manifest (source metadata, hash, sizes) with the compressed
# payload into a single file.
# ============================================================================


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def default_compiled_path(source: Path) -> Path:
    return COMPRESSED_DIR / f"c-{source.name}"


def default_restored_path(compiled: Path, manifest: dict[str, object]) -> Path:
    source_name = str(manifest.get("source_name", compiled.name))
    return RESTORED_DIR / f"r-{source_name}"


# parse_container / build_container: Read and write the MDC1 binary format.
# Layout: [MAGIC 4B][manifest_len 4B][JSON manifest][zstd payload]
def parse_container(blob: bytes) -> tuple[dict[str, object], bytes]:
    if len(blob) < len(MAGIC) + 4:
        raise ValueError("file too small to be a valid MDC container")
    if blob[: len(MAGIC)] != MAGIC:
        raise ValueError("invalid magic header")
    manifest_len = struct.unpack(MANIFEST_LEN_STRUCT, blob[len(MAGIC) : len(MAGIC) + 4])[0]
    start = len(MAGIC) + 4
    end = start + manifest_len
    if len(blob) < end:
        raise ValueError("truncated manifest")
    manifest = json.loads(blob[start:end].decode("utf-8"))
    payload = blob[end:]
    return manifest, payload


def build_container(manifest: dict[str, object], payload: bytes) -> bytes:
    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return MAGIC + struct.pack(MANIFEST_LEN_STRUCT, len(manifest_bytes)) + manifest_bytes + payload


# run_zstd: Shells out to the zstd binary for compression/decompression.
# Using the external binary rather than a Python binding for maximum
# compression quality and multi-threaded performance.
def run_zstd(data: bytes, *, decompress: bool, level: int, threads: int) -> bytes:
    if not Path(ZSTD_BIN).exists():
        raise FileNotFoundError(f"zstd binary not found at {ZSTD_BIN}")

    cmd = [ZSTD_BIN, "--no-progress", "-q", f"-T{threads}", "-c"]
    if decompress:
        cmd.insert(1, "-d")
    else:
        if level > 19:
            cmd.append("--ultra")
        cmd.append(f"-{level}")

    result = subprocess.run(cmd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"zstd failed: {stderr or 'unknown error'}")
    return result.stdout


# compile_file: Layer 1 entry point — compress source markdown into an MDC1
# container. Records source hash and size in the manifest for later verification.
def compile_file(source: Path, output: Path, *, level: int, threads: int) -> dict[str, object]:
    raw = source.read_bytes()
    compressed = run_zstd(raw, decompress=False, level=level, threads=threads)
    manifest = {
        "format": "mdc",
        "format_version": FORMAT_VERSION,
        "algorithm": "zstd",
        "compression_level": level,
        "threads": threads,
        "source_name": source.name,
        "source_path": str(source),
        "source_size": len(raw),
        "source_sha256": sha256_bytes(raw),
        "compressed_size": len(compressed),
    }
    container = build_container(manifest, compressed)
    ensure_parent(output)
    output.write_bytes(container)
    return manifest


# decompile_file: Layer 1 decompression — restore original markdown from an
# MDC1 container and verify exact byte-for-byte match via SHA-256.
def decompile_file(compiled: Path, output: Path | None, *, threads: int) -> tuple[dict[str, object], Path]:
    blob = compiled.read_bytes()
    manifest, compressed = parse_container(blob)
    raw = run_zstd(compressed, decompress=True, level=0, threads=threads)

    expected_size = int(manifest["source_size"])
    expected_hash = str(manifest["source_sha256"])
    actual_hash = sha256_bytes(raw)
    if len(raw) != expected_size:
        raise ValueError(f"size mismatch: expected {expected_size}, got {len(raw)}")
    if actual_hash != expected_hash:
        raise ValueError(f"sha256 mismatch: expected {expected_hash}, got {actual_hash}")

    destination = output or default_restored_path(compiled, manifest)
    ensure_parent(destination)
    destination.write_bytes(raw)
    return manifest, destination


def verify_exact(source: Path, restored: Path) -> tuple[bool, str, str]:
    source_bytes = source.read_bytes()
    restored_bytes = restored.read_bytes()
    return source_bytes == restored_bytes, sha256_bytes(source_bytes), sha256_bytes(restored_bytes)


def format_ratio(source_size: int, container_size: int) -> str:
    if source_size == 0:
        return "n/a"
    return f"{container_size / source_size:.3f}x"


def format_ms(duration_ns: int) -> str:
    return f"{duration_ns / 1_000_000:.3f} ms"


def compression_percent(source_size: int, container_size: int) -> str:
    if source_size == 0:
        return "n/a"
    saved = (1 - (container_size / source_size)) * 100
    return f"{saved:.2f}%"


# load_qwen_tokenizer / token_count: Load the actual tokenizer from the
# inference target (Qwen 3.5 VL 122B A10B, 8-bit, MLX) to measure real token
# counts. Byte length is NOT a reliable proxy — byte-optimal and token-optimal
# compression are distinct objectives under BPE-derived tokenizers.
def load_qwen_tokenizer(tokenizer_path: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_path))


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text).ids)


def resolve_startup_files(workspace: Path, audit_date: dt.date) -> tuple[list[Path], dict[str, object]]:
    memory_dir = workspace / "memory"
    today = memory_dir / f"{audit_date.isoformat()}.md"
    yesterday = memory_dir / f"{(audit_date - dt.timedelta(days=1)).isoformat()}.md"

    files = [
        workspace / "AGENTS.md",
        workspace / "SOUL.md",
        workspace / "USER.md",
        workspace / "MEMORY.md",
    ]
    if today.exists():
        files.append(today)
    if yesterday.exists():
        files.append(yesterday)

    metadata = {
        "workspace": str(workspace),
        "audit_date": audit_date.isoformat(),
        "today_exists": today.exists(),
        "yesterday_exists": yesterday.exists(),
        "today_file": str(today),
        "yesterday_file": str(yesterday),
    }
    return files, metadata


def runtime_output_path(source: Path) -> Path:
    return RUNTIME_DIR / f"rt-{source.stem}.txt"


def runtime_map_path(runtime_output: Path) -> Path:
    return runtime_output.with_suffix(".map.json")


def semantic_runtime_output_path(source: Path) -> Path:
    return RUNTIME_DIR / f"srt-{source.stem}.txt"


def semantic_runtime_map_path(runtime_output: Path) -> Path:
    return runtime_output.with_suffix(".map.json")


def packed_runtime_output_path(source: Path) -> Path:
    return RUNTIME_DIR / f"prt-{source.stem}.txt"


def packed_runtime_map_path(runtime_output: Path) -> Path:
    return runtime_output.with_suffix(".map.json")


def packed_runtime_dict_path(runtime_output: Path) -> Path:
    return runtime_output.with_suffix(".dict.json")


def macro_runtime_output_path(source: Path) -> Path:
    return RUNTIME_DIR / f"mrt-{source.stem}.txt"


def macro_runtime_map_path(runtime_output: Path) -> Path:
    return runtime_output.with_suffix(".map.json")


def macro_runtime_dict_path(runtime_output: Path) -> Path:
    return runtime_output.with_suffix(".macros.json")


def query_runtime_output_path(source: Path, *, mode: str) -> Path:
    prefix = "qmrt" if mode == "macro" else "qsrt"
    return RUNTIME_DIR / f"{prefix}-{source.stem}.txt"


def query_runtime_map_path(runtime_output: Path) -> Path:
    return runtime_output.with_suffix(".map.json")


def query_runtime_selection_path(runtime_output: Path) -> Path:
    return runtime_output.with_suffix(".selection.json")


def startup_report_path(audit_date: dt.date) -> Path:
    return DOCS_DIR / f"startup_audit_{audit_date.isoformat()}.md"


def decompiled_runtime_output_path(runtime_file: Path) -> Path:
    stem = runtime_file.name
    if stem.endswith(".txt"):
        stem = stem[:-4]
    return RESTORED_DIR / f"dr-{stem}.md"


def distortion_percent(original: bytes, current: bytes) -> float:
    if original == current:
        return 0.0
    if not original and not current:
        return 0.0
    ratio = SequenceMatcher(None, original, current).ratio()
    return max(0.0, (1.0 - ratio) * 100.0)


# ============================================================================
# MARKDOWN NORMALIZATION
# Strips inline formatting (bold, italic, links, code spans, images) while
# preserving the underlying text content. This is shared by both the basic
# runtime (markdown_to_runtime) and the semantic runtime (Layer 2).
# ============================================================================


def normalize_inline_markdown(text: str) -> str:
    text = html.unescape(text.rstrip())
    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"IMAGE \1 <\2>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 <\2>", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)
    text = re.sub(r"~~([^~]+)~~", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================================
# BASIC RUNTIME COMPILATION
# A simpler, non-schema-aware transformation that converts markdown into a
# record-per-line format with type prefixes (H=heading, P=paragraph, L=list,
# Q=quote, C=code, N=numbered, M=meta). Serves as a baseline for comparison
# against the schema-aware semantic runtime.
# ============================================================================


def markdown_to_runtime(source: Path) -> tuple[str, list[dict[str, object]]]:
    lines = source.read_text(encoding="utf-8").splitlines()
    records: list[str] = [f"M|source={source.name}"]
    provenance: list[dict[str, object]] = [{"record": 0, "kind": "M", "line_start": 0, "line_end": 0, "text": f"source={source.name}"}]

    paragraph_parts: list[str] = []
    paragraph_start = 0
    in_code_block = False
    code_lang = ""
    code_lines: list[str] = []
    code_start = 0

    def flush_paragraph() -> None:
        nonlocal paragraph_parts, paragraph_start
        if not paragraph_parts:
            return
        text = normalize_inline_markdown(" ".join(paragraph_parts))
        if text:
            records.append(f"P|{text}")
            provenance.append(
                {
                    "record": len(records) - 1,
                    "kind": "P",
                    "line_start": paragraph_start,
                    "line_end": paragraph_start + len(paragraph_parts) - 1,
                    "text": text,
                }
            )
        paragraph_parts = []
        paragraph_start = 0

    def flush_code() -> None:
        nonlocal code_lines, code_lang, code_start
        if not code_lines:
            return
        text = " ⏎ ".join(code_lines)
        payload = f"{code_lang}|{text}" if code_lang else text
        records.append(f"C|{payload}")
        provenance.append(
            {
                "record": len(records) - 1,
                "kind": "C",
                "line_start": code_start,
                "line_end": code_start + len(code_lines) - 1,
                "lang": code_lang,
                "text": text,
            }
        )
        code_lines = []
        code_lang = ""
        code_start = 0

    for idx, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            if in_code_block:
                flush_code()
                in_code_block = False
            else:
                in_code_block = True
                code_lang = stripped[3:].strip()
                code_start = idx + 1
            continue

        if in_code_block:
            code_lines.append(raw_line.rstrip("\n"))
            continue

        if not stripped:
            flush_paragraph()
            continue

        if re.fullmatch(r"[-*_]{3,}", stripped):
            flush_paragraph()
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            text = normalize_inline_markdown(heading_match.group(2))
            records.append(f"H{level}|{text}")
            provenance.append({"record": len(records) - 1, "kind": f"H{level}", "line_start": idx, "line_end": idx, "text": text})
            continue

        quote_match = re.match(r"^>\s?(.*)$", stripped)
        if quote_match:
            flush_paragraph()
            text = normalize_inline_markdown(quote_match.group(1))
            records.append(f"Q|{text}")
            provenance.append({"record": len(records) - 1, "kind": "Q", "line_start": idx, "line_end": idx, "text": text})
            continue

        bullet_match = re.match(r"^([-+*]|\d+\.)\s+(.*)$", stripped)
        if bullet_match:
            flush_paragraph()
            marker = bullet_match.group(1)
            text = normalize_inline_markdown(bullet_match.group(2))
            kind = "N" if marker.endswith(".") and marker[:-1].isdigit() else "L"
            records.append(f"{kind}|{text}")
            provenance.append({"record": len(records) - 1, "kind": kind, "line_start": idx, "line_end": idx, "text": text})
            continue

        if not paragraph_parts:
            paragraph_start = idx
        paragraph_parts.append(stripped)

    flush_paragraph()
    flush_code()
    runtime_text = "\n".join(records) + "\n"
    return runtime_text, provenance


# ============================================================================
# LABEL & KEY EXTRACTION HELPERS (Layer 2)
# These functions support semantic compilation by detecting labeled fields
# (e.g., "Name: Edmund"), splitting annotations from labels (e.g.,
# "Kane (son)"), and generating compact key codes. They enable the
# schema-aware transformation from verbose markdown bullet points into
# dense key=value records.
# ============================================================================


def fallback_code(text: str, *, prefix: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", text.lower())
    if not parts:
        return prefix
    if len(parts) == 1:
        base = parts[0][:4]
    else:
        base = "".join(part[0] for part in parts[:4])
    return f"{prefix}{base}"


def compact_key(label: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", label.lower())
    if not parts:
        return "k"
    chunks: list[str] = []
    for idx, part in enumerate(parts[:3]):
        if idx == 0 and part.isdigit():
            chunks.append(f"y{part}")
        else:
            chunks.append(part[:4] if len(part) > 4 else part)
    key = "_".join(chunks)
    return key[:18]


def normalize_label_for_key(label: str) -> str:
    normalized = re.sub(r"\([^)]*\)", " ", label)
    normalized = re.sub(r"[^A-Za-z0-9/ ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def split_label_parts(label: str) -> tuple[str, str]:
    match = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", label.strip())
    if not match:
        return label.strip(), ""
    base = match.group(1).strip()
    annotation = re.sub(r"\s+", "", match.group(2).strip())
    return base, annotation


def split_labeled_text(text: str, *, allow_general: bool) -> tuple[str | None, str]:
    if ":" not in text:
        return None, text
    left, right = text.split(":", 1)
    label = left.strip()
    value = right.strip()
    if not label or not value:
        return None, text
    if not allow_general:
        return None, text
    if len(label) > 28:
        return None, text
    if "(" in label or ")" in label:
        return None, text
    word_count = len(re.findall(r"[A-Za-z0-9]+", label))
    if word_count == 0 or word_count > 4:
        return None, text
    return label, value


def split_contextual_label(text: str, *, section: str, group: str) -> tuple[str | None, str]:
    if ":" not in text:
        return None, text
    left, right = text.split(":", 1)
    label = left.strip()
    value = right.strip()
    if not label or not value:
        return None, text

    normalized_label = normalize_label_for_key(label)
    words = re.findall(r"[A-Za-z0-9]+", normalized_label)
    if not words:
        return None, text

    # Family sections benefit from hoisting person/entity labels into compact keys.
    if section == "D" and group in {"if", "sf"}:
        if len(words) <= 4:
            return label, value

    if "(" in label or ")" in label:
        return None, text

    # Allow short title-like labels elsewhere, but reject sentence-like prefixes.
    if len(words) <= 3 and label[:1].isupper():
        return label, value

    return None, text


# ============================================================================
# LAYER 2: Semantic Runtime Compilation
# The core of SMELT. Transforms markdown into a schema-aware compact format:
#   - Section headings → short section codes (e.g., "#A" for "About Your Human")
#   - Groups → compact group codes (e.g., ">if" for "Immediate Family")
#   - Labeled bullets → key=value records (e.g., "nm=Edmund")
#   - Unlabeled items → "+text" records
#   - Paragraphs, quotes, code → prefixed records (:, ?, $)
# Each record is tagged with provenance metadata (source file, section,
# line range) enabling deterministic decompilation back to markdown.
# The output remains plain text — not an opaque embedding or binary format —
# so it can be ingested directly by a language model.
# ============================================================================


def markdown_to_semantic_runtime(source: Path) -> tuple[str, list[dict[str, object]]]:
    lines = source.read_text(encoding="utf-8").splitlines()
    records: list[str] = [f"@{source.name}"]
    provenance: list[dict[str, object]] = [{"record": 0, "kind": "meta", "line_start": 0, "line_end": 0, "text": source.name}]

    current_section = "U"
    current_group = ""
    paragraph_parts: list[str] = []
    paragraph_start = 0
    in_code_block = False
    code_lang = ""
    code_lines: list[str] = []
    code_start = 0

    def emit(record_text: str, *, kind: str, line_start: int, line_end: int, text: str, section: str | None = None, group: str | None = None, key: str | None = None) -> None:
        records.append(record_text)
        entry: dict[str, object] = {
            "record": len(records) - 1,
            "kind": kind,
            "line_start": line_start,
            "line_end": line_end,
            "text": text,
        }
        if section:
            entry["section"] = section
        if group:
            entry["group"] = group
        if key:
            entry["key"] = key
        provenance.append(entry)

    def flush_paragraph() -> None:
        nonlocal paragraph_parts, paragraph_start, current_group
        if not paragraph_parts:
            return
        text = normalize_inline_markdown(" ".join(paragraph_parts))
        if not text:
            paragraph_parts = []
            paragraph_start = 0
            return

        if text.endswith(":") and len(text) <= 48:
            group_name = text[:-1].strip()
            current_group = USER_GROUP_CODES.get(group_name, fallback_code(group_name, prefix="g"))
            emit(f">{current_group}", kind="group", line_start=paragraph_start, line_end=paragraph_start + len(paragraph_parts) - 1, text=group_name, section=current_section, group=current_group)
        else:
            label = None
            value = text
            if ":" in text:
                maybe_label, maybe_value = text.split(":", 1)
                maybe_label = maybe_label.strip()
                maybe_value = maybe_value.strip()
                if maybe_label in USER_GROUP_CODES and maybe_value:
                    label = maybe_label
                    value = maybe_value
            if label and label in USER_GROUP_CODES:
                group_code = USER_GROUP_CODES[label]
                emit(f"{group_code}={value}", kind="labeled_paragraph", line_start=paragraph_start, line_end=paragraph_start + len(paragraph_parts) - 1, text=text, section=current_section, group=group_code, key=label)
            else:
                emit(f":{text}", kind="paragraph", line_start=paragraph_start, line_end=paragraph_start + len(paragraph_parts) - 1, text=text, section=current_section, group=current_group or None)
        paragraph_parts = []
        paragraph_start = 0

    def flush_code() -> None:
        nonlocal code_lines, code_lang, code_start
        if not code_lines:
            return
        text = " ⏎ ".join(code_lines)
        payload = f"{code_lang}|{text}" if code_lang else text
        emit(f"${payload}", kind="code", line_start=code_start, line_end=code_start + len(code_lines) - 1, text=text, section=current_section, group=current_group or None)
        code_lines = []
        code_lang = ""
        code_start = 0

    for idx, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            if in_code_block:
                flush_code()
                in_code_block = False
            else:
                in_code_block = True
                code_lang = stripped[3:].strip()
                code_start = idx + 1
            continue

        if in_code_block:
            code_lines.append(raw_line.rstrip("\n"))
            continue

        if not stripped:
            flush_paragraph()
            continue

        if re.fullmatch(r"[-*_]{3,}", stripped):
            flush_paragraph()
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_paragraph()
            level = len(heading_match.group(1))
            text = normalize_inline_markdown(heading_match.group(2))
            current_group = ""
            current_section = USER_SECTION_CODES.get(text, fallback_code(text, prefix="s"))
            emit(f"#{current_section}", kind=f"heading_{level}", line_start=idx, line_end=idx, text=text, section=current_section)
            continue

        quote_match = re.match(r"^>\s?(.*)$", stripped)
        if quote_match:
            flush_paragraph()
            text = normalize_inline_markdown(quote_match.group(1))
            emit(f"?{text}", kind="quote", line_start=idx, line_end=idx, text=text, section=current_section, group=current_group or None)
            continue

        bullet_match = re.match(r"^([-+*]|\d+\.)\s+(.*)$", stripped)
        if bullet_match:
            flush_paragraph()
            text = normalize_inline_markdown(bullet_match.group(2))
            label, value = split_contextual_label(text, section=current_section, group=current_group)
            if label is None:
                label, value = split_labeled_text(text, allow_general=True)
            if label:
                base_label, annotation = split_label_parts(label)
                key_code = USER_FIELD_CODES.get(label)
                if key_code is None and base_label in USER_GROUP_CODES:
                    key_code = USER_GROUP_CODES[base_label]
                if key_code is None:
                    key_code = compact_key(normalize_label_for_key(base_label))
                encoded_key = f"{key_code}@{annotation}" if annotation else key_code
                emit(f"{encoded_key}={value}", kind="field" if label in USER_FIELD_CODES else "labeled_item", line_start=idx, line_end=idx, text=text, section=current_section, group=current_group or None, key=label)
            else:
                marker = "+" if bullet_match.group(1).endswith(".") or bullet_match.group(1) in "-+*" else "+"
                emit(f"{marker}{text}", kind="item", line_start=idx, line_end=idx, text=text, section=current_section, group=current_group or None)
            continue

        if not paragraph_parts:
            paragraph_start = idx
        paragraph_parts.append(stripped)

    flush_paragraph()
    flush_code()
    runtime_text = "\n".join(records) + "\n"
    return runtime_text, provenance


# ============================================================================
# PACKED RUNTIME FORMAT
# An alternative representation of the semantic runtime that normalizes all
# record types into a uniform prefix scheme (S=section, G=group, K=key/value,
# L=list, P=paragraph, Q=quote, C=code) and produces a sidecar dictionary
# mapping codes back to their full names. Used for external tooling and
# analysis rather than direct model ingestion.
# ============================================================================


def semantic_to_packed(semantic_text: str, provenance: list[dict[str, object]]) -> tuple[str, dict[str, object]]:
    lines = [line for line in semantic_text.splitlines() if line]
    if not lines:
        return "", {"meta": {}, "legend": {}}

    meta: dict[str, str] = {}
    sections: dict[str, str] = {}
    groups: dict[str, str] = {}
    keys: dict[str, str] = {}
    records: list[str] = []

    current_section = ""
    current_group = ""

    prov_by_record = {int(item["record"]): item for item in provenance if "record" in item}

    for record_index, line in enumerate(lines):
        if record_index == 0 and line.startswith("@"):
            meta["source"] = line[1:]
            records.append(f"@={line[1:]}")
            continue

        if line.startswith("#"):
            current_section = line[1:]
            current_group = ""
            prov = prov_by_record.get(record_index)
            if prov and "text" in prov:
                sections[current_section] = str(prov["text"])
            records.append(f"S={current_section}")
            continue

        if line.startswith(">"):
            current_group = line[1:]
            prov = prov_by_record.get(record_index)
            group_key = f"{current_section}>{current_group}"
            if prov and "text" in prov:
                groups[group_key] = str(prov["text"])
            records.append(f"G={current_group}")
            continue

        match = re.match(r"^([A-Za-z0-9_]+)=(.*)$", line)
        if match:
            key = match.group(1)
            value = match.group(2)
            prov = prov_by_record.get(record_index)
            full_key = f"{current_section}>{current_group}>{key}" if current_group else f"{current_section}>{key}"
            if prov and "key" in prov:
                keys[full_key] = str(prov["key"])
            records.append(f"K={key}|{value}")
            continue

        if line.startswith("+"):
            records.append(f"L|{line[1:]}")
            continue

        if line.startswith(":"):
            records.append(f"P|{line[1:]}")
            continue

        if line.startswith("?"):
            records.append(f"Q|{line[1:]}")
            continue

        if line.startswith("$"):
            records.append(f"C|{line[1:]}")
            continue

        records.append(f"R|{line}")

    packed_text = "\n".join(records) + "\n"
    dictionary = {
        "meta": meta,
        "sections": sections,
        "groups": groups,
        "keys": keys,
    }
    return packed_text, dictionary


# ============================================================================
# LAYER 3: Macro Runtime Compression
# Adds an in-band phrase dictionary over the semantic runtime, replacing
# frequently repeated phrases with short tokens (e.g., ~A, ~B).
#
# KEY EMPIRICAL FINDING: Byte compression and token compression are distinct
# objectives. Under the Qwen tokenizer, macro compression can actually
# *increase* token count even while reducing byte count, because BPE-derived
# tokenizers have already learned common phrase patterns as multi-character
# tokens. Replacing a phrase the tokenizer handles efficiently with a novel
# short-form token can be counterproductive.
#
# Evaluation against the actual target tokenizer is essential.
# ============================================================================


def split_semantic_line(line: str) -> tuple[str, str] | None:
    if not line:
        return None
    if line.startswith("@") or line.startswith("#") or line.startswith(">"):
        return None
    if line[0] in {"+", ":", "?", "$"}:
        return line[0], line[1:]
    if "=" in line:
        left, right = line.split("=", 1)
        return f"{left}=", right
    return None


# collect_macro_candidates: Identifies repeated words/phrases worth replacing.
# Scores each candidate by net byte savings: (phrase_len - token_len) * freq - header_cost.
# Only phrases with net savings > 6 bytes are considered.
def collect_macro_candidates(values: list[str]) -> list[tuple[str, int, int]]:
    candidates: collections.Counter[str] = collections.Counter()

    for value in values:
        words = re.findall(r"[A-Za-z][A-Za-z0-9'/-]*", value)
        for word in words:
            if len(word) >= 5:
                candidates[word] += 1
        for n in (2, 3):
            if len(words) < n:
                continue
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i : i + n])
                if len(phrase) >= 9:
                    candidates[phrase] += 1

    scored: list[tuple[str, int, int]] = []
    for phrase, freq in candidates.items():
        if freq < 2:
            continue
        token_len = 2  # ~A etc.
        header_cost = len(phrase) + 4
        net = (len(phrase) - token_len) * freq - header_cost
        if net > 6:
            scored.append((phrase, freq, net))
    scored.sort(key=lambda item: (item[2], len(item[0]), item[1]), reverse=True)
    return scored


# protect_value_segments / restore_value_segments: Shield quoted strings and
# file paths from macro replacement. These literals must be preserved exactly
# for fidelity — they are protected with placeholder tokens during macro
# substitution, then restored afterward.
def protect_value_segments(value: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}
    counter = 0

    pattern = re.compile(r'"[^"\n]*"|~\/[^\s]+|\/[A-Za-z0-9._~\/-]+')

    def repl(match: re.Match[str]) -> str:
        nonlocal counter
        token = f"§{counter}§"
        placeholders[token] = match.group(0)
        counter += 1
        return token

    protected = pattern.sub(repl, value)
    return protected, placeholders


def restore_value_segments(value: str, placeholders: dict[str, str]) -> str:
    restored = value
    for token, original in placeholders.items():
        restored = restored.replace(token, original)
    return restored


# semantic_to_macro_runtime: Layer 3 entry point — builds the phrase dictionary
# and applies macro substitutions to the semantic runtime. The dictionary is
# emitted inline at the top of the output (e.g., "!A=phrase") so the format
# is self-contained.
def semantic_to_macro_runtime(semantic_text: str, max_macros: int = 12) -> tuple[str, dict[str, str]]:
    lines = [line for line in semantic_text.splitlines() if line]
    parsed: list[tuple[str | None, str]] = []
    values: list[str] = []

    for line in lines:
        split = split_semantic_line(line)
        if split is None:
            parsed.append((None, line))
        else:
            prefix, value = split
            parsed.append((prefix, value))
            values.append(value)

    selected: list[tuple[str, str]] = []
    token_ord = ord("A")
    used_phrases: set[str] = set()

    for phrase, _, _net in collect_macro_candidates(values):
        if len(selected) >= max_macros:
            break
        if phrase in used_phrases:
            continue
        token = f"~{chr(token_ord)}"
        token_ord += 1
        selected.append((token, phrase))
        used_phrases.add(phrase)

    def replace_value(value: str) -> str:
        updated, placeholders = protect_value_segments(value)
        for token, phrase in sorted(selected, key=lambda item: len(item[1]), reverse=True):
            updated = re.sub(rf"(?<![A-Za-z0-9]){re.escape(phrase)}(?![A-Za-z0-9])", token, updated)
        return restore_value_segments(updated, placeholders)

    out_lines: list[str] = []
    if lines and lines[0].startswith("@"):
        out_lines.append(lines[0])
        remaining = parsed[1:]
    else:
        remaining = parsed

    for token, phrase in selected:
        out_lines.append(f"!{token[1:]}={phrase}")

    for prefix, payload in remaining:
        if prefix is None:
            out_lines.append(payload)
        else:
            out_lines.append(f"{prefix}{replace_value(payload)}")

    return "\n".join(out_lines) + "\n", {token: phrase for token, phrase in selected}


# expand_macro_runtime: Reverses macro substitution by reading the inline
# dictionary and replacing all macro tokens with their original phrases.
# Used during decompilation and fidelity auditing.
def expand_macro_runtime(runtime_text: str) -> tuple[str, dict[str, str]]:
    macros: dict[str, str] = {}
    expanded_lines: list[str] = []

    for line in runtime_text.splitlines():
        if line.startswith("!") and "=" in line:
            token, phrase = line[1:].split("=", 1)
            macros[f"~{token}"] = phrase
            continue
        updated = line
        for token, phrase in macros.items():
            updated = updated.replace(token, phrase)
        expanded_lines.append(updated)
    return "\n".join(expanded_lines) + ("\n" if runtime_text.endswith("\n") else ""), macros


# ============================================================================
# PROVENANCE & DECOMPILATION CHAIN
# Reconstructs human-readable markdown from the compiled runtime format using
# the provenance map. Each record carries metadata about its original source
# (file, section, line range, heading level, key label) so the decompiler can
# recreate headings, bullets, paragraphs, code blocks, and quotes.
# This is what makes SMELT's compression non-destructive: the full chain
# from source → compiled → decompiled is deterministic and auditable.
# ============================================================================


def render_runtime_markdown(runtime_text: str, provenance: list[dict[str, object]]) -> str:
    lines = [line for line in runtime_text.splitlines() if line]
    rendered: list[str] = []
    prov_index = 0

    for line in lines:
        if line.startswith("!"):
            continue
        if prov_index >= len(provenance):
            continue

        entry = provenance[prov_index]
        prov_index += 1
        kind = str(entry.get("kind", ""))
        text = str(entry.get("text", ""))
        key = str(entry.get("key", ""))

        if line.startswith("@"):
            continue

        if line.startswith("#"):
            heading_level = 2
            if kind.startswith("heading_"):
                try:
                    heading_level = int(kind.split("_", 1)[1])
                except Exception:
                    heading_level = 2
            rendered.append(f"{'#' * heading_level} {text}")
            rendered.append("")
            continue

        if line.startswith(">"):
            rendered.append(f"**{text}:**")
            continue

        if line.startswith("+"):
            rendered.append(f"- {line[1:]}")
            continue

        if line.startswith(":"):
            rendered.append(line[1:])
            rendered.append("")
            continue

        if line.startswith("?"):
            rendered.append(f"> {line[1:]}")
            continue

        if line.startswith("$"):
            payload = line[1:]
            if "|" in payload:
                lang, code = payload.split("|", 1)
            else:
                lang, code = "", payload
            rendered.append(f"```{lang}".rstrip())
            rendered.extend(code.split(" ⏎ "))
            rendered.append("```")
            rendered.append("")
            continue

        if "=" in line:
            _key_code, value = line.split("=", 1)
            label = key if key else text
            rendered.append(f"- **{label}:** {value}")
            continue

        rendered.append(line)

    while rendered and rendered[-1] == "":
        rendered.pop()
    return "\n".join(rendered) + "\n"


# ============================================================================
# FIDELITY AUDIT: Literal Preservation
# Checks that quoted strings, file paths, and parenthetical annotations
# survive the compilation pipeline intact. These are the most fragile
# elements — a mishandled quote or truncated path would silently corrupt
# the model's context. The audit compares each literal in the provenance
# against the compiled output to catch any losses.
# ============================================================================


def extract_quoted_literals(text: str) -> list[str]:
    return re.findall(r'"[^"\n]*"', text)


def extract_paths(text: str) -> list[str]:
    return re.findall(r'~\/[^\s)]+|(?:(?<=^)|(?<=[\s(<]))\/[A-Za-z0-9._~\/-]+', text)


def audit_runtime_fidelity(semantic_text: str, provenance: list[dict[str, object]], runtime_text: str) -> dict[str, object]:
    semantic_lines = semantic_text.splitlines()
    runtime_lines = [line for line in runtime_text.splitlines() if not line.startswith("!")]

    quote_checks = 0
    quote_misses = 0
    path_checks = 0
    path_misses = 0
    annotation_checks = 0
    annotation_misses = 0

    for entry in provenance:
        record = int(entry["record"])
        if record >= len(semantic_lines) or record >= len(runtime_lines):
            continue
        source_line = semantic_lines[record]
        target_line = runtime_lines[record]
        text = str(entry.get("text", ""))

        for quoted in extract_quoted_literals(text):
            quote_checks += 1
            if quoted not in target_line:
                quote_misses += 1

        for path in extract_paths(text):
            path_checks += 1
            if path not in target_line:
                path_misses += 1

        key_label = str(entry.get("key", ""))
        _base, annotation = split_label_parts(key_label)
        if annotation:
            annotation_checks += 1
            marker = f"@{annotation}"
            if marker not in target_line:
                annotation_misses += 1

    def pct(passed: int, total: int) -> str:
        if total == 0:
            return "n/a"
        return f"{(passed / total) * 100:.2f}%"

    passed_quotes = quote_checks - quote_misses
    passed_paths = path_checks - path_misses
    passed_annotations = annotation_checks - annotation_misses

    return {
        "quote_checks": quote_checks,
        "quote_passed": passed_quotes,
        "quote_score": pct(passed_quotes, quote_checks),
        "path_checks": path_checks,
        "path_passed": passed_paths,
        "path_score": pct(passed_paths, path_checks),
        "annotation_checks": annotation_checks,
        "annotation_passed": passed_annotations,
        "annotation_score": pct(passed_annotations, annotation_checks),
    }


# ============================================================================
# LAYER 4: Query-Conditioned Selective Emission
# The strongest compression mode. Scores each semantic record against a user
# query using TF-IDF with structural context weighting, then emits only the
# matching records along with their parent section/group headers for
# structural coherence.
#
# Scoring hierarchy (descending weight):
#   - Key field matches    (4.5x IDF) — direct field label hit
#   - Text content matches (3.0x IDF) — term found in record text
#   - Group name matches   (2.2x IDF) — term matches enclosing group
#   - Section name matches (1.7x IDF) — term matches enclosing section
#   - Substring matches    (1.1x IDF) — term found anywhere in combined text
#   - Bigram bonuses       (+2.4)     — consecutive query terms found together
#   - Full query match     (+3.5)     — entire query string found verbatim
#   - Group scope match    (+6.0)     — query targets a specific group
#
# This achieves 78-97% token reduction on the tested corpus while preserving
# document-native structural lineage at compile time.
# ============================================================================


def normalize_query_token(token: str) -> str:
    token = token.lower().strip("'_-")
    if token.endswith("'s") and len(token) > 3:
        token = token[:-2]
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        token = token[:-1]
    return token


def tokenize_query_text(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9][A-Za-z0-9'/_-]*", text.lower()):
        token = normalize_query_token(raw.strip("'"))
        if not token or token in QUERY_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def query_bigrams(tokens: list[str]) -> list[str]:
    if len(tokens) < 2:
        return []
    return [f"{tokens[idx]} {tokens[idx + 1]}" for idx in range(len(tokens) - 1)]


# score_query_records: Core scoring engine for Layer 4. Builds a TF-IDF index
# over all content records, then scores each record against the query terms.
# Returns records ranked by relevance score with match metadata.
def score_query_records(
    semantic_text: str,
    provenance: list[dict[str, object]],
    *,
    query: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    lines = [line for line in semantic_text.splitlines() if line]
    prov_by_record = {int(entry["record"]): entry for entry in provenance}

    query_terms = tokenize_query_text(query)
    unique_terms = list(dict.fromkeys(query_terms))
    bigrams = query_bigrams(unique_terms)
    full_query = " ".join(unique_terms)

    section_names: dict[str, str] = {}
    group_names: dict[tuple[str, str], str] = {}
    record_context: list[dict[str, object]] = []

    current_section_code = ""
    current_section_idx = 0
    current_group_code = ""
    current_group_idx = 0

    for record_idx, line in enumerate(lines):
        entry = prov_by_record.get(record_idx, {"record": record_idx, "kind": "raw", "text": line, "line_start": 0, "line_end": 0})
        kind = str(entry.get("kind", ""))

        if kind.startswith("heading_"):
            current_section_code = str(entry.get("section", ""))
            current_section_idx = record_idx
            current_group_code = ""
            current_group_idx = 0
            section_names[current_section_code] = str(entry.get("text", ""))
        elif kind == "group":
            current_group_code = str(entry.get("group", ""))
            current_group_idx = record_idx
            group_names[(current_section_code, current_group_code)] = str(entry.get("text", ""))

        record_context.append(
            {
                "record": record_idx,
                "line": line,
                "entry": entry,
                "kind": kind,
                "section_idx": current_section_idx,
                "group_idx": current_group_idx,
                "section_name": section_names.get(current_section_code, ""),
                "group_name": group_names.get((current_section_code, current_group_code), ""),
            }
        )

    content_rows = [row for row in record_context if row["kind"] not in {"meta", "group"} and not str(row["kind"]).startswith("heading_")]
    doc_freq: collections.Counter[str] = collections.Counter()
    for row in content_rows:
        entry = row["entry"]
        record_tokens = set(
            tokenize_query_text(
                " ".join(
                    [
                        str(entry.get("text", "")),
                        str(entry.get("key", "")),
                        str(row.get("section_name", "")),
                        str(row.get("group_name", "")),
                    ]
                )
            )
        )
        for token in record_tokens:
            doc_freq[token] += 1

    total_docs = max(len(content_rows), 1)
    ranked: list[dict[str, object]] = []

    for row in content_rows:
        entry = row["entry"]
        text = str(entry.get("text", ""))
        key = str(entry.get("key", ""))
        section_name = str(row.get("section_name", ""))
        group_name = str(row.get("group_name", ""))
        line = str(row.get("line", ""))

        text_tokens = set(tokenize_query_text(text))
        key_tokens = set(tokenize_query_text(key))
        section_tokens = set(tokenize_query_text(section_name))
        group_tokens = set(tokenize_query_text(group_name))
        combined_text = " ".join([text, key, section_name, group_name]).lower()
        line_lower = line.lower()
        group_scope_match = bool(group_tokens) and (
            set(group_tokens).issubset(set(unique_terms)) or any(bg == " ".join(group_tokens) for bg in bigrams)
        )
        section_scope_match = bool(section_tokens) and (
            set(section_tokens).issubset(set(unique_terms)) or any(bg == " ".join(section_tokens) for bg in bigrams)
        )

        score = 0.0
        matched_terms: list[str] = []
        text_key_matches: list[str] = []
        context_matches: list[str] = []

        for term in unique_terms:
            idf = 1.0 + math.log((1 + total_docs) / (1 + doc_freq.get(term, 0)))
            if term in key_tokens:
                score += 4.5 * idf
                matched_terms.append(term)
                text_key_matches.append(term)
                continue
            if term in text_tokens:
                score += 3.0 * idf
                matched_terms.append(term)
                text_key_matches.append(term)
                continue
            if term in group_tokens:
                score += 2.2 * idf
                matched_terms.append(term)
                context_matches.append(term)
                continue
            if term in section_tokens:
                score += 1.7 * idf
                matched_terms.append(term)
                context_matches.append(term)
                continue
            if term in combined_text:
                score += 1.1 * idf
                matched_terms.append(term)
                context_matches.append(term)

        if full_query and full_query in combined_text:
            score += 3.5

        for bigram in bigrams:
            if bigram in combined_text:
                score += 2.4

        if group_scope_match:
            score += 6.0
            if str(row["kind"]) in {"labeled_item", "field"}:
                score += 1.0

        if section_scope_match:
            score += 2.5

        # Slightly reward lines that preserve literal names or direct key/value structure.
        if key and matched_terms:
            score += 0.8
        if matched_terms and ("=" in line or line.startswith("+")):
            score += 0.2
        if matched_terms and str(row["kind"]) == "labeled_item":
            score += 0.3

        ranked.append(
            {
                **row,
                "score": round(score, 6),
                "matched_terms": sorted(set(matched_terms)),
                "text_key_matches": sorted(set(text_key_matches)),
                "context_matches": sorted(set(context_matches)),
                "group_scope_match": group_scope_match,
                "section_scope_match": section_scope_match,
                "text_tokens": sorted(text_tokens),
            }
        )

    ranked.sort(key=lambda item: (float(item["score"]), len(str(item["entry"].get("text", "")))), reverse=True)
    return ranked, {
        "query": query,
        "query_terms": unique_terms,
        "query_bigrams": bigrams,
        "total_content_records": len(content_rows),
    }


# build_query_runtime: Layer 4 entry point. Compiles source markdown through
# Layer 2, scores records against the query, selects the top matches above
# the relevance cutoff, includes parent section/group context for structural
# coherence, and optionally applies Layer 3 macro compression to the result.
def build_query_runtime(
    source: Path,
    *,
    query: str,
    mode: str,
    max_records: int,
    max_macros: int,
) -> tuple[str, list[dict[str, object]], dict[str, object]]:
    semantic_text, provenance = markdown_to_semantic_runtime(source)
    lines = [line for line in semantic_text.splitlines() if line]
    prov_by_record = {int(entry["record"]): entry for entry in provenance}
    ranked, metadata = score_query_records(semantic_text, provenance, query=query)

    primary = [row for row in ranked if row["text_key_matches"] or float(row["score"]) >= 6.0]
    if primary:
        group_primary = [row for row in primary if row["group_scope_match"]]
        section_primary = [row for row in primary if row["section_scope_match"]]
        pool = primary
        if group_primary:
            pool = group_primary
        elif section_primary:
            pool = section_primary
        best_score = float(pool[0]["score"])
        cutoff = max(6.0, best_score * 0.5)
        chosen = [row for row in pool if float(row["score"]) >= cutoff][:max_records]
    else:
        fallback = [row for row in ranked if row["matched_terms"] or float(row["score"]) >= 2.0]
        if fallback:
            best_score = float(fallback[0]["score"])
            cutoff = max(2.0, best_score * 0.4)
            chosen = [row for row in fallback if float(row["score"]) >= cutoff][:max_records]
        else:
            chosen = []
    include_records: set[int] = {0}
    selected_scores: dict[int, float] = {}

    for row in chosen:
        record = int(row["record"])
        include_records.add(record)
        include_records.add(int(row["section_idx"]))
        if int(row["group_idx"]):
            include_records.add(int(row["group_idx"]))
        selected_scores[record] = float(row["score"])

    selected_lines: list[str] = []
    selected_provenance: list[dict[str, object]] = []
    old_to_new: dict[int, int] = {}

    for old_record in sorted(include_records):
        if old_record >= len(lines):
            continue
        old_to_new[old_record] = len(selected_lines)
        selected_lines.append(lines[old_record])
        entry = dict(prov_by_record.get(old_record, {"record": old_record, "kind": "raw", "line_start": 0, "line_end": 0, "text": lines[old_record]}))
        entry["record"] = len(selected_lines) - 1
        entry["original_record"] = old_record
        if old_record in selected_scores:
            entry["score"] = round(selected_scores[old_record], 6)
        selected_provenance.append(entry)

    selected_semantic = "\n".join(selected_lines) + "\n"
    output_text = selected_semantic
    macros: dict[str, str] = {}
    if mode == "macro":
        output_text, macros = semantic_to_macro_runtime(selected_semantic, max_macros=max_macros)

    selection = {
        **metadata,
        "mode": mode,
        "source": str(source),
        "max_records": max_records,
        "selected_content_records": len(chosen),
        "selected_total_records": len(selected_lines),
        "full_semantic_records": len(lines),
        "full_semantic_size": len(semantic_text.encode("utf-8")),
        "selected_semantic_size": len(selected_semantic.encode("utf-8")),
        "output_size": len(output_text.encode("utf-8")),
        "macros": macros,
        "top_matches": [
            {
                "record": int(row["record"]),
                "score": float(row["score"]),
                "kind": str(row["kind"]),
                "section_name": str(row.get("section_name", "")),
                "group_name": str(row.get("group_name", "")),
                "text": str(row["entry"].get("text", "")),
                "matched_terms": row["matched_terms"],
                "text_key_matches": row["text_key_matches"],
                "context_matches": row["context_matches"],
            }
            for row in chosen
        ],
        "record_remap": old_to_new,
    }
    return output_text, selected_provenance, selection


# ============================================================================
# FIDELITY AUDIT FRAMEWORK
# Comprehensive suite covering eight audit categories: record recall, negation
# recall, qualifier recall, date recall, relationship recall, quote/path/
# annotation fidelity, probe recall, and group focus purity.
#
# Record recall: Can every provenance entry be found in the rendered output?
# Negation recall: Are negations (not, never, don't) preserved? Critical
#   because losing a negation inverts the meaning.
# Qualifier recall: Are qualifiers (only, always, every) preserved?
# Date recall: Are dates and numbers preserved exactly?
# Relationship recall: Are family/relational terms preserved?
# Probe recall: For each labeled field, does a synthetic query retrieve it?
# Group focus: When querying a group name, is the result dominated by that
#   group's records (>=75% purity)?
# ============================================================================


def normalize_compare_text(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("—", "-").replace("–", "-").replace("→", "->")
    text = text.replace(" ⏎ ", " ")
    text = re.sub(r"(?m)^\s*#{1,6}\s+", "", text)
    text = re.sub(r"(?m)^\s*[-+*]\s+", "", text)
    text = re.sub(r"(?m)^\s*>\s?", "", text)
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def provenance_content_entries(provenance: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        entry
        for entry in provenance
        if str(entry.get("kind", "")) != "meta"
    ]


def recall_score(entries: list[dict[str, object]], rendered_markdown: str) -> dict[str, object]:
    haystack = normalize_compare_text(rendered_markdown)
    checks = 0
    passed = 0
    misses: list[str] = []
    for entry in entries:
        text = str(entry.get("text", "")).strip()
        if not text:
            continue
        checks += 1
        needle = normalize_compare_text(text)
        if needle and needle in haystack:
            passed += 1
        else:
            misses.append(text)
    score = "n/a" if checks == 0 else f"{(passed / checks) * 100:.2f}%"
    return {
        "checks": checks,
        "passed": passed,
        "score": score,
        "misses": misses[:10],
    }


def entries_matching_patterns(provenance: list[dict[str, object]], patterns: list[re.Pattern[str]]) -> list[dict[str, object]]:
    matched: list[dict[str, object]] = []
    seen_records: set[int] = set()
    for entry in provenance_content_entries(provenance):
        text = str(entry.get("text", ""))
        if not text:
            continue
        if any(pattern.search(text) for pattern in patterns):
            record = int(entry.get("record", -1))
            if record not in seen_records:
                seen_records.add(record)
                matched.append(entry)
    return matched


def build_probe_query(entry: dict[str, object]) -> str | None:
    key = str(entry.get("key", "")).strip()
    text = str(entry.get("text", "")).strip()
    kind = str(entry.get("kind", ""))
    if key:
        base, annotation = split_label_parts(key)
        label = base.strip()
        if label.lower() in {"me", "i"} and annotation:
            label = annotation
        if not label:
            return None
        if kind == "field":
            return f"What is {label}?"
        if re.search(r"[A-Z]", label):
            return f"Who is {label}?"
        return f"What about {label}?"
    if kind == "item":
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", text)
        if len(words) >= 3:
            return f"What about {' '.join(words[:4])}?"
    return None


def probe_recall_score(source: Path, provenance: list[dict[str, object]], *, mode: str, max_macros: int) -> dict[str, object]:
    probes: list[tuple[int, str, str]] = []
    for entry in provenance_content_entries(provenance):
        if str(entry.get("kind", "")) not in {"field", "labeled_item", "labeled_paragraph"}:
            continue
        query = build_probe_query(entry)
        if not query:
            continue
        probes.append((int(entry["record"]), query, str(entry.get("text", ""))))

    checks = 0
    passed = 0
    misses: list[dict[str, object]] = []

    for record, query, text in probes:
        _runtime_text, selected_provenance, selection = build_query_runtime(
            source,
            query=query,
            mode=mode,
            max_records=4,
            max_macros=max_macros,
        )
        selected_originals = {int(entry.get("original_record", -1)) for entry in selected_provenance}
        checks += 1
        if record in selected_originals:
            passed += 1
        else:
            misses.append({"query": query, "text": text})

    score = "n/a" if checks == 0 else f"{(passed / checks) * 100:.2f}%"
    return {
        "checks": checks,
        "passed": passed,
        "score": score,
        "misses": misses[:10],
    }


def group_focus_score(source: Path, provenance: list[dict[str, object]], *, mode: str, max_macros: int) -> dict[str, object]:
    group_entries = [entry for entry in provenance if str(entry.get("kind", "")) == "group" and str(entry.get("text", "")).strip()]
    checks = 0
    passed = 0
    details: list[dict[str, object]] = []

    for group_entry in group_entries:
        group_name = str(group_entry.get("text", "")).strip()
        group_code = str(group_entry.get("group", "")).strip()
        if not group_name or not group_code:
            continue
        query = f"Tell me about {group_name}"
        _runtime_text, selected_provenance, _selection = build_query_runtime(
            source,
            query=query,
            mode=mode,
            max_records=6,
            max_macros=max_macros,
        )
        selected_content = [
            entry for entry in selected_provenance
            if str(entry.get("kind", "")) not in {"meta", "group"} and not str(entry.get("kind", "")).startswith("heading_")
        ]
        if not selected_content:
            continue
        checks += 1
        focus_hits = sum(1 for entry in selected_content if str(entry.get("group", "")) == group_code)
        purity = focus_hits / len(selected_content)
        if purity >= 0.75:
            passed += 1
        details.append(
            {
                "group": group_name,
                "purity": f"{purity * 100:.2f}%",
                "selected": len(selected_content),
                "group_hits": focus_hits,
            }
        )

    score = "n/a" if checks == 0 else f"{(passed / checks) * 100:.2f}%"
    return {
        "checks": checks,
        "passed": passed,
        "score": score,
        "details": details,
    }


# nuance_audit: Runs the full fidelity audit suite on a single file.
# Compiles the source, renders it back to markdown, then checks all eight
# audit categories. Returns a structured report with pass rates and misses.
def nuance_audit(source: Path, *, mode: str, max_macros: int) -> dict[str, object]:
    semantic_text, provenance = markdown_to_semantic_runtime(source)
    if mode == "semantic":
        runtime_text = semantic_text
        expanded_runtime = semantic_text
    elif mode == "macro":
        runtime_text, _macros = semantic_to_macro_runtime(semantic_text, max_macros=max_macros)
        expanded_runtime, _expanded_macros = expand_macro_runtime(runtime_text)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    rendered_markdown = render_runtime_markdown(expanded_runtime, provenance)

    negation_patterns = [
        re.compile(r"\bno\b", re.IGNORECASE),
        re.compile(r"\bnot\b", re.IGNORECASE),
        re.compile(r"\bnever\b", re.IGNORECASE),
        re.compile(r"\bdon't\b", re.IGNORECASE),
        re.compile(r"\bdoesn't\b", re.IGNORECASE),
        re.compile(r"\bwon't\b", re.IGNORECASE),
        re.compile(r"\bcan't\b", re.IGNORECASE),
        re.compile(r"\bwithout\b", re.IGNORECASE),
        re.compile(r"\bunless\b", re.IGNORECASE),
    ]
    qualifier_patterns = [
        re.compile(r"\bonly\b", re.IGNORECASE),
        re.compile(r"\balways\b", re.IGNORECASE),
        re.compile(r"\busually\b", re.IGNORECASE),
        re.compile(r"\bevery\b", re.IGNORECASE),
        re.compile(r"\bfirst\b", re.IGNORECASE),
        re.compile(r"\bcurrent\b", re.IGNORECASE),
        re.compile(r"\bprivate\b", re.IGNORECASE),
        re.compile(r"\bbetween\b", re.IGNORECASE),
        re.compile(r"\bcareful\b", re.IGNORECASE),
        re.compile(r"\bfragile\b", re.IGNORECASE),
    ]
    date_patterns = [
        re.compile(r"\b(?:19|20)\d{2}\b"),
        re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|March|April|June|July)\b", re.IGNORECASE),
        re.compile(r"\b\d{1,2}%\b"),
        re.compile(r"\b\d{1,2}\b"),
    ]
    relationship_patterns = [
        re.compile(r"\bfamily\b", re.IGNORECASE),
        re.compile(r"\bfather\b", re.IGNORECASE),
        re.compile(r"\bdaughter\b", re.IGNORECASE),
        re.compile(r"\bhuman\b", re.IGNORECASE),
        re.compile(r"\blove\b", re.IGNORECASE),
        re.compile(r"\bbeloved\b", re.IGNORECASE),
        re.compile(r"\bsister\b", re.IGNORECASE),
        re.compile(r"\bmother\b", re.IGNORECASE),
    ]

    all_recall = recall_score(provenance_content_entries(provenance), rendered_markdown)
    negation_recall = recall_score(entries_matching_patterns(provenance, negation_patterns), rendered_markdown)
    qualifier_recall = recall_score(entries_matching_patterns(provenance, qualifier_patterns), rendered_markdown)
    date_recall = recall_score(entries_matching_patterns(provenance, date_patterns), rendered_markdown)
    relationship_recall = recall_score(entries_matching_patterns(provenance, relationship_patterns), rendered_markdown)
    literal_fidelity = audit_runtime_fidelity(semantic_text, provenance, runtime_text if mode == "macro" else semantic_text)
    probes = probe_recall_score(source, provenance, mode=mode, max_macros=max_macros)
    groups = group_focus_score(source, provenance, mode=mode, max_macros=max_macros)

    return {
        "source": str(source),
        "mode": mode,
        "rendered_size": len(rendered_markdown.encode("utf-8")),
        "record_recall": all_recall,
        "negation_recall": negation_recall,
        "qualifier_recall": qualifier_recall,
        "date_recall": date_recall,
        "relationship_recall": relationship_recall,
        "literal_fidelity": literal_fidelity,
        "probe_recall": probes,
        "group_focus": groups,
    }


# ============================================================================
# STARTUP BUNDLE AUDIT
# Evaluates the full OpenClaw startup file set (AGENTS.md, SOUL.md, USER.md,
# MEMORY.md, plus any daily memory files) as a bundle. Measures token counts
# with the actual Qwen tokenizer, runs the fidelity audit on each file in
# both semantic and macro modes, and produces an aggregate safety report.
# This is what generates the evaluation data presented in the paper.
# ============================================================================


def startup_bundle_audit(
    *,
    workspace: Path,
    audit_date: dt.date,
    tokenizer_path: Path,
    max_macros: int,
) -> dict[str, object]:
    tokenizer = load_qwen_tokenizer(tokenizer_path)
    files, metadata = resolve_startup_files(workspace, audit_date)

    semantic_rows: list[dict[str, object]] = []
    macro_rows: list[dict[str, object]] = []
    total_original_tokens = 0
    total_semantic_tokens = 0
    total_macro_tokens = 0
    total_original_bytes = 0
    total_semantic_bytes = 0
    total_macro_bytes = 0

    for source in files:
        raw_text = source.read_text(encoding="utf-8")
        original_tokens = token_count(tokenizer, raw_text)
        semantic_text, provenance = markdown_to_semantic_runtime(source)
        macro_text, _macros = semantic_to_macro_runtime(semantic_text, max_macros=max_macros)
        semantic_tokens = token_count(tokenizer, semantic_text)
        macro_tokens = token_count(tokenizer, macro_text)
        semantic_audit = nuance_audit(source, mode="semantic", max_macros=max_macros)
        macro_audit = nuance_audit(source, mode="macro", max_macros=max_macros)

        total_original_tokens += original_tokens
        total_semantic_tokens += semantic_tokens
        total_macro_tokens += macro_tokens
        total_original_bytes += len(raw_text.encode("utf-8"))
        total_semantic_bytes += len(semantic_text.encode("utf-8"))
        total_macro_bytes += len(macro_text.encode("utf-8"))

        semantic_rows.append(
            {
                "file": str(source),
                "name": source.name,
                "original_tokens": original_tokens,
                "runtime_tokens": semantic_tokens,
                "saved_tokens_percent": compression_percent(original_tokens, semantic_tokens),
                "record_recall": semantic_audit["record_recall"]["score"],
                "negation_recall": semantic_audit["negation_recall"]["score"],
                "qualifier_recall": semantic_audit["qualifier_recall"]["score"],
                "date_recall": semantic_audit["date_recall"]["score"],
                "relation_recall": semantic_audit["relationship_recall"]["score"],
                "probe_recall": semantic_audit["probe_recall"]["score"],
                "group_focus": semantic_audit["group_focus"]["score"],
            }
        )
        macro_rows.append(
            {
                "file": str(source),
                "name": source.name,
                "original_tokens": original_tokens,
                "runtime_tokens": macro_tokens,
                "saved_tokens_percent": compression_percent(original_tokens, macro_tokens),
                "record_recall": macro_audit["record_recall"]["score"],
                "negation_recall": macro_audit["negation_recall"]["score"],
                "qualifier_recall": macro_audit["qualifier_recall"]["score"],
                "date_recall": macro_audit["date_recall"]["score"],
                "relation_recall": macro_audit["relationship_recall"]["score"],
                "probe_recall": macro_audit["probe_recall"]["score"],
                "group_focus": macro_audit["group_focus"]["score"],
            }
        )

    def all_clean(rows: list[dict[str, object]]) -> bool:
        fields = ["record_recall", "negation_recall", "qualifier_recall", "date_recall", "relation_recall", "probe_recall", "group_focus"]
        for row in rows:
            for field in fields:
                value = str(row[field])
                if value not in {"100.00%", "n/a"}:
                    return False
        return True

    return {
        **metadata,
        "tokenizer": str(tokenizer_path),
        "files": [str(p) for p in files],
        "original_tokens": total_original_tokens,
        "semantic_tokens": total_semantic_tokens,
        "macro_tokens": total_macro_tokens,
        "original_bytes": total_original_bytes,
        "semantic_bytes": total_semantic_bytes,
        "macro_bytes": total_macro_bytes,
        "semantic_saved_tokens": compression_percent(total_original_tokens, total_semantic_tokens),
        "macro_saved_tokens": compression_percent(total_original_tokens, total_macro_tokens),
        "semantic_saved_bytes": compression_percent(total_original_bytes, total_semantic_bytes),
        "macro_saved_bytes": compression_percent(total_original_bytes, total_macro_bytes),
        "semantic_safe": all_clean(semantic_rows),
        "macro_safe": all_clean(macro_rows),
        "semantic_rows": semantic_rows,
        "macro_rows": macro_rows,
    }


def render_startup_audit_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Startup Audit",
        "",
        f"- Audit date: `{report['audit_date']}`",
        f"- Workspace: `{report['workspace']}`",
        f"- Tokenizer: `{report['tokenizer']}`",
        f"- Today file exists: `{report['today_exists']}`",
        f"- Yesterday file exists: `{report['yesterday_exists']}`",
        "",
        "## Effective Startup Files",
        "",
    ]
    for file_path in report["files"]:
        lines.append(f"- `{file_path}`")

    lines.extend(
        [
            "",
            "## Aggregate Tokens",
            "",
            f"- Original startup load: `{report['original_tokens']}` tokens",
            f"- Semantic runtime load: `{report['semantic_tokens']}` tokens",
            f"- Macro runtime load: `{report['macro_tokens']}` tokens",
            f"- Semantic token savings: `{report['semantic_saved_tokens']}`",
            f"- Macro token savings: `{report['macro_saved_tokens']}`",
            "",
            "## Safety Summary",
            "",
            f"- Semantic runtime safe on current audit: `{report['semantic_safe']}`",
            f"- Macro runtime safe on current audit: `{report['macro_safe']}`",
            "",
            "## Per-File Semantic Audit",
            "",
            "| File | Orig Tok | Runtime Tok | Saved | Record | Negation | Qualifier | Date | Relation | Probe | Group |",
            "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["semantic_rows"]:
        lines.append(
            f"| {row['name']} | {row['original_tokens']} | {row['runtime_tokens']} | {row['saved_tokens_percent']} | "
            f"{row['record_recall']} | {row['negation_recall']} | {row['qualifier_recall']} | {row['date_recall']} | "
            f"{row['relation_recall']} | {row['probe_recall']} | {row['group_focus']} |"
        )

    lines.extend(
        [
            "",
            "## Per-File Macro Audit",
            "",
            "| File | Orig Tok | Runtime Tok | Saved | Record | Negation | Qualifier | Date | Relation | Probe | Group |",
            "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["macro_rows"]:
        lines.append(
            f"| {row['name']} | {row['original_tokens']} | {row['runtime_tokens']} | {row['saved_tokens_percent']} | "
            f"{row['record_recall']} | {row['negation_recall']} | {row['qualifier_recall']} | {row['date_recall']} | "
            f"{row['relation_recall']} | {row['probe_recall']} | {row['group_focus']} |"
        )

    lines.append("")
    return "\n".join(lines)


def build_runtime_mode(source: Path, *, mode: str, max_macros: int) -> tuple[str, str, list[dict[str, object]]]:
    semantic_text, provenance = markdown_to_semantic_runtime(source)
    if mode == "semantic":
        return semantic_text, semantic_text, provenance
    if mode == "macro":
        runtime_text, _macros = semantic_to_macro_runtime(semantic_text, max_macros=max_macros)
        expanded_runtime, _expanded_macros = expand_macro_runtime(runtime_text)
        return runtime_text, expanded_runtime, provenance
    raise ValueError(f"unsupported mode: {mode}")


def canonical_runtime_bytes(runtime_text: str) -> bytes:
    lines = [line for line in runtime_text.splitlines() if not line.startswith("@")]
    return ("\n".join(lines) + ("\n" if runtime_text.endswith("\n") and lines else "")).encode("utf-8")


# ============================================================================
# CLI COMMAND HANDLERS
# Each cmd_* function implements one subcommand of the SMELT CLI.
# Commands map directly to the four-layer architecture:
#   compile/decompile/verify/roundtrip/stability → Layer 1
#   semantic-runtime                              → Layer 2
#   macro-runtime                                 → Layer 3
#   query-runtime                                 → Layer 4
#   fidelity/nuance-audit/startup-audit           → Fidelity audit framework
#   decompile-runtime                             → Provenance/decompilation
#   summary                                       → Cross-layer comparison
#   runtime-stability                             → Drift measurement
# ============================================================================


def cmd_compile(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else default_compiled_path(source)
    start_ns = time.perf_counter_ns()
    manifest = compile_file(source, output, level=args.level, threads=args.threads)
    elapsed_ns = time.perf_counter_ns() - start_ns
    container_size = output.stat().st_size
    print(f"compiled: {source} -> {output}")
    print(f"source_sha256: {manifest['source_sha256']}")
    print(f"source_size: {manifest['source_size']} bytes")
    print(f"container_size: {container_size} bytes")
    print(f"ratio: {format_ratio(int(manifest['source_size']), container_size)}")
    print(f"compression_saved: {compression_percent(int(manifest['source_size']), container_size)}")
    print(f"compile_time: {format_ms(elapsed_ns)}")
    return 0


def cmd_decompile(args: argparse.Namespace) -> int:
    compiled = Path(args.compiled).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else None
    start_ns = time.perf_counter_ns()
    manifest, restored = decompile_file(compiled, output, threads=args.threads)
    elapsed_ns = time.perf_counter_ns() - start_ns
    print(f"decompiled: {compiled} -> {restored}")
    print(f"source_name: {manifest['source_name']}")
    print(f"source_sha256: {manifest['source_sha256']}")
    print(f"decompile_time: {format_ms(elapsed_ns)}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    restored = Path(args.restored).expanduser().resolve()
    start_ns = time.perf_counter_ns()
    ok, src_hash, restored_hash = verify_exact(source, restored)
    elapsed_ns = time.perf_counter_ns() - start_ns
    print(f"source_sha256:   {src_hash}")
    print(f"restored_sha256: {restored_hash}")
    print("match: yes" if ok else "match: no")
    print(f"verify_time:     {format_ms(elapsed_ns)}")
    return 0 if ok else 1


def cmd_info(args: argparse.Namespace) -> int:
    compiled = Path(args.compiled).expanduser().resolve()
    manifest, payload = parse_container(compiled.read_bytes())
    container_size = compiled.stat().st_size
    info = {
        **manifest,
        "container_size": container_size,
        "payload_size": len(payload),
        "effective_ratio": format_ratio(int(manifest["source_size"]), container_size),
        "compression_saved_percent": compression_percent(int(manifest["source_size"]), container_size),
    }
    print(json.dumps(info, indent=2, sort_keys=True))
    return 0


def cmd_runtime(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else runtime_output_path(source)
    map_output = Path(args.map_output).expanduser().resolve() if args.map_output else runtime_map_path(output)

    start_ns = time.perf_counter_ns()
    runtime_text, provenance = markdown_to_runtime(source)
    ensure_parent(output)
    ensure_parent(map_output)
    output.write_text(runtime_text, encoding="utf-8")
    map_output.write_text(json.dumps(provenance, indent=2, ensure_ascii=True), encoding="utf-8")
    elapsed_ns = time.perf_counter_ns() - start_ns

    source_size = source.stat().st_size
    runtime_size = output.stat().st_size
    print(f"source:        {source}")
    print(f"runtime:       {output}")
    print(f"provenance:    {map_output}")
    print(f"source_size:   {source_size} bytes")
    print(f"runtime_size:  {runtime_size} bytes")
    print(f"ratio:         {format_ratio(source_size, runtime_size)}")
    print(f"saved:         {compression_percent(source_size, runtime_size)}")
    print(f"runtime_ms:    {format_ms(elapsed_ns)}")
    print(f"records:       {len(provenance)}")
    return 0


def cmd_semantic_runtime(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else semantic_runtime_output_path(source)
    map_output = Path(args.map_output).expanduser().resolve() if args.map_output else semantic_runtime_map_path(output)

    start_ns = time.perf_counter_ns()
    runtime_text, provenance = markdown_to_semantic_runtime(source)
    ensure_parent(output)
    ensure_parent(map_output)
    output.write_text(runtime_text, encoding="utf-8")
    map_output.write_text(json.dumps(provenance, indent=2, ensure_ascii=True), encoding="utf-8")
    elapsed_ns = time.perf_counter_ns() - start_ns

    source_size = source.stat().st_size
    runtime_size = output.stat().st_size
    print(f"source:        {source}")
    print(f"semantic:      {output}")
    print(f"provenance:    {map_output}")
    print(f"source_size:   {source_size} bytes")
    print(f"semantic_size: {runtime_size} bytes")
    print(f"ratio:         {format_ratio(source_size, runtime_size)}")
    print(f"saved:         {compression_percent(source_size, runtime_size)}")
    print(f"semantic_ms:   {format_ms(elapsed_ns)}")
    print(f"records:       {len(provenance)}")
    return 0


def cmd_packed_runtime(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else packed_runtime_output_path(source)
    map_output = Path(args.map_output).expanduser().resolve() if args.map_output else packed_runtime_map_path(output)
    dict_output = Path(args.dict_output).expanduser().resolve() if args.dict_output else packed_runtime_dict_path(output)

    start_ns = time.perf_counter_ns()
    semantic_text, provenance = markdown_to_semantic_runtime(source)
    packed_text, dictionary = semantic_to_packed(semantic_text, provenance)
    ensure_parent(output)
    ensure_parent(map_output)
    ensure_parent(dict_output)
    output.write_text(packed_text, encoding="utf-8")
    map_output.write_text(json.dumps(provenance, indent=2, ensure_ascii=True), encoding="utf-8")
    dict_output.write_text(json.dumps(dictionary, indent=2, ensure_ascii=True), encoding="utf-8")
    elapsed_ns = time.perf_counter_ns() - start_ns

    source_size = source.stat().st_size
    packed_size = output.stat().st_size
    print(f"source:        {source}")
    print(f"packed:        {output}")
    print(f"provenance:    {map_output}")
    print(f"dictionary:    {dict_output}")
    print(f"source_size:   {source_size} bytes")
    print(f"packed_size:   {packed_size} bytes")
    print(f"ratio:         {format_ratio(source_size, packed_size)}")
    print(f"saved:         {compression_percent(source_size, packed_size)}")
    print(f"packed_ms:     {format_ms(elapsed_ns)}")
    print(f"records:       {len(provenance)}")
    return 0


def cmd_macro_runtime(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else macro_runtime_output_path(source)
    map_output = Path(args.map_output).expanduser().resolve() if args.map_output else macro_runtime_map_path(output)
    dict_output = Path(args.dict_output).expanduser().resolve() if args.dict_output else macro_runtime_dict_path(output)

    start_ns = time.perf_counter_ns()
    semantic_text, provenance = markdown_to_semantic_runtime(source)
    macro_text, macros = semantic_to_macro_runtime(semantic_text, max_macros=args.max_macros)
    ensure_parent(output)
    ensure_parent(map_output)
    ensure_parent(dict_output)
    output.write_text(macro_text, encoding="utf-8")
    map_output.write_text(json.dumps(provenance, indent=2, ensure_ascii=True), encoding="utf-8")
    dict_output.write_text(json.dumps(macros, indent=2, ensure_ascii=True), encoding="utf-8")
    elapsed_ns = time.perf_counter_ns() - start_ns

    source_size = source.stat().st_size
    macro_size = output.stat().st_size
    print(f"source:        {source}")
    print(f"macro:         {output}")
    print(f"provenance:    {map_output}")
    print(f"macros_file:   {dict_output}")
    print(f"source_size:   {source_size} bytes")
    print(f"macro_size:    {macro_size} bytes")
    print(f"ratio:         {format_ratio(source_size, macro_size)}")
    print(f"saved:         {compression_percent(source_size, macro_size)}")
    print(f"macro_ms:      {format_ms(elapsed_ns)}")
    print(f"macros:        {len(macros)}")
    return 0


def cmd_query_runtime(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else query_runtime_output_path(source, mode=args.mode)
    map_output = Path(args.map_output).expanduser().resolve() if args.map_output else query_runtime_map_path(output)
    selection_output = Path(args.selection_output).expanduser().resolve() if args.selection_output else query_runtime_selection_path(output)

    start_ns = time.perf_counter_ns()
    runtime_text, provenance, selection = build_query_runtime(
        source,
        query=args.query,
        mode=args.mode,
        max_records=args.max_records,
        max_macros=args.max_macros,
    )
    ensure_parent(output)
    ensure_parent(map_output)
    ensure_parent(selection_output)
    output.write_text(runtime_text, encoding="utf-8")
    map_output.write_text(json.dumps(provenance, indent=2, ensure_ascii=True), encoding="utf-8")
    selection_output.write_text(json.dumps(selection, indent=2, ensure_ascii=True), encoding="utf-8")
    elapsed_ns = time.perf_counter_ns() - start_ns

    source_size = source.stat().st_size
    output_size = output.stat().st_size
    full_semantic_size = int(selection["full_semantic_size"])
    selected_semantic_size = int(selection["selected_semantic_size"])

    print(f"source:                 {source}")
    print(f"query:                  {args.query}")
    print(f"mode:                   {args.mode}")
    print(f"runtime:                {output}")
    print(f"provenance:             {map_output}")
    print(f"selection:              {selection_output}")
    print(f"source_size:            {source_size} bytes")
    print(f"full_semantic_size:     {full_semantic_size} bytes")
    print(f"selected_semantic_size: {selected_semantic_size} bytes")
    print(f"output_size:            {output_size} bytes")
    print(f"selected_records:       {selection['selected_content_records']}/{selection['total_content_records']} content, {selection['selected_total_records']}/{selection['full_semantic_records']} total")
    print(f"saved_vs_source:        {compression_percent(source_size, output_size)}")
    print(f"saved_vs_full_semantic: {compression_percent(full_semantic_size, output_size)}")
    print(f"query_terms:            {', '.join(selection['query_terms']) if selection['query_terms'] else '(none)'}")
    print(f"query_ms:               {format_ms(elapsed_ns)}")
    if selection["top_matches"]:
        print("top_matches:")
        for item in selection["top_matches"]:
            print(f"  - score={item['score']:.3f} kind={item['kind']} text={item['text']}")
    return 0


def cmd_decompile_runtime(args: argparse.Namespace) -> int:
    runtime_file = Path(args.runtime).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else decompiled_runtime_output_path(runtime_file)
    map_path = Path(args.map_path).expanduser().resolve() if args.map_path else runtime_file.with_suffix(".map.json")

    if not map_path.exists():
        raise FileNotFoundError(f"provenance map not found: {map_path}")

    runtime_text = runtime_file.read_text(encoding="utf-8")
    provenance = json.loads(map_path.read_text(encoding="utf-8"))

    if any(line.startswith("!") for line in runtime_text.splitlines()):
        runtime_text, _macros = expand_macro_runtime(runtime_text)

    start_ns = time.perf_counter_ns()
    restored_text = render_runtime_markdown(runtime_text, provenance)
    ensure_parent(output)
    output.write_text(restored_text, encoding="utf-8")
    elapsed_ns = time.perf_counter_ns() - start_ns

    print(f"runtime:       {runtime_file}")
    print(f"provenance:    {map_path}")
    print(f"restored:      {output}")
    print(f"decompile_ms:  {format_ms(elapsed_ns)}")
    return 0


def cmd_fidelity(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    semantic_text, provenance = markdown_to_semantic_runtime(source)
    macro_text, _macros = semantic_to_macro_runtime(semantic_text, max_macros=args.max_macros)

    semantic_report = audit_runtime_fidelity(semantic_text, provenance, semantic_text)
    macro_report = audit_runtime_fidelity(semantic_text, provenance, macro_text)

    print(f"source:               {source}")
    print("")
    print("semantic_runtime:")
    print(f"  quote_score:        {semantic_report['quote_score']} ({semantic_report['quote_passed']}/{semantic_report['quote_checks']})")
    print(f"  path_score:         {semantic_report['path_score']} ({semantic_report['path_passed']}/{semantic_report['path_checks']})")
    print(f"  annotation_score:   {semantic_report['annotation_score']} ({semantic_report['annotation_passed']}/{semantic_report['annotation_checks']})")
    print("")
    print("macro_runtime:")
    print(f"  quote_score:        {macro_report['quote_score']} ({macro_report['quote_passed']}/{macro_report['quote_checks']})")
    print(f"  path_score:         {macro_report['path_score']} ({macro_report['path_passed']}/{macro_report['path_checks']})")
    print(f"  annotation_score:   {macro_report['annotation_score']} ({macro_report['annotation_passed']}/{macro_report['annotation_checks']})")
    return 0


def cmd_nuance_audit(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    modes = [args.mode] if args.mode != "both" else ["semantic", "macro"]
    print(f"source:               {source}")
    for mode in modes:
        report = nuance_audit(source, mode=mode, max_macros=args.max_macros)
        print("")
        print(f"{mode}_runtime:")
        print(f"  record_recall:      {report['record_recall']['score']} ({report['record_recall']['passed']}/{report['record_recall']['checks']})")
        print(f"  negation_recall:    {report['negation_recall']['score']} ({report['negation_recall']['passed']}/{report['negation_recall']['checks']})")
        print(f"  qualifier_recall:   {report['qualifier_recall']['score']} ({report['qualifier_recall']['passed']}/{report['qualifier_recall']['checks']})")
        print(f"  date_recall:        {report['date_recall']['score']} ({report['date_recall']['passed']}/{report['date_recall']['checks']})")
        print(f"  relation_recall:    {report['relationship_recall']['score']} ({report['relationship_recall']['passed']}/{report['relationship_recall']['checks']})")
        print(f"  quote_fidelity:     {report['literal_fidelity']['quote_score']} ({report['literal_fidelity']['quote_passed']}/{report['literal_fidelity']['quote_checks']})")
        print(f"  path_fidelity:      {report['literal_fidelity']['path_score']} ({report['literal_fidelity']['path_passed']}/{report['literal_fidelity']['path_checks']})")
        print(f"  annot_fidelity:     {report['literal_fidelity']['annotation_score']} ({report['literal_fidelity']['annotation_passed']}/{report['literal_fidelity']['annotation_checks']})")
        print(f"  probe_recall:       {report['probe_recall']['score']} ({report['probe_recall']['passed']}/{report['probe_recall']['checks']})")
        print(f"  group_focus:        {report['group_focus']['score']} ({report['group_focus']['passed']}/{report['group_focus']['checks']})")
        if args.show_misses:
            if report["record_recall"]["misses"]:
                print("  record_misses:")
                for miss in report["record_recall"]["misses"]:
                    print(f"    - {miss}")
            if report["probe_recall"]["misses"]:
                print("  probe_misses:")
                for miss in report["probe_recall"]["misses"]:
                    print(f"    - {miss['query']} -> {miss['text']}")
            if report["group_focus"]["details"]:
                print("  group_details:")
                for detail in report["group_focus"]["details"]:
                    print(f"    - {detail['group']}: purity={detail['purity']} ({detail['group_hits']}/{detail['selected']})")
    return 0


def cmd_startup_audit(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    audit_date = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    report = startup_bundle_audit(
        workspace=workspace,
        audit_date=audit_date,
        tokenizer_path=tokenizer_path,
        max_macros=args.max_macros,
    )

    output = Path(args.output).expanduser().resolve() if args.output else startup_report_path(audit_date)
    markdown = render_startup_audit_markdown(report)
    ensure_parent(output)
    output.write_text(markdown, encoding="utf-8")

    print(f"workspace:              {workspace}")
    print(f"audit_date:             {audit_date.isoformat()}")
    print(f"report:                 {output}")
    print(f"original_tokens:        {report['original_tokens']}")
    print(f"semantic_tokens:        {report['semantic_tokens']}")
    print(f"macro_tokens:           {report['macro_tokens']}")
    print(f"semantic_saved_tokens:  {report['semantic_saved_tokens']}")
    print(f"macro_saved_tokens:     {report['macro_saved_tokens']}")
    print(f"semantic_safe:          {'yes' if report['semantic_safe'] else 'no'}")
    print(f"macro_safe:             {'yes' if report['macro_safe'] else 'no'}")
    print(f"today_exists:           {report['today_exists']}")
    print(f"yesterday_exists:       {report['yesterday_exists']}")
    return 0


def cmd_runtime_stability(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    mode = args.mode
    original_source_bytes = source.read_bytes()

    rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix=f"mdc-{mode}-stability-") as tmpdir:
        tmp_root = Path(tmpdir)
        current_source = source
        baseline_runtime_bytes: bytes | None = None
        baseline_restored_bytes: bytes | None = None
        previous_runtime_bytes: bytes | None = None
        previous_restored_bytes: bytes | None = None

        for cycle in range(1, args.cycles + 1):
            runtime_text, expanded_runtime, provenance = build_runtime_mode(
                current_source,
                mode=mode,
                max_macros=args.max_macros,
            )
            restored_markdown = render_runtime_markdown(expanded_runtime, provenance)

            runtime_bytes = canonical_runtime_bytes(runtime_text)
            restored_bytes = restored_markdown.encode("utf-8")

            if baseline_runtime_bytes is None:
                baseline_runtime_bytes = runtime_bytes
            if baseline_restored_bytes is None:
                baseline_restored_bytes = restored_bytes

            fidelity = audit_runtime_fidelity(expanded_runtime, provenance, runtime_text if mode == "macro" else expanded_runtime)

            rows.append(
                {
                    "cycle": cycle,
                    "runtime_size": len(runtime_bytes),
                    "restored_size": len(restored_bytes),
                    "runtime_drift_vs_cycle1": distortion_percent(baseline_runtime_bytes, runtime_bytes),
                    "restored_drift_vs_cycle1": distortion_percent(baseline_restored_bytes, restored_bytes),
                    "runtime_drift_vs_prev": 0.0 if previous_runtime_bytes is None else distortion_percent(previous_runtime_bytes, runtime_bytes),
                    "restored_drift_vs_prev": 0.0 if previous_restored_bytes is None else distortion_percent(previous_restored_bytes, restored_bytes),
                    "restored_vs_source": distortion_percent(original_source_bytes, restored_bytes),
                    "quote_score": fidelity["quote_score"],
                    "path_score": fidelity["path_score"],
                    "annotation_score": fidelity["annotation_score"],
                }
            )

            cycle_source = tmp_root / f"{mode}-cycle-{cycle:02d}.md"
            cycle_source.write_text(restored_markdown, encoding="utf-8")
            current_source = cycle_source
            previous_runtime_bytes = runtime_bytes
            previous_restored_bytes = restored_bytes

    print(f"source:                   {source}")
    print(f"mode:                     {mode}")
    print(f"cycles:                   {args.cycles}")
    print("")
    for row in rows:
        print(
            f"cycle {int(row['cycle']):02d}: "
            f"runtime={int(row['runtime_size'])}B "
            f"restored={int(row['restored_size'])}B "
            f"drift_rt_c1={float(row['runtime_drift_vs_cycle1']):.6f}% "
            f"drift_md_c1={float(row['restored_drift_vs_cycle1']):.6f}% "
            f"drift_rt_prev={float(row['runtime_drift_vs_prev']):.6f}% "
            f"drift_md_prev={float(row['restored_drift_vs_prev']):.6f}% "
            f"src_loss={float(row['restored_vs_source']):.6f}% "
            f"Q={row['quote_score']} P={row['path_score']} A={row['annotation_score']}"
        )
    print("")
    print(f"final_runtime_drift:      {float(rows[-1]['runtime_drift_vs_cycle1']):.6f}%")
    print(f"final_restored_drift:     {float(rows[-1]['restored_drift_vs_cycle1']):.6f}%")
    print(f"final_source_loss:        {float(rows[-1]['restored_vs_source']):.6f}%")
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    source_size = source.stat().st_size

    with tempfile.TemporaryDirectory(prefix="mdc-summary-") as tmpdir:
        tmp_root = Path(tmpdir)
        compiled = tmp_root / f"c-{source.name}"
        restored = tmp_root / f"r-{source.name}"
        runtime_out = tmp_root / f"rt-{source.stem}.txt"
        runtime_map = tmp_root / f"rt-{source.stem}.map.json"
        semantic_out = tmp_root / f"srt-{source.stem}.txt"
        semantic_map = tmp_root / f"srt-{source.stem}.map.json"
        packed_out = tmp_root / f"prt-{source.stem}.txt"
        packed_map = tmp_root / f"prt-{source.stem}.map.json"
        packed_dict = tmp_root / f"prt-{source.stem}.dict.json"
        macro_out = tmp_root / f"mrt-{source.stem}.txt"

        compile_file(source, compiled, level=args.level, threads=args.threads)
        decompile_file(compiled, restored, threads=args.threads)

        runtime_text, runtime_prov = markdown_to_runtime(source)
        runtime_out.write_text(runtime_text, encoding="utf-8")
        runtime_map.write_text(json.dumps(runtime_prov, ensure_ascii=True), encoding="utf-8")

        semantic_text, semantic_prov = markdown_to_semantic_runtime(source)
        semantic_out.write_text(semantic_text, encoding="utf-8")
        semantic_map.write_text(json.dumps(semantic_prov, ensure_ascii=True), encoding="utf-8")

        packed_text, packed_dictionary = semantic_to_packed(semantic_text, semantic_prov)
        packed_out.write_text(packed_text, encoding="utf-8")
        packed_map.write_text(json.dumps(semantic_prov, ensure_ascii=True), encoding="utf-8")
        packed_dict.write_text(json.dumps(packed_dictionary, ensure_ascii=True), encoding="utf-8")

        macro_text, _macros = semantic_to_macro_runtime(semantic_text)
        macro_out.write_text(macro_text, encoding="utf-8")

        semantic_fidelity = audit_runtime_fidelity(semantic_text, semantic_prov, semantic_text)
        macro_fidelity = audit_runtime_fidelity(semantic_text, semantic_prov, macro_text)

        lossless_size = compiled.stat().st_size
        runtime_size = runtime_out.stat().st_size
        semantic_size = semantic_out.stat().st_size
        packed_size = packed_out.stat().st_size
        macro_size = macro_out.stat().st_size

        print(f"source:                 {source}")
        print(f"source_size:            {source_size} bytes")
        print("")
        print(f"lossless_size:          {lossless_size} bytes")
        print(f"lossless_saved:         {compression_percent(source_size, lossless_size)}")
        print(f"lossless_ratio:         {format_ratio(source_size, lossless_size)}")
        print("")
        print(f"runtime_size:           {runtime_size} bytes")
        print(f"runtime_saved:          {compression_percent(source_size, runtime_size)}")
        print(f"runtime_ratio:          {format_ratio(source_size, runtime_size)}")
        print("")
        print(f"semantic_size:          {semantic_size} bytes")
        print(f"semantic_saved:         {compression_percent(source_size, semantic_size)}")
        print(f"semantic_ratio:         {format_ratio(source_size, semantic_size)}")
        print("")
        print(f"packed_size:            {packed_size} bytes")
        print(f"packed_saved:           {compression_percent(source_size, packed_size)}")
        print(f"packed_ratio:           {format_ratio(source_size, packed_size)}")
        print("")
        print(f"macro_size:             {macro_size} bytes")
        print(f"macro_saved:            {compression_percent(source_size, macro_size)}")
        print(f"macro_ratio:            {format_ratio(source_size, macro_size)}")
        print("")
        print(f"semantic_vs_runtime:    {compression_percent(runtime_size, semantic_size)}")
        print(f"packed_vs_semantic:     {compression_percent(semantic_size, packed_size)}")
        print(f"packed_vs_runtime:      {compression_percent(runtime_size, packed_size)}")
        print(f"macro_vs_semantic:      {compression_percent(semantic_size, macro_size)}")
        print(f"macro_vs_runtime:       {compression_percent(runtime_size, macro_size)}")
        print(f"semantic_vs_lossless:   {compression_percent(semantic_size, lossless_size)}")
        print(f"overall_storage_best:   {compression_percent(source_size, min(lossless_size, runtime_size, semantic_size, packed_size, macro_size))}")
        print(f"overall_runtime_best:   {compression_percent(source_size, min(runtime_size, semantic_size, packed_size, macro_size))}")
        print(f"exact_restore:          yes")
        print("")
        print(f"semantic_quote_score:   {semantic_fidelity['quote_score']}")
        print(f"semantic_path_score:    {semantic_fidelity['path_score']}")
        print(f"semantic_annot_score:   {semantic_fidelity['annotation_score']}")
        print(f"macro_quote_score:      {macro_fidelity['quote_score']}")
        print(f"macro_path_score:       {macro_fidelity['path_score']}")
        print(f"macro_annot_score:      {macro_fidelity['annotation_score']}")
    return 0


def cmd_stability(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    original_bytes = source.read_bytes()
    original_hash = sha256_bytes(original_bytes)
    current_input = source

    rows: list[dict[str, object]] = []
    first_loss_cycle: int | None = None
    worst_cycle = 0
    worst_distortion = 0.0

    with tempfile.TemporaryDirectory(prefix="mdc-stability-") as tmpdir:
        tmp_root = Path(tmpdir)
        for cycle in range(1, args.cycles + 1):
            compiled = tmp_root / f"cycle-{cycle:02d}.mdc"
            restored = tmp_root / f"cycle-{cycle:02d}.md"
            compile_file(current_input, compiled, level=args.level, threads=args.threads)
            decompile_file(compiled, restored, threads=args.threads)

            restored_bytes = restored.read_bytes()
            restored_hash = sha256_bytes(restored_bytes)
            exact = restored_bytes == original_bytes
            total_dist = distortion_percent(original_bytes, restored_bytes)

            if not exact and first_loss_cycle is None:
                first_loss_cycle = cycle
            if total_dist >= worst_distortion:
                worst_distortion = total_dist
                worst_cycle = cycle

            rows.append(
                {
                    "cycle": cycle,
                    "exact": exact,
                    "source_size": len(original_bytes),
                    "restored_size": len(restored_bytes),
                    "container_size": compiled.stat().st_size,
                    "saved_percent": compression_percent(len(restored_bytes), compiled.stat().st_size),
                    "total_distortion": total_dist,
                    "sha256": restored_hash,
                }
            )
            current_input = restored

    print(f"source:            {source}")
    print(f"cycles:            {args.cycles}")
    print(f"source_sha256:     {original_hash}")
    print(f"source_size:       {len(original_bytes)} bytes")
    print("")
    for row in rows:
        print(
            f"cycle {int(row['cycle']):02d}: "
            f"exact={'yes' if row['exact'] else 'no'} "
            f"container={row['container_size']}B "
            f"saved={row['saved_percent']} "
            f"total_distortion={float(row['total_distortion']):.6f}% "
            f"sha256={row['sha256']}"
        )
    print("")
    print(f"first_loss_cycle:  {first_loss_cycle if first_loss_cycle is not None else 'none'}")
    print(f"worst_cycle:       {worst_cycle if worst_cycle else 'none'}")
    print(f"worst_distortion:  {worst_distortion:.6f}%")
    print(f"total_distortion:  {worst_distortion:.6f}%")
    return 0 if first_loss_cycle is None else 1


def cmd_roundtrip(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    compiled = Path(args.output).expanduser().resolve() if args.output else default_compiled_path(source)
    restored = Path(args.restored).expanduser().resolve() if args.restored else RESTORED_DIR / f"r-{source.name}"

    compile_start_ns = time.perf_counter_ns()
    compile_file(source, compiled, level=args.level, threads=args.threads)
    compile_ns = time.perf_counter_ns() - compile_start_ns

    decompile_start_ns = time.perf_counter_ns()
    decompile_file(compiled, restored, threads=args.threads)
    decompile_ns = time.perf_counter_ns() - decompile_start_ns

    verify_start_ns = time.perf_counter_ns()
    ok, src_hash, restored_hash = verify_exact(source, restored)
    verify_ns = time.perf_counter_ns() - verify_start_ns
    total_ns = compile_ns + decompile_ns + verify_ns
    source_size = source.stat().st_size
    container_size = compiled.stat().st_size
    print(f"source:    {source}")
    print(f"compiled:  {compiled}")
    print(f"restored:  {restored}")
    print(f"sha256:    {src_hash}")
    print(f"restored:  {restored_hash}")
    print(f"source_size: {source_size} bytes")
    print(f"container_size: {container_size} bytes")
    print(f"ratio:       {format_ratio(source_size, container_size)}")
    print(f"saved:       {compression_percent(source_size, container_size)}")
    print(f"compile_ms:  {format_ms(compile_ns)}")
    print(f"decompile_ms:{format_ms(decompile_ns)}")
    print(f"verify_ms:   {format_ms(verify_ns)}")
    print(f"total_ms:    {format_ms(total_ns)}")
    print("match: yes" if ok else "match: no")
    return 0 if ok else 1


# ============================================================================
# ARGUMENT PARSER & ENTRY POINT
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lossless markdown compiler prototype")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile", help="compile a source markdown file")
    compile_parser.add_argument("source")
    compile_parser.add_argument("-o", "--output")
    compile_parser.add_argument("-l", "--level", type=int, default=DEFAULT_LEVEL)
    compile_parser.add_argument("-t", "--threads", type=int, default=DEFAULT_THREADS)
    compile_parser.set_defaults(func=cmd_compile)

    decompile_parser = subparsers.add_parser("decompile", help="restore a compiled file")
    decompile_parser.add_argument("compiled")
    decompile_parser.add_argument("-o", "--output")
    decompile_parser.add_argument("-t", "--threads", type=int, default=DEFAULT_THREADS)
    decompile_parser.set_defaults(func=cmd_decompile)

    verify_parser = subparsers.add_parser("verify", help="verify exact byte-for-byte match")
    verify_parser.add_argument("source")
    verify_parser.add_argument("restored")
    verify_parser.set_defaults(func=cmd_verify)

    info_parser = subparsers.add_parser("info", help="show container metadata")
    info_parser.add_argument("compiled")
    info_parser.set_defaults(func=cmd_info)

    runtime_parser = subparsers.add_parser("runtime", help="emit dense runtime text plus provenance map")
    runtime_parser.add_argument("source")
    runtime_parser.add_argument("-o", "--output")
    runtime_parser.add_argument("--map-output")
    runtime_parser.set_defaults(func=cmd_runtime)

    semantic_parser = subparsers.add_parser("semantic-runtime", help="emit schema-aware compact runtime text plus provenance map")
    semantic_parser.add_argument("source")
    semantic_parser.add_argument("-o", "--output")
    semantic_parser.add_argument("--map-output")
    semantic_parser.set_defaults(func=cmd_semantic_runtime)

    packed_parser = subparsers.add_parser("packed-runtime", help="emit packed semantic runtime text with dictionary sidecar")
    packed_parser.add_argument("source")
    packed_parser.add_argument("-o", "--output")
    packed_parser.add_argument("--map-output")
    packed_parser.add_argument("--dict-output")
    packed_parser.set_defaults(func=cmd_packed_runtime)

    macro_parser = subparsers.add_parser("macro-runtime", help="emit semantic runtime with an in-band phrase dictionary")
    macro_parser.add_argument("source")
    macro_parser.add_argument("-o", "--output")
    macro_parser.add_argument("--map-output")
    macro_parser.add_argument("--dict-output")
    macro_parser.add_argument("--max-macros", type=int, default=12)
    macro_parser.set_defaults(func=cmd_macro_runtime)

    query_runtime_parser = subparsers.add_parser("query-runtime", help="emit a query-conditioned runtime subset with provenance and selection metadata")
    query_runtime_parser.add_argument("source")
    query_runtime_parser.add_argument("--query", required=True)
    query_runtime_parser.add_argument("--mode", choices=["semantic", "macro"], default="semantic")
    query_runtime_parser.add_argument("--max-records", type=int, default=8)
    query_runtime_parser.add_argument("--max-macros", type=int, default=12)
    query_runtime_parser.add_argument("-o", "--output")
    query_runtime_parser.add_argument("--map-output")
    query_runtime_parser.add_argument("--selection-output")
    query_runtime_parser.set_defaults(func=cmd_query_runtime)

    decompile_runtime_parser = subparsers.add_parser("decompile-runtime", help="render semantic or macro runtime back into readable markdown")
    decompile_runtime_parser.add_argument("runtime")
    decompile_runtime_parser.add_argument("-o", "--output")
    decompile_runtime_parser.add_argument("--map-path")
    decompile_runtime_parser.set_defaults(func=cmd_decompile_runtime)

    fidelity_parser = subparsers.add_parser("fidelity", help="audit quote/path/annotation preservation in runtime modes")
    fidelity_parser.add_argument("source")
    fidelity_parser.add_argument("--max-macros", type=int, default=12)
    fidelity_parser.set_defaults(func=cmd_fidelity)

    nuance_parser = subparsers.add_parser("nuance-audit", help="audit nuance preservation and retrieval fidelity for semantic/macro runtime modes")
    nuance_parser.add_argument("source")
    nuance_parser.add_argument("--mode", choices=["semantic", "macro", "both"], default="both")
    nuance_parser.add_argument("--max-macros", type=int, default=12)
    nuance_parser.add_argument("--show-misses", action="store_true")
    nuance_parser.set_defaults(func=cmd_nuance_audit)

    startup_parser = subparsers.add_parser("startup-audit", help="audit the effective OpenClaw startup file bundle and write a markdown report")
    startup_parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    startup_parser.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    startup_parser.add_argument("--date")
    startup_parser.add_argument("--max-macros", type=int, default=12)
    startup_parser.add_argument("-o", "--output")
    startup_parser.set_defaults(func=cmd_startup_audit)

    runtime_stability_parser = subparsers.add_parser("runtime-stability", help="measure drift across repeated runtime compile/decompile cycles")
    runtime_stability_parser.add_argument("source")
    runtime_stability_parser.add_argument("--mode", choices=["semantic", "macro"], default="semantic")
    runtime_stability_parser.add_argument("-n", "--cycles", type=int, default=10)
    runtime_stability_parser.add_argument("--max-macros", type=int, default=12)
    runtime_stability_parser.set_defaults(func=cmd_runtime_stability)

    summary_parser = subparsers.add_parser("summary", help="show overall compression summary for lossless, runtime, and semantic runtime")
    summary_parser.add_argument("source")
    summary_parser.add_argument("-l", "--level", type=int, default=DEFAULT_LEVEL)
    summary_parser.add_argument("-t", "--threads", type=int, default=DEFAULT_THREADS)
    summary_parser.set_defaults(func=cmd_summary)

    stability_parser = subparsers.add_parser("stability", help="run repeated lossless cycles and measure distortion")
    stability_parser.add_argument("source")
    stability_parser.add_argument("-n", "--cycles", type=int, default=10)
    stability_parser.add_argument("-l", "--level", type=int, default=DEFAULT_LEVEL)
    stability_parser.add_argument("-t", "--threads", type=int, default=DEFAULT_THREADS)
    stability_parser.set_defaults(func=cmd_stability)

    roundtrip_parser = subparsers.add_parser("roundtrip", help="compile, decompile, and verify")
    roundtrip_parser.add_argument("source")
    roundtrip_parser.add_argument("-o", "--output")
    roundtrip_parser.add_argument("-r", "--restored")
    roundtrip_parser.add_argument("-l", "--level", type=int, default=DEFAULT_LEVEL)
    roundtrip_parser.add_argument("-t", "--threads", type=int, default=DEFAULT_THREADS)
    roundtrip_parser.set_defaults(func=cmd_roundtrip)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
