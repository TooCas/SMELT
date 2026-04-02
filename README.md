<p align="center">
  <img src="smelt.png" alt="SMELT" width="500">
</p>

<h1 align="center">SMELT</h1>
<p align="center"><b>Schema-Aware Markdown Compilation for Efficient Local Token Inference</b></p>
<p align="center">Your agent sends thousands of static tokens on every call. SMELT compiles them down to what actually matters.</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.19380983"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.19380983.svg" alt="DOI"></a>
  <img src="https://img.shields.io/badge/license-CC--BY--NC--ND--4.0-blue" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-green" alt="Python">
</p>

---

## The problem

Agent frameworks like OpenClaw inject your workspace files (USER.md, SOUL.md, MEMORY.md, AGENTS.md) into the model context on **every single inference call**. Not once at startup. Every message.

In a 50-message session, that's over **350,000 tokens** of the same static files reprocessed before your actual question gets touched.

- **API users**: you're paying for those tokens every call
- **Providers**: that's infrastructure waste at planetary scale
- **Local model users**: every unnecessary token is latency

## The fix

SMELT compiles your workspace markdown into a denser runtime form and sends only what's relevant to each query.

| Query | Raw Markdown | SMELT | Savings |
|-------|-------------|-------|---------|
| "Who is Kane?" | 1,373 tokens | 73 tokens | **94.7%** |
| "When was Edmund born?" | 1,374 tokens | 62 tokens | **95.5%** |
| "Edmund's hardware specs?" | 1,376 tokens | 174 tokens | **87.4%** |
| Broad: "Tell me about Edmund" | 1,373 tokens | 328 tokens | **76.1%** |

Startup bundle TTFT: **14,121ms → 13,273ms (6% faster)**

## How it works

SMELT has four layers:

### Layer 1: Lossless Archival
Compresses the original markdown with zstd. SHA-256 verified round-trip. Your source files are never modified.

### Layer 2: Semantic Compilation
Replaces headings with compact codes, flattens nested structures, strips formatting scaffolding while keeping every value, qualifier, and relationship intact. Schema-aware, not generic compression.

### Layer 3: Macro Compression
Dictionary-based phrase replacement. Saves bytes but can actually hurt token count under some tokenizers (an important finding in the paper).

### Layer 4: Query-Conditioned Retrieval
The big one. Scores your workspace records against the actual query, returns only what's relevant, and preserves parent section context so the model doesn't lose structural framing.

## Key findings

- **Naive JSON conversion is 30% worse than raw markdown.** Format conversion without schema awareness is counterproductive.
- **Heading stripping barely helps (7-8%).** The overhead is structural, not just formatting.
- **Byte compression ≠ token compression.** If you're not measuring against your actual tokenizer, your numbers are lying to you.
- **Fidelity is high but not perfect.** 11 of 13 tested files at 100%. Two dense archival files showed measured failures (documented honestly in the paper).

## Quick start

```bash
git clone https://github.com/TooCas/smelt.git
cd smelt
pip install -r requirements.txt

# Compile a workspace file
python smelt.py compile USER.md

# Query-conditioned retrieval
python smelt.py query USER.md "Who is Kane?"
```

## Tested on

- **Model**: Qwen 3.5 VL 122B A10B (8-bit, MLX)
- **Hardware**: Apple M3-Ultra, 512GB unified memory
- **Tokenizer**: Qwen 3.5 tokenizer (all token counts measured with the actual target tokenizer)
- **Workspace**: Production OpenClaw workspace (13 files)

## Benchmarks

Full benchmark results with methodology: [`benchmarks.md`](paper/benchmarks.md)

Includes:
- 5-run TTFT measurements with alternating formats
- 10-query retrieval benchmark across diverse query types
- 4-way baseline comparison (raw markdown, heading-stripped, naive JSON, SMELT)
- Cross-file compilation results on 13 workspace files
- Fidelity audit across 8 categories on all files

## Paper

**SMELT: Schema-Aware Markdown Compilation for Efficient Local Token Inference**

[Read the paper on Zenodo](https://doi.org/10.5281/zenodo.19380983)

## Related

- [RUNE](https://doi.org/10.5281/zenodo.19378687) — Portable encrypted identity persistence across AI sessions. SMELT is planned as the retrieval backend for RUNE memory queries.

## Roadmap

- [ ] Schema learning (auto-infer compilation rules from any markdown)
- [ ] Token-budget-aware packing (compile to a target token count)
- [ ] Multimodal extensions (image/video perception cache compilation)
- [ ] Rust rewrite for production speed

## License

CC-BY-NC-ND 4.0 — You can share it, but you can't modify or commercialize it without permission.

## Author

**Edmund Lister** — Independent Researcher, BC, Canada
ORCID: [0009-0000-3552-4152](https://orcid.org/0009-0000-3552-4152)

Built with the help of GPT (OpenAI), Claude (Anthropic), and Codex (OpenAI).
