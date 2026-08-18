# Phase 1 Qwen3-TTS Generation Report

## Status

Phase 1 produced a complete five-story bilingual pack with the official non-streaming
Qwen3-TTS Base model. The pack contains ten validated WAV files and is marked
`ready_for_review`; it has not been manually approved or published.

On 25 July 2026, the user completed the listening review, reported that all ten files sounded
good, and accepted the short story lengths for the prototype. This passes the Phase 1 human
listening gate. The original CLI manifest remains `ready_for_review` because publication approval
is now performed through the Phase 2 control service.

- Generation date: 24 July 2026
- Primary run: `run_qwen_samples_20260724`
- Primary pack: `pack_477f50c0187d46d3`
- Manifest: `data/packs/pack_477f50c0187d46d3/manifest.json`
- Cache verification run: `run_qwen_cache_check_20260724`
- Cache verification pack: `pack_31335e7b0290480c`

## Implemented pipeline

The repository now provides:

- Folder-based English and Mandarin story discovery and validation.
- Reference conversion to mono 24 kHz WAV, including an FFmpeg fallback for M4A/AAC input.
- Deterministic paragraph- and sentence-aware chunking.
- A Qwen engine that loads the model once and reuses a prepared voice-clone prompt.
- An English-reference fallback for Mandarin when no Mandarin reference is supplied.
- A deterministic fake engine for automated testing without a GPU.
- Content-addressed chunk caching based on the model, generation settings, reference hashes,
  language, and exact chunk text.
- Preserved raw chunks, per-chunk metadata, structured JSONL logs, final WAV assembly, peak
  normalization, validation, checksums, and review-state manifests.
- Safe failed-run resume and completed-pack overwrite protection.

The command-line interfaces are:

```text
story-pipeline validate-library
story-pipeline generate
story-pipeline smoke-test
```

## Story library

The current story library contains seven short bilingual stories:

| Story ID | English title | Trigger object |
| --- | --- | --- |
| `forest` | Forest | `forest` |
| `dad` | Dad | `dad` |
| `moon` | Moon | `moon` |
| `dog` | Dog | `dog` |
| `river` | River | `river` |
| `house` | House | `house` |
| `bear` | Bear | `bear` |

These stories produce fourteen full-story WAVs and fourteen card-word WAVs. They should be reviewed
before being treated as final research study content.

## Reference recording

The supplied reference was processed as follows:

- Source: `ref/Recording (136).m4a`
- Source format: stereo AAC at 44.1 kHz
- Source duration: 28.352 seconds
- Normalized format: mono PCM WAV at 24 kHz
- Normalized SHA-256: `3c71ff36178372c7b795594cc33c5ff76c81640412b1a861b89273f538f56fbe`
- Duration warnings: none; the recording is inside the preferred 10–40 second range
- Transcript: supplied separately in `ref/ref.txt` and not reproduced in this report

Only an English reference was supplied. English outputs used it directly, while every Mandarin
output used it as the explicitly recorded cross-language fallback. The raw reference and
transcript are ignored by Git and are not included in story packs.

## Model and runtime

| Property | Value |
| --- | --- |
| Engine | Official non-streaming Qwen3-TTS |
| Model | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
| Model revision | `5d83992436eae1d760afd27aff78a71d676296fc` |
| `qwen-tts` | 0.1.1 |
| PyTorch | 2.6.0+cu124 |
| CUDA runtime | 12.4 |
| GPU | NVIDIA GeForce RTX 4070 |
| GPU memory | 12,877,955,072 bytes |
| Peak PyTorch allocation | 2,957,065,216 bytes, approximately 2.75 GiB |
| Python | 3.10.12 |
| Platform | WSL2/Linux x86-64 |

The peak GPU figure is PyTorch's allocator measurement. It is not a whole-device NVML peak and
does not include every external process or driver allocation.

## GPU smoke test

The bilingual smoke test passed before the full pack was generated.

| Language | Duration | Generation time | Validation warnings |
| --- | ---: | ---: | ---: |
| English | 2.80 s | 4.168 s | 0 |
| Mandarin | 3.12 s | 4.417 s | 0 |

The smoke test took 8.595 seconds for its two inference calls and used the same pinned model and
normalized reference as the full run.

## Full generation metrics

| Story | Language | Chunks | Audio duration | Generation time | RTF | Warnings |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Apple Picnic | English | 2 | 22.27 s | 29.976 s | 1.346 | 0 |
| Apple Picnic | Mandarin | 2 | 21.71 s | 29.039 s | 1.338 | 0 |
| Moon Adventure | English | 2 | 22.91 s | 30.505 s | 1.332 | 0 |
| Moon Adventure | Mandarin | 2 | 24.67 s | 32.758 s | 1.328 | 0 |
| Dad's Little Toolbox | English | 2 | 27.47 s | 37.067 s | 1.349 | 0 |
| Dad's Little Toolbox | Mandarin | 2 | 25.23 s | 33.013 s | 1.308 | 0 |
| Rainy Day Band | English | 2 | 24.99 s | 33.455 s | 1.339 | 0 |
| Rainy Day Band | Mandarin | 2 | 25.07 s | 33.378 s | 1.331 | 0 |
| Sleepy Little Train | English | 2 | 23.07 s | 30.356 s | 1.316 | 0 |
| Sleepy Little Train | Mandarin | 2 | 23.95 s | 32.042 s | 1.338 | 0 |

Aggregate results:

- Final WAV files: 10
- Generated chunks: 20
- Total finished audio: 241.34 seconds, or approximately 4 minutes 1 second
- Sum of inference time: 321.589 seconds
- Recorded run window: 322.973 seconds
- Weighted real-time factor: 1.333
- Per-file RTF range: 1.308–1.349
- Per-file duration range: 21.71–27.47 seconds
- Audio validation warnings: 0
- Primary-run cache hits: 0, as expected for the initial generation

RTF is the sum of chunk inference time divided by assembled output duration. The output duration
includes configured inter-chunk silence. Initial model loading happens before the recorded run
window and is therefore not included in this RTF or run-duration measurement.

## Validation and cache proof

Every final file passed the implemented checks:

- WAV readability
- Positive duration and plausible text-to-duration range
- Finite samples
- Non-silence
- Clipping threshold
- 24 kHz sample rate
- SHA-256 comparison against the manifest

An identical second generation produced:

- Cache hits: 20 of 20 chunks
- Regenerated chunks: 0
- Recorded cache-run window: 1.260 seconds, excluding model startup
- Final audio hashes: identical for all ten files across both packs

The normal automated suite also passed with 24 tests, and Ruff reported no lint errors. Tests
cover library validation, English and Mandarin chunking, FFmpeg M4A conversion, audio assembly,
cache corruption, selective cache invalidation, manifests, complete fake generation, interrupted
run recovery, resume input protection, and completed-run protection.

## Reproduction

Validate the library:

```bash
story-pipeline validate-library --library story_library/library.yaml
```

Run the bilingual GPU smoke test:

```bash
story-pipeline smoke-test \
  --reference-audio 'ref/Recording (136).m4a' \
  --reference-transcript-file ref/ref.txt \
  --output-dir data/smoke-test
```

Generate a new seven-story pack:

```bash
story-pipeline generate \
  --engine qwen \
  --library story_library/library.yaml \
  --reference-audio 'ref/Recording (136).m4a' \
  --reference-transcript-file ref/ref.txt \
  --voice-id caregiver_sample \
  --voice-name 'Caregiver Sample' \
  --output-root data
```

The environment must contain a compatible CUDA-enabled PyTorch installation and the project's
`qwen` optional dependencies, as described in the repository README.

## Remaining evaluation

The current metrics establish technical execution and file integrity. They do not establish:

- Perceived speaker similarity
- Naturalness or emotional appropriateness
- English or Mandarin pronunciation accuracy
- Cross-language accent quality
- Text completeness or hallucinated/repeated speech
- Child, caregiver, or researcher preference

All ten files need manual listening before approval. A later evaluation phase can add structured
human ratings and, if useful, offline ASR text-alignment and speaker-embedding similarity metrics.
