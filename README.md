# Cached Qwen3-TTS story-pack generator

This repository implements Phase 1 of a local-first storytelling pipeline. It validates a
folder-based multilingual story library, clones a caregiver voice with the official non-streaming
Qwen3-TTS Base model, caches generated chunks by their complete inputs, and writes review-ready
WAV packs with reproducibility metadata.

The included library contains seven short stories in English, Mandarin Chinese, Japanese,
Korean, German, Portuguese, and Spanish. Each story generates a full-story WAV and a
separate card-word WAV in each selected language: 98 files when all seven are selected.
The added translations follow the current English text, including the short presentation
message in `bear/en.txt`; Portuguese uses Brazilian wording.
All audio should be reviewed before use as research study content.

## Requirements

- Ubuntu/WSL2 or Linux with Python 3.10–3.12.
- FFmpeg/libsndfile available to the operating system.
- For real generation: an NVIDIA GPU, a compatible CUDA-enabled PyTorch build, and enough VRAM
  for `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.
- Audio and voices you have permission to clone.

Create an environment and install the CPU-safe development dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

For Qwen, install a CUDA-enabled `torch` and `torchaudio` build appropriate for the machine,
then install the optional engine dependency:

```bash
python -m pip install -e '.[qwen]'
```

The model defaults to revision `5d83992436eae1d760afd27aff78a71d676296fc`. The first online
run downloads it from Hugging Face unless that revision is already cached.

## Story library

`story_library/library.yaml` declares the library and ordered story directory IDs. Every
`story_library/stories/<story-id>/` directory contains:

- `story.yaml` with a matching safe `story_id`, positive version, translated titles, localized
  `spoken_word` values, and trigger object IDs;
- a non-empty UTF-8 `<language-code>.txt` for every language in `required_languages`.

The pipeline supports English, Mandarin Chinese, Japanese, Korean, German,
Portuguese, Spanish, and French (French translations are not bundled).
Language codes are `en`, `zh`, `ja`, `ko`, `de`, `pt`, `es`, and `fr`.
Japanese uses sentence boundaries without inserting spaces; Korean and Portuguese
preserve word spaces. New output languages use the prepared English voice reference
for cross-language cloning. Mandarin can still use its optional dedicated reference.

Validate it with:

```bash
story-pipeline validate-library --library story_library/library.yaml
```

Adding a story requires only its directory/files and a new entry in `library.yaml`.

The localized `spoken_word` in `story.yaml` controls exactly what the card says. Generated manifests
expose these short clips through `word_audio`, separately from the full-story `audio` entries.

## Generate without a GPU

The deterministic fake engine exercises the complete pipeline. Create a short non-silent WAV
and matching transcript, then run:

```bash
story-pipeline generate \
  --engine fake \
  --library story_library/library.yaml \
  --reference-audio /absolute/path/reference.wav \
  --reference-transcript-file /absolute/path/reference.txt \
  --voice-id caregiver_001 \
  --voice-name 'Test caregiver'
```

Omitting `--story` selects all stories; omitting `--language` selects every language declared by
the library.
Mandarin uses the English reference unless both `--mandarin-reference-audio` and
`--mandarin-reference-transcript-file` are supplied. Use `--output-root` to change `data/`.

Each run writes private references to `data/references/`, resumable chunks and failure details
to `data/jobs/`, reusable chunks to `data/cache/chunks/`, and a completed pack to
`data/packs/<pack-id>/`. Generated data is ignored by Git.

To continue a failed run without changing any inputs:

```bash
story-pipeline generate [the same options] --run-id run_ID_FROM_OUTPUT --resume
```

A completed run cannot be overwritten. A new identical run receives a new pack ID but reuses
the validated content-addressed chunk cache.

## Generate with Qwen

Use the same command with `--engine qwen` (the default). Useful overrides include `--device`,
`--dtype`, `--attention-implementation`, `--model-name`, `--model-revision`, `--seed`,
`--max-chunk-characters`, and `--chunk-silence-ms`.

The engine loads once, creates one reusable ICL voice-clone prompt for each supplied reference,
and reuses it for every selected chunk. Raw chunks remain available for review and recovery.

Run the explicit bilingual GPU smoke test before a large pack:

```bash
story-pipeline smoke-test \
  --reference-audio /absolute/path/reference.wav \
  --reference-transcript-file /absolute/path/reference.txt \
  --output-dir data/smoke-test
```

It writes two WAVs and `smoke_test.json` containing hashes, timings, runtime versions, and GPU
information. GPU tests are deliberately not part of normal automated tests.

## Manifest and review boundary

Every pack contains `manifest.json` with status `ready_for_review`. It records voice, library,
model and source versions; reference selection; exact chunk text and cache metadata; output
hashes; durations; warnings; generation time; and real-time factor. Phase 1 does not approve or
publish packs. A later control service will set `approved_at` and change status to `approved`
without changing story/audio entries.

## Researcher control service

Phase 2 adds a local FastAPI and SQLite control service for consented voice uploads, background
generation jobs, progress tracking, browser-based listening review, approval, and approved-pack
downloads.

Run the complete browser workflow without a GPU:

```bash
CONTROL_TTS_ENGINE=fake story-control-service
```

For actual Qwen generation, use an environment with CUDA PyTorch and the `qwen` optional
dependencies, then set `CONTROL_TTS_ENGINE=qwen`. Open <http://127.0.0.1:8000>. The service binds
to localhost by default and only exposes packs to device routes after explicit approval.
Control-service generations default to `QWEN_MAX_NEW_TOKENS=600` and three attempts. A chunk that
reaches the limit is discarded and retried with a deterministic alternate seed instead of
allowing a noisy tail into a pack.

Single-word card audio uses a separate strict profile. The synthesizer receives terminal
punctuation, uses a 48-token ceiling, trims model-produced leading and trailing silence, and
requires no more than two seconds of active and final audio. Invalid candidates are retried with
alternate seeds. The `word-prompt-v2` cache version prevents older untrimmed word clips from being
reused; unchanged story chunks remain cacheable.

Use **Prepare reference** in the navigation to convert an MP3, M4A, WAV, or other supported audio
file into a downloadable Qwen bundle. The bundle contains a mono 24 kHz PCM `reference.wav` and
the matching `reference.txt`; temporary uploads are removed after the download is served.

See [`docs/phase-2-control-service.md`](docs/phase-2-control-service.md) for configuration, API,
privacy behavior, and the local worker boundary.

## Export for offline Raspberry Pi playback

Export one compatible pack per voice into a self-contained directory containing only final WAVs,
public manifests, and voice-selection metadata:

```bash
story-pipeline export-pi \
  --pack data/packs/pack_for_mum \
  --pack data/packs/pack_for_dad \
  --selected-voice mum \
  --output-dir data/pi-exports/two-voice-study
```

The Pi needs no GPU or network connection after transfer. A dependency-free Go validation and
playback tool is provided in `pi/story-player`. See
[`docs/pi-offline-bundles.md`](docs/pi-offline-bundles.md) for the bundle layout, prototype export
option, transfer commands, voice switching, and Pi test procedure.

## Development checks

```bash
pytest
ruff check .
```

Known limitations: the Phase 2 service has no production authentication, HTTPS, CSRF protection,
Pi synchronisation package, ASR quality scoring, automatic translation, or story generation. Its
local worker creates an engine per job; a persistent HTTP GPU worker is Phase 3. Peak
normalization is used instead of perceptual loudness normalization, and duration checks are
warnings based on broad heuristics.

### GPU batch generation

Set `QWEN_BATCH_SIZE=2` (or `4`), or pass `--qwen-batch-size 2` to
`story-pipeline generate`. The default is `1`. The control service also reads
`QWEN_BATCH_SIZE`. Environment files must be loaded by your shell/service.
On the production Nectar Tesla T4, **keep `QWEN_DTYPE=float32` and use
`--dtype float32`**; batching does not change model precision.

One loaded Qwen model processes lists of uncached story chunks, grouped within
one story and output/reference language pair. Word clips retain their individual
candidate retries. Failed batches split into smaller batches, down to individual
chunks with the existing generation retries. Valid chunks are cached immediately;
an irrecoverable chunk marks the run failed, and `--resume` reuses saved chunks.

Batch size is excluded from cache keys and resume fingerprints, so existing cache
entries remain reusable when tuning it. Sampling can produce different audio for
fresh generations at different batch sizes; byte-identical Qwen output across
batch sizes is not promised. Manifests record the requested size, and new chunk
metadata records batch ID, actual size, position, total wall time, and an equal
share of that time as `generation_seconds`. `batch_finished` events in
`jobs/<run-id>/logs/generation.jsonl` include failed attempts too; sum their
`batch_wall_seconds` for total story inference time including fallback overhead.

Compare uncached sizes 1, 2, and 4 on the GPU:

```bash
story-pipeline benchmark \
  --reference-audio reference.wav \
  --reference-transcript-file reference.txt \
  --text-file story_library/stories/forest/en.txt \
  --max-chunk-characters 80 \
  --dtype float32 --max-new-tokens 600 \
  --output-dir data/benchmark
```

Use enough text for at least four chunks. The command loads one model, prepares
one English reference, warms up each batch size, then measures the same text
without cache reuse. JSON output and `benchmark.json` report wall seconds,
total audio seconds, effective RTF (wall/audio; lower is better), and peak CUDA
allocated bytes including model memory. CUDA is synchronized around measurements;
loading, prompt preparation, warmup and validation are excluded. Benchmark failures
are reported without silently falling back, so a size that exceeds GPU memory is
visible. Failed sizes produce exit code 2, and other sizes are still attempted.

For three isolated full-service batch-4 trials on the T4, use the
[batch-4 validation kit](validation/batch4/README.md). It includes a separate
long-story library, a fresh-service launcher, and a log summary command.
