# Cached Qwen3-TTS story-pack generator

This repository generates multilingual story and card-word audio with Qwen3-TTS, provides a
browser workflow for voice uploads and audio review, and exports approved packs for offline
Raspberry Pi playback. Generated chunks are cached by their inputs, and packs include
reproducibility metadata.

The included library contains seven short stories in English, Mandarin Chinese, Japanese,
Korean, German, Portuguese, and Spanish. Each story generates a full-story WAV and a
separate card-word WAV in each selected language: 98 files when all seven are selected.
The added translations follow the current English text, including the short presentation
message in `bear/en.txt`; Portuguese uses Brazilian wording.
All audio should be reviewed before use as research study content.

## Simplest workflow: GPU → phone → Raspberry Pi

**Generate and approve on the GPU server, export a bundle, copy it through your Mac to the Pi,
then play it locally. Downloads to the Pi are not automatic.** Once the files are on the Pi,
playback works offline; the phone, Mac, and GPU can disconnect.

These instructions use the existing GPU server at `ubuntu@130.216.216.156`, its service on
port **8000**, and your Mac's forwarded port **8001**. Replace `RPI_USER@RPI_HOST` below with
your Pi's SSH login, for example `jaybird@raspberrypi.local`. Replace `pack_REPLACE_ME` with
the pack ID shown on the review page.

### 1. Check or start the GPU service

**On your Mac**, connect to the GPU server:

```bash
ssh -i ~/.ssh/nectar_gpu ubuntu@130.216.216.156
```

**In that GPU terminal**, check whether the service is already running:

```bash
curl -fsS http://127.0.0.1:8000/api/v1/voices
```

If it returns the saved voices, skip starting it again. If the connection is refused, start it:

```bash
cd ~/qwen-pipeline
source .venv/bin/activate
CONTROL_TTS_ENGINE=qwen \
QWEN_DTYPE=float32 \
QWEN_BATCH_SIZE=4 \
CONTROL_HOST=127.0.0.1 \
CONTROL_PORT=8000 \
CONTROL_DATA_ROOT=./data \
DATABASE_URL=sqlite:///./data/control.db \
story-control-service
```

This uses the server's existing Qwen/CUDA environment. Keep this terminal open while generating.
Use another SSH terminal for the export commands later. If SSH needs the key unlocked, run
`ssh-add ~/.ssh/nectar_gpu` on your Mac and enter its passphrase there.

### 2. Open the website on your Mac or phone

**In a separate Mac terminal**, open the tunnel and leave it running:

```bash
ssh -N -o ExitOnForwardFailure=yes -i ~/.ssh/nectar_gpu \
  -L 0.0.0.0:8001:127.0.0.1:8000 ubuntu@130.216.216.156
```

- **Mac:** open <http://localhost:8001>.
- **Phone:** join the same Wi-Fi as the Mac. Find the Mac's Wi-Fi IP in
  **System Settings → Wi-Fi → Details → TCP/IP**, then open
  `http://MAC_WIFI_IP:8001`, for example `http://192.168.1.42:8001`.

The phone connects to the **Mac's address**, not `localhost` or the GPU's address. Keep the Mac
awake. Use trusted Wi-Fi: the forwarded website has no login. If the tunnel says the port is in
use, close the previous tunnel or local service on port 8001. If only the phone cannot connect,
check the Mac firewall and whether the Wi-Fi blocks communication between devices.

### 3. Create a voice, generate, and approve

In the browser:

1. For a new voice, open **Add voice**, choose two different languages, and record yourself reading
   each displayed passage exactly. Upload the recordings, confirm consent, then **Create voice**.
   M4A, MP3, and WAV are supported; the GPU server needs FFmpeg for formats it cannot decode directly.
   You can skip this step and select an existing saved voice.
2. Open **Generate pack**. Choose your voice and stories. Its recorded languages are selected
   automatically. For an English-only voice generating Mandarin or Spanish, check
   **Enable cross-language generation**, choose English as the fallback, and select the output
   languages. A matching recording always takes priority when one exists.
3. Click **Start generation**. Wait for **READY_FOR_REVIEW**, then open the review page.
4. Listen to every story and word clip and click **Approve pack**. Copy the `pack_…` ID shown there.

Generation continues on the server if you close the phone page. Keep the server process running.
Seven stories in two languages produce **28 WAV files**: one story and one word clip per language.
An upload marked “Recording accepted” confirms usable audio, not that the passage was read correctly.

### 4. Export the approved pack on the GPU

**In a GPU terminal**, replace the example ID and run:

```bash
cd ~/qwen-pipeline
source .venv/bin/activate
PACK_ID=pack_REPLACE_ME
story-pipeline export-pi \
  --pack "data/packs/$PACK_ID" \
  --output-dir "data/pi-exports/$PACK_ID"
```

The exporter checks approval and WAV checksums. With one pack, its voice is selected automatically.
The output includes final audio and player metadata; it excludes private voice recordings,
reference transcripts, models, and caches. The export directory must be new. If you already
exported this pack successfully, use that existing export for the next step.

**Prepare reference** downloads a voice-reference ZIP, not a story pack. There is currently no
one-click Pi-bundle download in the browser; use `export-pi` and the transfers below.

### 5. Download to your Mac, then copy to the Pi

**In a Mac terminal**, use the same pack ID:

```bash
PACK_ID=pack_REPLACE_ME
mkdir -p "$HOME/Downloads"
scp -i ~/.ssh/nectar_gpu -r \
  "ubuntu@130.216.216.156:/home/ubuntu/qwen-pipeline/data/pi-exports/$PACK_ID" \
  "$HOME/Downloads/"

ssh RPI_USER@RPI_HOST 'mkdir -p ~/story-bundles'
scp -r "$HOME/Downloads/$PACK_ID" RPI_USER@RPI_HOST:~/story-bundles/
```

For the **first setup**, also copy the standalone player source from your Mac checkout:

```bash
cd /Users/jaybird/p4p/qwen-pipeline
scp -r pi/story-player RPI_USER@RPI_HOST:~/
```

The Pi only needs the exported bundle and this player. It does not need Python, Qwen, or your GPU
SSH key. For a new generated pack, repeat steps 4–5 with its new ID; existing bundles stay separate.

### 6. Build and play on the Pi

**SSH into the Pi**:

```bash
ssh RPI_USER@RPI_HOST
```

On Raspberry Pi OS, install the build/playback tools once. The player requires **Go 1.20 or newer**:

```bash
sudo apt update
sudo apt install -y golang-go alsa-utils
go version
```

Build once, then verify and play the copied bundle:

```bash
cd ~/story-player
go build -o story-player .
PACK_ID=pack_REPLACE_ME
./story-player -bundle "$HOME/story-bundles/$PACK_ID" verify
./story-player -bundle "$HOME/story-bundles/$PACK_ID" list
./story-player -bundle "$HOME/story-bundles/$PACK_ID" play-word forest en
./story-player -bundle "$HOME/story-bundles/$PACK_ID" play forest en
```

Use a story and language actually included in your pack. For example, replace `en` with `zh` for
Mandarin. If verification succeeds but there is no sound, check the Pi's speaker connection and
selected ALSA output (`aplay -l` lists playback devices).

The included player is a command-line tool: it does not start at boot or listen for NFC cards.
Your NFC application must call `play-word` for the scanned trigger; a story button can call `play`
for the same trigger. See [Pi integration details](docs/pi-offline-bundles.md).

### What is automatic?

| Step | Current behavior |
| --- | --- |
| Convert uploaded voice recordings | Automatic during upload validation/voice creation |
| Generate stories and word clips | Automatic after **Start generation** |
| Approve audio | Manual listening and **Approve pack** |
| Export a Pi bundle | Manual `story-pipeline export-pi` command |
| Download/install on the Pi | Manual `scp` transfers; approval does not push files |
| Pick up newer packs | Manual export and transfer, then use the new bundle path |
| Validate/play a local clip | Player checks its checksum and invokes `aplay` when called |
| Start on boot / react to NFC | Not supplied by this standalone player |

The service has approved-manifest and audio-file API routes, but the Pi player does not poll them
or sync automatically. Device-specific pack assignment is not implemented either. For multiple
voices in one offline bundle, see [multi-voice export](docs/pi-offline-bundles.md#add-multiple-voices).

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
preserve word spaces. The browser service prefers matching language references and supports an
explicit cross-language fallback. The standalone CLI uses its English reference for cross-language
cloning unless a dedicated Mandarin reference is supplied.

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
hashes; durations; warnings; generation time; and real-time factor. The generation CLI does not approve or
publish packs. Browser approval in the control service sets `approved_at` and changes status to
`approved` without changing story/audio entries.

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

One loaded Qwen model processes uncached story chunks **across stories**. The job
plans all story chunks and checks the cache before synthesis. Compatible chunks
share output language, reference language and audio/transcript hashes, model/revision,
dtype, and generation parameters. Request order determines group and chunk order;
finished chunks are assembled back into their original stories without changing
chunk boundaries. Four uncached one-chunk English stories can therefore use one
batch of four. English, Spanish, and Japanese are always separate groups.
Identical cache keys within the same job are synthesized once and reused at each
story destination. Word clips retain their separate token limits and individual
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
`batch_started` and `batch_finished` include configured/actual size, story IDs,
chunk indices, language, and reference language. `batch_fallback` links the failed
batch to its smaller retries via `parent_batch_id`; `chunks_planned` records initial
cache hits and unique misses. Chunk completion events are emitted as soon as each
validated chunk is cached and materialized, so service progress advances during synthesis.

Manifest/request `timing` and the `run_completed`/`run_failed` log events report
`story_batch_seconds` (all attempts, including failures), `word_generation_seconds`
(including candidate preparation/retries), their sum as `generation_wall_seconds`,
and `run_wall_seconds` (the pipeline call including planning and assembly, excluding
model loading in the service). These job totals exclude historical inference time
for cache hits. Per-chunk timings remain the original equal share of their successful
batch, and cached chunks retain their historical metadata.

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
