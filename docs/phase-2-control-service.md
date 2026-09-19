# Phase 2 Control Service

## Purpose

Phase 2 wraps the Phase 1 generator in a local researcher-facing FastAPI application. It stores
voices, generation jobs, packs, and per-file metadata in SQLite while keeping reference audio and
generated files in the existing local storage layout.

The control service is deliberately local-first and binds to `127.0.0.1` by default. It does not
yet provide production authentication or HTTPS.

## Run locally

Install the project and development dependencies:

```bash
python -m pip install -e '.[dev]'
```

Start with the fake engine, which does not need a GPU:

```bash
CONTROL_TTS_ENGINE=fake story-control-service
```

Open <http://127.0.0.1:8000>. The workflow is:

1. Optionally use **Prepare reference** to convert an audio file and exact transcript into a
   downloadable Qwen-ready WAV/text bundle.
2. Add a participant voice: choose two distinct languages and upload a recording of each displayed passage.
3. Choose stories. The participant’s recorded languages are selected automatically.
4. Start generation and monitor chunk progress.
5. Listen to every generated file on the review page.
6. Approve or reject the pack.
7. Only an approved pack becomes available through device download routes.

## Languages

The control service supports English (`en`), Mandarin Chinese (`zh`), Japanese (`ja`),
Korean (`ko`), German (`de`), Portuguese (`pt`), Spanish (`es`), and French (`fr`).
The bundled library includes all of these except French. The generation form enables
only languages for which the active story library has a complete translation. To enable one, add its code to `required_languages` in
`library.yaml`, add its title and `spoken_word` to every `story.yaml`, and add a non-empty
`<code>.txt` to every story
directory. An API request for an incomplete language is rejected before a GPU job starts.

Every control-service job requires a reference in each requested language. Spanish generation
requires a Spanish recording; there is no English fallback. If `languages` is omitted from a job
request, all languages recorded for that voice are used. An explicit empty selection is rejected.

Voices have any number of language references internally; the participant form requires exactly
two. New recordings live under `references/<voice_id>/v1/<language>/reference.wav` and
`reference.txt`. Each reference stores its language, transcript, duration, and normalized audio
SHA-256 in `voice_references`. Startup adds this table and backfills existing English/Mandarin
voices idempotently without moving files. Legacy columns and upload fields remain for compatibility.
The standalone CLI retains its legacy cross-language cloning behavior; use the control service
for participant study generation.

Fixed passages for all supported languages are defined in
`src/story_voice_pipeline/control_service/passages.py`. The server supplies the transcript;
participants never type one. Review these passages before collecting study recordings. Stored
transcripts are snapshots, so future passage edits do not alter existing voices.
The upload check validates decoding and non-silent audio, reports duration warnings, and converts
to mono 24 kHz WAV with the default configuration. It does not recognize speech or verify that the
passage was read correctly. Phone formats that libsndfile cannot decode require FFmpeg.

For real Qwen generation, use an environment containing compatible CUDA-enabled PyTorch and the
project's `qwen` optional dependencies:

```bash
python -m pip install -e '.[qwen]'
CONTROL_TTS_ENGINE=qwen story-control-service
```

Qwen generation is capped at 600 new tokens per chunk by default to prevent non-terminating
generations. Output that reaches the cap is discarded and retried up to three times with
deterministic alternate seeds. Override these controls with `QWEN_MAX_NEW_TOKENS` and
`QWEN_MAX_GENERATION_ATTEMPTS` if required.

Card-word assets have an independent two-second quality boundary. They use punctuated synthesis
text and a 48-token cap, then undergo energy-based silence trimming and peak normalization. A
candidate with less than 150 ms or more than two seconds of detected active audio is rejected and
retried. The original localized noun remains unchanged in the manifest as `source_text`.

The local processor serializes jobs so only one model-generation job runs at a time. It constructs
the engine for each job in Phase 2. Keeping Qwen resident in a separate worker is Phase 3.

## Configuration

Configuration is read from environment variables. `.env.example` lists all supported values.
Important defaults are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CONTROL_HOST` | `127.0.0.1` | Bind address |
| `CONTROL_PORT` | `8000` | HTTP port |
| `CONTROL_DATA_ROOT` | `data` | References, jobs, cache, packs, and default database root |
| `CONTROL_STORY_LIBRARY` | `story_library/library.yaml` | Active fixed library |
| `CONTROL_TTS_ENGINE` | `fake` | `fake` or `qwen` |
| `DATABASE_URL` | derived SQLite path | SQLAlchemy database URL |
| `CONTROL_MAX_UPLOAD_BYTES` | 50 MiB | Per-reference upload limit |

The service uses the same Qwen, chunking, silence, and seed environment variables documented in
`.env.example`.

## API

### Voices

```text
POST /api/v1/voices
GET  /api/v1/voices
GET  /api/v1/voices/{voice_id}
```

Voice creation is `multipart/form-data`: `name`, `consent_confirmed=true`, repeated `languages`
fields, and matching repeated `audios` files in the same order. One or more distinct supported
languages are accepted by the API. Transcripts come from the fixed passage catalog.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/voices \
  -F 'name=Participant 04' -F 'consent_confirmed=true' \
  -F 'languages=en' -F 'languages=es' \
  -F 'audios=@english.m4a' -F 'audios=@spanish.m4a'
```

Responses include `languages` and a `references` object keyed by language. Voice detail also
includes each stored transcript. For older clients, `english_audio`/`english_transcript` and optional
`mandarin_audio`/`mandarin_transcript` are still accepted together. These cannot be mixed with the
new repeated fields. They do not bypass the generation requirement for matching references.

`POST /api/v1/references/validate` accepts `language` and `audio`, checks the recording, returns
its duration and warnings, and removes temporary files. The participant form uses this endpoint
before showing “Recording accepted”; creation independently validates both recordings again.

### Languages

```text
GET /api/v1/languages
```

Returns the supported codes, display names, whether each has complete content in the active
library, and whether the target is experimental.

### Jobs

```text
POST /api/v1/jobs
GET  /api/v1/jobs/{job_id}
POST /api/v1/jobs/{job_id}/retry
POST /api/v1/jobs/{job_id}/cancel
```

Example creation body:

```json
{
  "voice_id": "voice_123",
  "story_ids": ["forest", "moon"],
  "languages": ["en", "zh"]
}
```

Omitting `story_ids` selects the complete library. Progress responses include state, current
story/language/chunk, completed chunks, total chunks, failure detail, and resulting pack ID.
Cancellation is checked at pipeline event boundaries; an inference call already in progress may
finish before cancellation is observed.

### Review and approval

```text
GET  /api/v1/story-packs
GET  /api/v1/story-packs/{pack_id}
POST /api/v1/story-packs/{pack_id}/approve
POST /api/v1/story-packs/{pack_id}/reject
```

Review audio is served through a separate local route. Approval atomically updates the manifest
to `approved`, records `approved_at`, and changes the database job and pack states. Rejected and
review-ready packs are never exposed through device routes.

### Device publication

```text
GET /api/v1/devices/{device_id}/latest-manifest
GET /api/v1/packs/{pack_id}/files/{file_path}
```

The manifest endpoint returns the most recently approved pack. File requests must match an exact
manifest audio path and pass traversal checks. Device-specific assignments are deferred until Pi
integration.

## Storage and privacy

- Uploaded source files are deleted after normalization.
- Normalized private references and transcripts remain beneath `data/references/`.
- References are never included in generated packs or device downloads.
- File names and IDs are generated server-side.
- Upload size, consent, transcript, and reference readability are validated.
- Pack files remain unavailable through public device routes until approval.
- `data/` and `ref/` are ignored by Git.

Production deployment still requires authentication, authorization, HTTPS, CSRF protection, and
a deliberate voice-deletion/retention policy.

## Tests

The integration suite runs the complete HTTP lifecycle with the fake engine, including upload,
generation, progress persistence, review playback, approval, publication, checksum verification,
rejection isolation, failure recording, and retry.

```bash
pytest
ruff check .
```

Normal automated tests do not load Qwen or require a GPU.
