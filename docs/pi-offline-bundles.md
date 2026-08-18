# Offline Raspberry Pi bundles

## Purpose

An offline Pi bundle contains reviewed story WAVs, public pack manifests, a voice catalog, and the
current voice selection. It deliberately excludes caregiver reference recordings, transcripts,
Qwen model files, generation caches, jobs, and database files. Once exported, playback requires
neither the generation computer nor a GPU connection.

## Export a prototype

Export the existing reviewed Qwen pack from the repository root:

```bash
story-pipeline export-pi \
  --pack data/packs/pack_477f50c0187d46d3 \
  --output-dir data/pi-exports/caregiver-sample-prototype \
  --selected-voice caregiver_sample \
  --allow-review-ready
```

`--allow-review-ready` exists only for prototype hardware testing. Normal exports reject any pack
that has not been approved through the control service.

An output directory must be new. This prevents a partial export or stale files from being mistaken
for a complete bundle. Export a new directory when adding or replacing voices.

## Add multiple voices

Generate and review the same library for each voice, then provide one pack per voice:

```bash
story-pipeline export-pi \
  --pack data/packs/pack_for_mum \
  --pack data/packs/pack_for_dad \
  --pack data/packs/pack_for_researcher \
  --selected-voice mum \
  --output-dir data/pi-exports/three-voice-study
```

The exporter requires every pack to use the same library version, story IDs, trigger objects, and
languages. It verifies every source WAV checksum, copies only final audio, verifies each copy, and
then atomically exposes the completed output directory.

## Bundle layout

```text
three-voice-study/
├── catalog.json
├── player-state.json
└── packs/
    ├── mum/
    │   └── pack_for_mum/
    │       ├── manifest.json
    │       ├── forest/en.wav
    │       ├── forest/zh.wav
    │       ├── forest/word/en.wav
    │       ├── forest/word/zh.wav
    │       └── ...
    └── dad/
        └── pack_for_dad/
            ├── manifest.json
            └── ...
```

`catalog.json` describes the installed voices. `player-state.json` holds the selected voice and
the active immutable pack for every installed voice. Switching voices changes only the state file.

## Copy and test on the Pi

Copy the complete bundle and the standalone Go player source:

```bash
rsync -av --progress \
  data/pi-exports/caregiver-sample-prototype/ \
  pi@PI_HOST:/var/lib/story-player/

rsync -av pi/story-player/ pi@PI_HOST:~/story-player/
```

On the Pi:

```bash
cd ~/story-player
go build -o story-player .
./story-player -bundle /var/lib/story-player verify
./story-player -bundle /var/lib/story-player list
./story-player -bundle /var/lib/story-player play forest en
./story-player -bundle /var/lib/story-player play-word forest en
./story-player -bundle /var/lib/story-player play-word forest zh
```

For multiple voices:

```bash
./story-player -bundle /var/lib/story-player select dad
./story-player -bundle /var/lib/story-player play-word forest en
```

The player resolves the selected voice, active pack, NFC trigger, language, and requested audio type
through the local JSON files. `play` selects the full story; `play-word` selects the short localized
card word. It validates the selected WAV immediately before calling `aplay`.

## Existing Go application integration

The standalone player is the first hardware checkpoint. Once its `verify`, `resolve`, `play`, and
`play-word` commands work on the Pi, its dependency-free resolver can be moved into the existing Go
program. The NFC loop can invoke `play-word` for triggers such as `forest`, `moon`, `dad`, and
`bear`; the story button can invoke `play` for the same trigger.
