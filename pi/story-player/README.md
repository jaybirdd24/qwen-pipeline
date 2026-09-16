# Offline Pi story player

This dependency-free Go program validates and plays audio from an exported Pi bundle. It does not
contact the control service, Qwen, Gemini, or a GPU.

Build it on the Raspberry Pi:

```bash
cd story-player
go build -o story-player .
```

Assuming the exported bundle was copied to `/var/lib/story-player`:

```bash
./story-player -bundle /var/lib/story-player verify
./story-player -bundle /var/lib/story-player list
./story-player -bundle /var/lib/story-player resolve forest en
./story-player -bundle /var/lib/story-player play forest en
./story-player -bundle /var/lib/story-player resolve-word forest en
./story-player -bundle /var/lib/story-player play-word forest en
./story-player -bundle /var/lib/story-player select caregiver_sample
```

`play` plays the full story, while `play-word` plays only the localized card word. Both validate the
selected WAV's SHA-256 before invoking `aplay`. The `select` command changes only
`player-state.json`; voice packs remain immutable. The NFC state machine should invoke `play-word`
when a card is scanned.

Language arguments use the codes in the exported pack, including `en`, `zh`, `ja`,
`ko`, `de`, `pt`, and `es` (and `fr` for libraries with French). Both story and word
commands check that the selected language is present in the pack.
