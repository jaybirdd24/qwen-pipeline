# Batch-4 service validation on the T4

This is a separate test workload and launcher. It changes no pipeline code,
production library, or global defaults. The original bilingual lantern story has
8 distinct paragraphs in each language. Paragraph boundaries produce 8 chunks
per language at the existing 400-character limit, exercising two four-chunk
batches per language plus the normal individual word generation.

No GPU results are included. These trials must run on the GPU server. Passing
this fixture does not prove that every larger or longer production workload fits.

## Prepare

Pull this branch on the GPU server and activate the existing project environment:

```bash
source .venv/bin/activate
pip install -e .
python validation/batch4/trials.py preflight
```

Preflight must show 8 chunks for `lantern/en` and `lantern/zh`. The launcher pins
`QWEN_DTYPE=float32`, `QWEN_BATCH_SIZE=4`, `MAX_CHUNK_CHARACTERS=400`, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before starting Python. It also
sets the newer `PYTORCH_ALLOC_CONF` alias to the same value to avoid conflicting
inherited settings. Existing model, seed, token budget, attention, and generation
attempt settings are inherited. Without overrides the service uses its normal
defaults, including 600 maximum new tokens and 3 generation attempts.

Stop any other generation on the T4 before starting. Use the same reference WAVs
and exact transcripts in all three trials. Keep Mandarin reference presence
consistent too. Upload only references you have permission to use; confirm the
usual consent field in the UI yourself.

## Run three trials

On your Mac, forward the isolated service port using your existing GPU SSH target:

```bash
ssh -i ~/.ssh/nectar_gpu -L 8001:127.0.0.1:8001 ubuntu@130.216.216.156
```

On the GPU server, run these **one at a time**, completing and stopping each
service with Ctrl-C before starting the next:

```bash
python validation/batch4/trials.py start --trial 1
python validation/batch4/trials.py start --trial 2
python validation/batch4/trials.py start --trial 3
```

For each trial:

1. Open `http://localhost:8001` on your Mac. Record the displayed English and Mandarin
   passages, then reuse those same two recordings in every trial. Both references are required.
2. Generate the lantern story in **both English and Mandarin**, once. Wait for
   the job to finish or fail. Do not submit another job or retry in that trial.
3. Review the generated story and word audio in the UI before stopping the server.
   Compare narration with `library/stories/lantern/en.txt` and `zh.txt`. Check
   paragraph order, missing/repeated words, truncation, noisy endings, and word clips.
4. Record listening findings for each language and trial in a local review file,
   including pass/fail and timestamps of defects. Audio review is never marked
   passed automatically.
5. Stop the service with Ctrl-C, then start the next trial.

In another GPU terminal, monitor memory:

```bash
watch -n 1 nvidia-smi
```

For a persistent memory trace during all three trials, optionally run:

```bash
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
  --format=csv -l 1 > /tmp/batch4-gpu.csv
```

Stop that monitor after the trials. This reports device-wide memory, not PyTorch
peak allocated memory. Each trial's `service.log` also captures allocator warnings
and model errors; check for `expandable_segments not supported` warnings.

Each launch creates a fresh process and an exclusive directory under
`data/batch4-validation/trial-N`, with an explicitly isolated database. It refuses
to overwrite even an empty prior trial directory. For a new experiment, use the
same new `--root data/batch4-validation-repeat` on all three launch commands and
the summary command. Never delete production caches to force a test.

## Summarize and decide

After stopping trial 3:

```bash
python validation/batch4/trials.py summarize > data/batch4-validation/summary.json
cat data/batch4-validation/summary.json
```

The summary requires exactly one generation per trial, matching source/reference
hashes and generation settings, Qwen float32 at batch size 4, no cache hits, the
full expected story chunk count, and successful four-chunk batches in both
languages. Missing/failed/partial trials cannot pass. Exit code 1 means the
three-trial stability check did not pass; the JSON still contains the evidence.

- `end_to_end_wall_seconds` uses service job timestamps, including model loading,
  reference preparation, story/word generation, validation, and pack assembly.
  It excludes service startup, upload, and queue waiting.
- `batch_wall_seconds` sums **all** story `batch_finished` events, including failed
  attempts and fallback. It excludes word generation and other job overhead.
- `fallback_recovered` means a run completed despite batch errors/rejections. This
  is separate from `stability_pass`, which requires no errors or rejections.
- `successful_four_chunk_batches`, `batch_sizes`, `failed_batches`, `oom_batches`,
  and `cache_hits` make single-generation or fallback-only runs visible.

Keep batch 4 experimental until all three stability checks and your listening
reviews pass. If OOM fallback recurs, use `QWEN_BATCH_SIZE=2` for normal service
operation while investigating. The launcher does not update production settings,
and no speed or audio-quality claim follows automatically from a completed run.
