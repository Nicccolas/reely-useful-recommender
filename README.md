# Reel Folder Recommender

A machine-learning project that **ranks a user's own folders by how well a short-form video (e.g. an Instagram Reel) fits each one**, instead of sorting them by recency of use.

Given a video and a user's folder names, the goal is an ordered list:

```text
Reel: deadlift tutorial
1. Gym                  0.91
2. Career               0.18
3. Funny                0.12
...
```

This is a personal learning project. It runs offline on local videos and simulated users, with no Instagram integration.

## Approach

Folder names are created by users, so a fixed set of topic labels doesn't apply. The problem is framed as **ranking**: score each `(reel, folder)` pair, then sort.

The first baseline is zero-shot, with no training:

```text
video -> audio -> Whisper -> transcript -> text embedding --\
                                                             cosine similarity -> ranked folders
folder name ------------------------------> text embedding --/
```

Later experiments compare other signals (visual video embeddings, raw audio embeddings, folder contents, usage history) against this baseline using ranking metrics (Hit@K, MRR). The full design is in [docs/SPEC.md](docs/SPEC.md).

## Status

| Step | State |
|---|---|
| Simulated users and labels | Done |
| Audio extraction and Whisper transcription | Done (`notebooks/01_transcribe.ipynb`) |
| Transcript and folder-name embeddings, ranking | Next |
| Evaluation (Hit@K, MRR, random and TF-IDF baselines) | Planned |

No ranking results yet.

### Transcription notes

- Clips of 30 s or less go through the Hugging Face ASR pipeline. Longer clips use Whisper's native long-form generation (`model.generate` with timestamps). `whisper-small` is used for both.
- Neither method was best on every clip: the pipeline with `chunk_length_s` dropped the start of one clip, and long-form with timestamps looped on another.
- Clips with no speech (music, crowd noise) produce stock phrases, repeated text or sound tags. These transcripts are kept unfiltered on purpose, so the ranking's behaviour on them can be observed.

## Data

The videos are not included in this repository (copyrighted and large).

- `data/videos.csv`: one row per video, with a description, a topic and `speech` / `visual` / `text` flags. Each flag is 1 if that signal alone would help identify the right folder.
- `data/users.json`: three simulated users with different folder styles (4, 7 and 5 folders): broad names, fine-grained names and vague personal names.
- `data/labels.csv`: for each video and user, the correct folder, plus any other acceptable folders.

To reproduce, add your own `.mp4` files as `data/raw/reel_001.mp4` and so on, and edit the CSV and JSON files to match.

## Setup

Requires Python 3.11 and ffmpeg. On Apple Silicon the notebooks use the `mps` GPU device.

```bash
conda create -n reel-recs python=3.11 -y
conda activate reel-recs
conda install -c conda-forge ffmpeg -y
pip install -r requirements.txt
python -m ipykernel install --user --name reel-recs
jupyter lab
```

Select the `reel-recs` kernel in the notebook. Whisper and the embedding models download from Hugging Face on first use.

## Repository layout

```text
CLAUDE.md          project context for Claude Code
docs/SPEC.md       full project specification
data/              videos.csv, users.json, labels.csv (media and transcripts are git-ignored)
notebooks/         experiments, run in numbered order
requirements.txt   pinned dependencies
```

Stable code will move into `src/` once the pipeline settles.
