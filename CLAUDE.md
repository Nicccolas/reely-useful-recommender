# Reel Folder Recommender

Rank a user's **existing, arbitrarily named folders** by how well a short-form video (Reel) fits each one. Output: ordered list of `(folder, score)`. Offline only, using local videos and simulated users. No Instagram integration.

Target interface: `rank_folders(video="reel_001.mp4", folders=[...]) -> [("Gym", 0.91), ...]`

Full spec: `docs/SPEC.md`. Read the relevant section before starting a new experiment, modality, or design decision.

## Core principles
- It's **ranking**: `score(reel, folder[, user/context])`, then sort. It is NOT fixed-label classification. Folder names are user-defined and dynamic.
- Start with frozen pretrained models and simple baselines (recency, frequency, TF-IDF, cosine similarity of embeddings). Fine-tune (contrastive/pairwise/LoRA) only when data and results justify it.
- Keep modalities separate at first (transcript, visual video, raw audio) so each one's contribution can be measured. Fusion method is an open empirical question.
- Video-language encoders (InternVideo2, LanguageBind_Video, X-CLIP) usually see **visual frames only, not audio**. Check each checkpoint.
- ASR (Whisper) → transcript → text embedding is a key baseline. Transcripts must be auto-generated, never hand-written.
- Later: represent folders by title + embeddings of Reels already in them (personalization), plus behavioral signals like recency, frequency, and size.
- Evaluate with ranking metrics: Hit@K, MRR, Recall@K, NDCG. Not just accuracy.

## Learning project
The user is building this to learn ML and wants to follow along and understand every step.
- Work in small steps the user can follow. Don't produce large finished pipelines in one go.
- Briefly explain the *why* behind each choice (model, metric, method), including trade-offs and alternatives.
- Keep code readable over clever. Add short comments on non-obvious ML concepts (e.g. why normalize embeddings before cosine similarity).
- After results come in, help interpret them: what they show, what's surprising, what to try next.
- Leave room for the user to write or attempt parts themselves when they want to.

## Workflow
- Experiment in Jupyter notebooks first. Move stable code into `src/` later.
- Grow the repo structure as needed. Don't scaffold `data/ notebooks/ src/ models/ app/ tests/` upfront.
- Never commit raw videos or private/copyrighted media. Keep paths and config out of tracked code.
- Demo (Gradio/Streamlit) and API (FastAPI) come later and are optional.

## Current milestone
Given a real video and a set of arbitrary folder names, produce a sensible ranking.
