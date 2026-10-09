# Reel Folder Recommendation System

## Project Overview

The goal of this project is to build a machine-learning system that recommends and ranks user-created folders when saving a short-form video such as an Instagram Reel.

The idea is motivated by Instagram's current save workflow. When a user saves a Reel, their existing folders/collections appear as a list. Currently, these folders may be ordered largely according to recency of use. The proposed system should instead rank the folders according to how relevant each folder is to the content of the Reel being saved.

For example, suppose a user has the following folders:

- Recipes
- Gym
- Travel
- Machine Learning
- Career
- Funny

If the user attempts to save a Reel about a deadlift tutorial, the system might return:

```text
1. Gym                  0.91
2. Career               0.18
3. Funny                0.12
4. Travel               0.07
5. Recipes              0.04
6. Machine Learning     0.03
```

The intended output is therefore an **ordered list of the user's existing folders**, not a single predefined topic classification.

---

# Core ML Problem

This should primarily be treated as a **ranking/recommendation problem**, rather than conventional multiclass classification.

A traditional classifier assumes a fixed set of labels:

```text
video → {sports, food, travel, politics, ...}
```

That assumption does not fit this application because folder names are created dynamically by individual users.

Different users could have completely different folder structures:

```text
User A:
- Food
- Workouts
- Memes

User B:
- Healthy Meals
- Italian Recipes
- Running
- Weightlifting
- Things to Send Sarah
```

The more appropriate formulation is:

```text
score(reel, candidate_folder [, user/context])
```

followed by:

```text
rank(folder_1, folder_2, ..., folder_n)
```

The model should therefore estimate how well a Reel belongs in each candidate folder and sort the folders accordingly.

---

# Important Product Constraint

This project will **not interact directly with Instagram**.

The initial system should operate offline on locally available short videos and simulated user folder structures.

A minimal system should be able to perform something conceptually like:

```python
rank_folders(
    video="reel_001.mp4",
    folders=[
        "Recipes",
        "Gym",
        "Travel",
        "Machine Learning",
        "Funny"
    ]
)
```

and return:

```python
[
    ("Gym", 0.91),
    ("Recipes", 0.31),
    ("Funny", 0.18),
    ("Travel", 0.10),
    ("Machine Learning", 0.05)
]
```

The ML/recommendation problem is the primary goal. A UI or Instagram-like demo can be added later if useful.

---

# Development Philosophy

The project should begin experimentally, likely in Jupyter notebooks.

The goal at the beginning is **not** to prematurely design a production architecture.

Use notebooks to:

- understand the data;
- experiment with representations;
- inspect embeddings;
- test similarity and ranking approaches;
- compare models;
- determine which video modalities are useful;
- develop evaluation metrics;
- establish baselines.

Once parts of the pipeline become stable and reusable, they can be refactored into Python modules under something like `src/`.

Possible future repository structure:

```text
reel-folder-recommender/
│
├── data/
├── notebooks/
├── src/
├── models/
├── outputs/
├── app/
├── tests/
├── requirements.txt
└── README.md
```

Do not create this entire structure prematurely. Let the repository grow as the project becomes clearer.

---

# Modalities Available in a Reel

A short-form video contains several different sources of semantic information:

```text
                         REEL
                           |
          +----------------+----------------+
          |                |                |
       visuals          speech        other audio
          |                |                |
   video embedding    transcript       audio embedding
                           |
                     text embedding
```

These modalities should initially be treated separately so their contribution can be evaluated.

## 1. Spoken Content / Transcript

One promising representation is:

```text
video
  ↓
extract audio
  ↓
automatic speech recognition
  ↓
transcript
  ↓
text embedding
```

This is likely to be particularly useful for talking-head Reels where the visual appearance says little about the topic.

Example:

```text
Visual:
Person sitting at desk

Speech:
"Here are three mistakes people make when training a random forest."
```

The visual information mostly indicates "person / desk / talking," while the transcript contains strong semantic information about machine learning.

The transcript should be generated automatically rather than manually summarized when evaluating the system.

### Recommended ASR models

A strong starting point is:

- `openai/whisper-large-v3-turbo`

Alternatives include:

- `openai/whisper-large-v3`
- smaller Whisper variants if local compute becomes an issue.

Whisper can be loaded through Hugging Face Transformers.

---

# Text Embeddings

Once speech has been transcribed, a text embedding model can create a semantic vector for the transcript.

The same or compatible text representation can be used for folder names.

Conceptually:

```text
Transcript:
"Five exercises for increasing shoulder strength"

       ↓ text encoder

reel_text_embedding


Folder:
"Gym"

       ↓ text encoder

folder_embedding


       ↓

cosine similarity
```

Potential starting models include:

### Lightweight baseline

```text
sentence-transformers/all-MiniLM-L6-v2
```

Useful because it is small, fast, and easy to work with while developing the pipeline.

### Stronger traditional SentenceTransformer baseline

```text
sentence-transformers/all-mpnet-base-v2
```

### More modern embedding model worth experimenting with

```text
Qwen/Qwen3-Embedding-0.6B
```

Other retrieval-oriented embedding models such as BGE models may also be worth comparing.

The exact embedding model should be treated as an experimental choice rather than fixed upfront.

---

# Direct Video Embeddings

The project should eventually test models that create semantic representations directly from the visual video stream.

This is different from independently embedding individual frames with an image model.

A true video encoder can model temporal information across frames:

```text
frame 1 → frame 2 → frame 3 → frame 4
                    |
              video encoder
                    |
              video embedding
```

Potential models include:

## InternVideo2

InternVideo2 is a pretrained video foundation model family designed for video understanding and video-language alignment.

Relevant checkpoints include InternVideo2 video-text / CLIP-style models available through OpenGVLab.

This is a strong candidate for testing:

```text
video → video embedding
folder title → text embedding
            ↓
      semantic similarity
```

Important: the standard video-text InternVideo2 embedding pipeline primarily represents the **visual video stream**. Do not assume the video's speech/audio is automatically represented. There are separate audiovisual variants of the InternVideo2 architecture, but this should be checked for the particular checkpoint being used.

## LanguageBind

Potential checkpoint:

```text
LanguageBind/LanguageBind_Video
```

LanguageBind aligns multiple modalities with language in related embedding spaces.

The video encoder should be understood as representing the video's **visual stream**, not automatically incorporating the audio track.

LanguageBind also has separate encoders for modalities such as audio.

This makes it potentially useful for experiments comparing:

```text
video ↔ language
audio ↔ language
```

## X-CLIP

X-CLIP is another possible video-text baseline, especially if a relatively straightforward Transformers-based implementation is useful.

It may be helpful as a simpler comparison against larger video foundation models.

---

# Important Distinction: Video vs Audio

Do not assume that a model called a "video-language model" necessarily uses the audio contained inside an `.mp4`.

For models such as the standard LanguageBind video encoder and common InternVideo2 video-text configurations:

```text
.mp4
 ↓
visual frames
 ↓
video encoder
 ↓
video embedding
```

The audio track is not necessarily incorporated.

Therefore:

```text
video embedding ≠ complete representation of everything in the Reel
```

This matters because short-form videos often contain critical semantic information in spoken language.

---

# Direct Audio Embeddings

A possible later experiment is to represent the raw audio directly rather than converting speech into text first.

Potential model family:

```text
CLAP
```

For example:

```text
laion/larger_clap_general
```

CLAP is conceptually similar to CLIP but aligns **audio and text**.

It can encode information that a transcript may discard, such as:

- music;
- environmental sounds;
- crowd noise;
- laughter;
- applause;
- vehicle sounds;
- animal sounds;
- certain acoustic characteristics.

However, for the primary task of determining the **semantic topic** of a spoken Reel, transcript → text embedding is expected to be a very strong starting representation.

Direct audio embeddings should therefore initially be considered an additional modality rather than an automatic replacement for transcription.

---

# Potential Multimodal Representation

An eventual system may combine several representations:

```text
                         REEL
                          |
              +-----------+-----------+
              |                       |
         visual stream              audio
              |                       |
         video encoder        +------+------+
              |               |             |
       video embedding       ASR       audio encoder
                              |             |
                         transcript     audio embedding
                              |
                         text encoder
                              |
                      transcript embedding

              +---------------+---------------+
                              |
                      ranking system
                              |
                     candidate folders
```

Exactly how these modalities should be combined is **not predetermined**.

Possible approaches could include:

- weighted similarity scores;
- concatenated embeddings;
- learned fusion;
- ranking models;
- neural models;
- contrastive training.

This should be explored empirically.

---

# Folder Representation

Initially, the simplest representation of a folder is its title:

```text
"Gym"
 ↓
text encoder
 ↓
folder embedding
```

However, this creates an obvious limitation.

Consider folders named:

```text
Good Stuff
Things
Later
Important
Cool
```

The folder title itself contains little useful semantic information.

A more advanced version of the system should therefore consider the **Reels already stored inside the folder**.

For example:

```text
                   "Good Stuff"
                        |
       +----------------+----------------+
       |                |                |
  ML tutorial      Python career      AI explanation
       |                |                |
    embedding        embedding          embedding
       +----------------+----------------+
                        |
                  aggregation
                        |
                folder representation
```

A folder representation could eventually combine:

```text
folder title
+
embeddings of previously saved Reels
```

This makes the recommendation system personalized.

Instead of learning what `"Good Stuff"` means in general, the system learns what `"Good Stuff"` means **for this particular user**.

The exact aggregation method should remain an experimental question.

---

# Behavioral / Personalized Information

The eventual recommendation score may contain more than semantic similarity.

Useful signals could include:

- similarity between Reel and folder;
- similarity between Reel and existing contents of the folder;
- folder usage frequency;
- recency of folder use;
- number of Reels stored in the folder;
- user's historical saving behavior.

This means a later ranking model might conceptually estimate:

```text
score =
    semantic relevance
    + folder-content relevance
    + user preference
    + behavioral context
```

The current Instagram-style recency heuristic therefore does not necessarily have to be discarded. Recency could become **one feature among several**.

---

# Supervised Learning

The earliest semantic-ranking baseline does not necessarily require supervised training.

For example:

```text
reel embedding
       |
       | cosine similarity
       |
folder embedding
```

can rank folders zero-shot using pretrained embeddings.

However, actual user saving behavior naturally produces supervised data.

Suppose the candidate folders are:

```text
Gym
Recipes
Travel
Funny
```

and the user chooses:

```text
Gym
```

This observation provides evidence that:

```text
(reel, Gym)
```

should rank above:

```text
(reel, Recipes)
(reel, Travel)
(reel, Funny)
```

The important conceptual point is that the unselected folders are not necessarily completely incorrect. They are primarily **less preferred than the selected folder for this Reel**.

Therefore, later stages of the project may be more naturally formulated using:

- learning-to-rank;
- pairwise ranking;
- contrastive learning;
- retrieval objectives;

rather than ordinary multiclass classification.

---

# Fine-Tuning

Fine-tuning is **not required for the initial project**.

Pretrained models can initially be used as frozen feature extractors:

```text
pretrained encoder
        ↓
     frozen
        ↓
    embeddings
        ↓
ranking/similarity model
```

Once sufficient labeled Reel-folder interactions exist, fine-tuning could become useful.

Conceptually, training could encourage:

```text
Reel
  ↕ close
folder actually selected
```

while pushing less appropriate candidate folders farther away.

This could potentially use:

- contrastive loss;
- triplet loss;
- ranking loss;
- other retrieval objectives.

Parameter-efficient methods such as **LoRA / PEFT** may be worth considering if a sufficiently large transformer is fine-tuned.

Hugging Face's role would likely include some combination of:

- **Hub** — pretrained model checkpoints;
- **Transformers** — loading/running models;
- **Datasets** — dataset management;
- **Sentence Transformers** — embeddings and contrastive/retrieval training;
- **PEFT** — LoRA/parameter-efficient fine-tuning;
- **Accelerate** — GPU/distributed training if needed.

Do not add fine-tuning merely to make the project appear more sophisticated. First establish whether pretrained embeddings already solve much of the problem.

---

# Suggested Experimental Framing

The project can naturally compare different representations of the same Reel.

For example:

```text
Experiment A
Transcript → text embedding → folder ranking

Experiment B
Visual video → video embedding → folder ranking

Experiment C
Transcript + video → folder ranking

Experiment D
Transcript + video + folder history

Experiment E
Learn ranking from historical folder selections

Experiment F
Fine-tune embedding/ranking model if justified
```

This is not intended as a mandatory sequence. Experiments should change depending on early results.

A particularly useful research question is:

> Which modalities and personalization signals provide meaningful improvements in Reel-to-folder ranking?

---

# Evaluation

Because the output is an **ordered list**, ordinary classification accuracy is insufficient.

Suppose the correct folder appears:

```text
Model A → position 1
Model B → position 2
Model C → position 8
```

These outcomes should not receive the same score.

Relevant ranking metrics include:

### Hit Rate @ K

Whether the folder actually selected by the user appears within the top `K` recommendations.

Examples:

```text
Hit@1
Hit@3
Hit@5
```

### Mean Reciprocal Rank (MRR)

Rewards systems that place the relevant folder closer to the top.

If the correct folder is:

```text
rank 1 → reciprocal rank = 1
rank 2 → reciprocal rank = 1/2
rank 5 → reciprocal rank = 1/5
```

### Recall@K

Useful if more than one folder can reasonably be considered relevant.

### NDCG

Potentially useful if relevance is graded rather than strictly correct/incorrect.

Early evaluation can use manually assigned target folders. Later evaluation should ideally use actual observed folder choices.

---

# Data Considerations

Initially, the project can use a relatively small collection of locally stored short-form videos.

Possible metadata could eventually resemble:

```text
video_id
video_path
user_id
candidate_folders
selected_folder
caption
transcript
```

Folder information might eventually contain:

```text
folder_id
user_id
folder_name
created_at
previously_saved_video_ids
```

Do not commit large raw video files or copyrighted/private material directly to Git.

Keep paths/configuration separate from the tracked code when appropriate.

The initial dataset does not need to resemble Instagram's full scale. The first objective is to demonstrate that the recommendation formulation works.

---

# Potential Baselines

Do not begin only with sophisticated neural systems.

Useful baselines may include:

### Recency baseline

Simply rank folders according to recent use.

This approximates the behavior the project is attempting to improve upon.

### Frequency baseline

Rank according to how often the user uses each folder.

### Keyword / TF-IDF baseline

For transcript-based experiments.

### Pretrained semantic embedding baseline

Transcript embedding vs folder-title embedding using cosine similarity.

### Folder-content embedding baseline

Compare the Reel against representations generated from Reels already inside each folder.

These baselines are valuable because a complicated multimodal model is only worthwhile if it beats simpler methods.

---

# Possible Demo

Instagram integration is not required.

Once the model is functional, a small demo could be built using:

- Gradio;
- Streamlit;
- or another lightweight interface.

Conceptually:

```text
┌──────────────────────────────────┐
│       Reel Folder Recommender    │
│                                  │
│        [ Upload Video ]          │
│                                  │
│        [ video preview ]         │
│                                  │
│ Save to:                         │
│                                  │
│  1. Gym                    91%   │
│  2. Healthy Food            63%  │
│  3. Recipes                 42%  │
│  4. Travel                   8%  │
│  5. Memes                    3%  │
└──────────────────────────────────┘
```

This would demonstrate the proposed user experience without needing access to Instagram.

A backend/API such as FastAPI can be added later if deploying the model becomes relevant. It is not necessary during early experimentation.

---

# Initial Model Shortlist

These are candidates to investigate, not mandatory choices.

## Speech-to-text

```text
openai/whisper-large-v3-turbo
openai/whisper-large-v3
```

## Text embeddings

```text
sentence-transformers/all-MiniLM-L6-v2
sentence-transformers/all-mpnet-base-v2
Qwen/Qwen3-Embedding-0.6B
BAAI/bge-base-en-v1.5
```

## Video / video-text representations

```text
InternVideo2 family / OpenGVLab checkpoints
LanguageBind/LanguageBind_Video
X-CLIP
```

## Audio-text representations

```text
LAION CLAP
laion/larger_clap_general
```

## Potential image/frame baseline

```text
CLIP
```

CLIP could be applied to sampled frames and aggregated to create a simple visual baseline before or alongside dedicated temporal video encoders.

---

# Key Principles for This Project

1. **Treat this primarily as ranking/recommendation, not fixed-label classification.**

2. **User-created folder names are dynamic**, so do not assume a universal predefined set of topic labels.

3. **Start with pretrained models and simple baselines before fine-tuning.**

4. **Separate modalities initially** so their contribution can be measured.

5. A video embedding does **not automatically imply audio understanding**. Check the exact model/checkpoint.

6. For spoken semantic content, **ASR → transcript → text embedding** is an important baseline.

7. Visual video embeddings are especially important when the Reel's meaning is not communicated through speech.

8. Raw audio embeddings may add information about music and non-speech sounds that transcripts discard.

9. Folder contents may eventually provide a much better representation than folder titles alone.

10. Historical saving behavior can turn the system into a genuinely personalized recommender.

11. Because recommendations are ordered, prioritize **ranking metrics** such as MRR and Hit@K rather than relying only on classification accuracy.

12. Fine-tuning should be introduced only when there is enough evidence/data to justify it.

13. Use notebooks for early experimentation and refactor stable functionality into reusable Python modules later.

14. Keep the architecture flexible. Early empirical results should determine which modalities, models, ranking methods, and personalization strategies are worth keeping.

---

# Current High-Level Objective

The first major milestone is simply:

> Given a real short-form video and a user's set of arbitrarily named folders, produce a sensible ranking of those folders according to how appropriate they are for saving that video.

Everything else—multimodal fusion, personalization, supervised ranking, fine-tuning, application UI, APIs, and deployment—can evolve from that core problem.
