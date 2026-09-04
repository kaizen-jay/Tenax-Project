This is all coded using Claude and ChatGPT...
A project built in a hurry..
But it taught me some lessons that i wont forget. Lessons to manage time.
Lesson on how to communicate with your team....



# AI Teacher — Hackathon Submission

A locally-run AI Teacher that ingests uploaded material or a topic, plans a
personalized lesson, teaches it beat-by-beat through generated video
(avatar + voice + subject-aware visuals + captions), questions the student
during the lesson, detects specific misconceptions and re-explains
differently when needed, and produces a final learning report.

## Problem statement
See `Round_2_Technical_Assessment.pdf`. In short: build an AI Teacher that
behaves like a real educator (plan → explain → question → evaluate → adapt),
not a Q&A chatbot, and delivers the lesson as video.

## Solution overview
Everything runs locally via Ollama + a TTS fallback chain — no paid APIs,
chosen to match the team's hardware constraints (MacBook Air M-series, CPU
only). Critically, the video pipeline has **zero external binary
dependencies** (no ImageMagick, no system ffmpeg install) — every frame is
composited directly with Pillow and encoded via `imageio-ffmpeg`, which
ships its own ffmpeg binary inside the pip package. This was a deliberate
rebuild after the original MoviePy+ImageMagick pipeline proved too
environment-fragile (font resolution, binary discovery, and audio codec
issues varied across machines) for reliable hackathon-day demoing.
The system is split into two halves matching the assessment's two tasks:

- **Task 1 (AI Teaching Video):** `core/lesson_planner.py` + `core/visuals.py`
  + `core/tts.py` + `core/avatar.py` + `core/video.py` turn a topic/material
  into a structured lesson of short video "beats," each with a
  subject-appropriate visual, narrated by a locally-rendered avatar.
- **Task 2 (Interactive & Adaptive):** `core/orchestrator.py` is a state
  machine (`EXPLAIN → AWAITING_ANSWER → EVALUATING → REMEDIATING/ADVANCING`)
  that pauses after each beat's check-in question, evaluates the student's
  actual answer (not just right/wrong — identifies the specific
  misconception), and either re-explains with a new analogy/example or
  advances, per section 12 of the brief.

## Key features
- RAG-grounded teaching from uploaded PDFs/DOCX/PPTX/TXT (`core/rag.py`,
  `core/ingestion.py`) — every retrieved chunk is tagged with its source
  and location; the planner is instructed to flag (not invent) content the
  material doesn't cover.
- Time-budgeted lesson structuring (5 / 20 / 60 minute modes) that changes
  beat count and depth, not just length.
- Subject-aware visual selection (math→graph, physics→diagram, history→
  timeline, programming→code, etc.) — see `core/visuals.py` `RENDERERS` map.
- Multilingual narration via a three-tier TTS fallback (`core/tts.py`):
  Piper (if a voice model is present) → macOS `say` (zero setup on Mac) →
  silent WAV of an estimated duration (guarantees the lesson always
  proceeds with captions even with no TTS backend available at all).
- Misconception-specific remediation loop, not generic "try again."
- Persistent learner profile across sessions (`core/learner_profile.py`).
- Final learning report derived directly from session logs (no separate
  scoring re-implementation to drift out of sync).

## System architecture
```
Upload/Topic
    ↓
core/ingestion.py + core/rag.py   (parse, chunk, embed, retrieve)
    ↓
core/lesson_planner.py            (LLM → structured beat list, JSON)
    ↓
core/orchestrator.py  ←───────────┐  (state machine drives the loop)
    ↓                             │
core/visuals.py + core/tts.py     │  student answer
+ core/avatar.py + core/video.py  │
    ↓                             │
 rendered beat video  →  Streamlit UI (app.py)  → student answers ─┘
    ↓ (on completion)
core/assessment.py → core/learner_profile.py
```

## AI/ML models used
- LLM: Ollama, `qwen2.5:7b` by default (swap in `core/llm.py:LLMConfig`;
  `deepseek-r1:14b` also tested well for explanation quality per prior
  local testing — trade-off is latency).
- Embeddings: `all-MiniLM-L6-v2` (sentence-transformers), local, CPU-fast.
- TTS: Piper (`rhasspy/piper`) if configured, else macOS `say`, else a
  silent fallback — see `core/tts.py`.
- Avatar: hand-rendered 2D face, amplitude-driven mouth animation — no ML
  model, chosen deliberately for CPU-only reliability (see
  `core/avatar.py` docstring for the reasoning).

## RAG implementation
FAISS flat index (cosine via normalized inner product) over chunks with
100–150 char overlap to avoid losing concepts split across page/slide
boundaries. Retrieval results carry `(source, location, score)` and are
formatted with explicit `[Source: ...]` tags in the prompt so the LLM can
be instructed to only draw from them.

## Personalization approach
Learner level, language, available time, and objective are collected up
front and drive the lesson planner prompt directly (not post-hoc filtering)
— level changes vocabulary/depth per-beat, time changes beat count/depth,
language changes both narration and captions.

## Assessment methodology
Score = (# check-ins answered "correct") / (total check-ins), computed
directly from `TeachingSession.logs` — the same log the remediation loop
uses, so the report can't disagree with what actually happened in the
lesson.

## Multilingual implementation
Lesson planning and narration text are generated in the target language by
prompting the LLM directly in that language; TTS voice is selected via
`core/tts.py:VOICE_MAP`. Extending to more languages = add a Piper voice
file + a map entry.

## Voice implementation
Three-tier fallback in `core/tts.py`: Piper (best quality, needs a
downloaded voice model) → macOS `say` (built into every Mac, zero setup,
used automatically if no Piper model is found) → silent WAV sized to an
estimated speaking duration (last resort, so a beat never fails to render
for lack of audio — captions still convey the content). Amplitude is
extracted from whichever WAV was produced to drive avatar mouth movement.

## Avatar/video generation approach
2D procedurally-drawn face (Pillow), composited frame-by-frame directly
onto a subject-appropriate matplotlib/Pygments-rendered slide — no video
library (MoviePy) in the loop at all. Frames are encoded straight to h264
via `imageio-ffmpeg`, which bundles its own ffmpeg binary, then the
narration audio is muxed in with the same bundled binary. Captions are
drawn with Pillow's own text renderer using a font bundled inside
matplotlib (`DejaVuSans-Bold.ttf`), guaranteeing a valid font path on any
machine that has matplotlib installed — no dependency on OS font
registration. See `core/avatar.py` for why a procedural 2D face was chosen
over a photorealistic talking-head model given local-only CPU constraints.

## APIs and third-party services
None. Fully local: Ollama (LLM), Piper/macOS `say` (TTS),
sentence-transformers + FAISS (retrieval), Pillow/Matplotlib/Pygments
(rendering), `imageio-ffmpeg` (bundled video encoder — no system
ffmpeg/ImageMagick install required).

## Setup instructions
```bash
# 1. Install Ollama and pull a model
ollama pull qwen2.5:7b

# 2. Python deps (imageio-ffmpeg bundles its own ffmpeg binary — nothing
#    else to install system-side for video)
cd ai_teacher
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. (Optional, better voice quality) Piper voice models — if skipped, the
#    app automatically falls back to macOS `say`, then to silent+captions.
#    Download from https://huggingface.co/rhasspy/piper-voices into models/piper/:
#      en_US-lessac-medium.onnx (+ .onnx.json)
#      hi_IN-pratham-medium.onnx (+ .onnx.json)

# 4. Run
ollama serve &         # if not already running
streamlit run app.py
```

## Deployment instructions
Local demo only (matches "local only" constraint) — run `streamlit run
app.py` on the presenting machine. No cloud deployment needed for judging;
if a hosted demo is required, deploy behind the same stack replacing
`localhost:11434` with a hosted Ollama endpoint.

## Known limitations
- Avatar is a simple 2D illustration with amplitude-driven lip movement,
  not a photorealistic talking head (deliberate local-CPU trade-off — see
  `core/avatar.py`).
- Without a Piper voice model installed, narration falls back to macOS
  `say` (Mac only) or silent audio with captions (any OS) — voice quality/
  availability depends on what's set up on the presenting machine.
- Piper TTS on romanized Hinglish text is imperfect (no native Hinglish
  voice model exists); currently falls back to the Hindi voice.
- Graph rendering (`core/visuals.py:render_graph_slide`) supports a small
  safe set of function shapes rather than arbitrary equations, to avoid
  `eval()`-ing model output.
- Remediation loop caps at 2 attempts per beat before advancing anyway, to
  avoid an unrecoverable stall in a live demo.
- No user auth / multi-tenant support — one JSON profile file per student,
  local only.

## Demo script (suggested, 3–7 min)
1. Upload a short PDF (e.g. a physics chapter) or type a topic.
2. Set level=beginner, language=English, time=20 minutes → show the
   generated lesson plan structure.
3. Play through 2 beats of generated video (avatar + visuals + voice).
4. Deliberately answer a check-in question wrong → show misconception
   detection + re-explanation with a different analogy → correct on retry.
5. Finish the lesson → show the learning report.
6. Restart in Hindi to show multilingual + re-grounding.
