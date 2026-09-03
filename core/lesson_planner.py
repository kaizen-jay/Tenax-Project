"""
Turns (topic or retrieved material) + learner preferences into a structured
lesson plan: an ordered list of "beats". Each beat is one chunk of teaching
that will become one video segment, followed by an optional check-in
question. This structure is what makes section 7 (time-based learning) and
section 6 (personalization) mechanical rather than vibes-based.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .llm import LocalLLM
from .rag import VectorStore, format_context_for_prompt

VISUAL_TYPES = [
    "equation", "graph", "step_by_step", "diagram", "formula", "process",
    "simulation_note", "labeled_diagram", "timeline", "map", "code",
    "execution_flow", "architecture_diagram", "plain_text", "analogy_image",
]


@dataclass
class LessonBeat:
    beat_id: int
    title: str
    explanation_points: List[str]         # bullet points to cover, in order
    visual_type: str                       # one of VISUAL_TYPES
    visual_spec: str                       # what specifically to draw/show
    example: Optional[str]
    check_in_question: Optional[str]       # None if this beat doesn't end in a question
    question_type: Optional[str]           # conceptual | mcq | short_answer | application | explain_back
    est_seconds: int
    grounded: bool = True                  # False if no material was retrieved for this beat


@dataclass
class LessonPlan:
    topic: str
    level: str                # beginner | intermediate | advanced
    language: str
    total_minutes: int
    subject_area: str         # math | physics | biology | history | programming | general
    beats: List[LessonBeat] = field(default_factory=list)
    learning_objective: str = ""


TIME_STRUCTURE_GUIDE = {
    # minutes: (beat_count, depth_note)
    5: (2, "Only the single most important idea. No detours."),
    20: (4, "Key concepts with one example each. One check-in question mid-lesson, one at the end."),
    60: (7, "Deeper coverage, multiple examples, a check-in after most beats, ends in a full assessment."),
}


def _pick_structure_guide(minutes: int) -> tuple[int, str]:
    if minutes <= 7:
        return TIME_STRUCTURE_GUIDE[5]
    if minutes <= 35:
        return TIME_STRUCTURE_GUIDE[20]
    return TIME_STRUCTURE_GUIDE[60]


PLANNER_SYSTEM = """You are an expert curriculum designer and teacher. You design lesson plans as \
a sequence of teaching "beats" — small, focused chunks that build on each other, matching how a \
good human teacher paces a real lesson. You never dump everything into one wall of text."""


def _subject_area_from_topic(llm: LocalLLM, topic: str) -> str:
    prompt = (
        f"Classify the subject area of this topic into exactly one word from this list: "
        f"math, physics, biology, history, programming, general.\n\nTopic: {topic}\n\n"
        f"Respond with only the single word."
    )
    resp = llm.generate(prompt).strip().lower()
    for area in ["math", "physics", "biology", "history", "programming"]:
        if area in resp:
            return area
    return "general"


def build_lesson_plan(
    llm: LocalLLM,
    topic: str,
    level: str,
    language: str,
    total_minutes: int,
    vector_store: Optional[VectorStore] = None,
    learning_objective: str = "",
) -> LessonPlan:
    beat_count, depth_note = _pick_structure_guide(total_minutes)
    subject_area = _subject_area_from_topic(llm, topic)

    # Ground in material if a vector store was provided (uploaded document flow).
    context = ""
    grounded = False
    if vector_store is not None:
        retrieved = vector_store.query(topic, k=8)
        context = format_context_for_prompt(retrieved)
        grounded = bool(retrieved)

    grounding_instruction = ""
    if vector_store is not None:
        grounding_instruction = (
            "\n\nYou MUST base the lesson ONLY on the material below. Do not introduce facts, "
            "numbers, or claims that are not present in it. If the material doesn't fully cover "
            "the topic, say so explicitly in the relevant beat rather than inventing content.\n\n"
            f"MATERIAL:\n{context}"
        )

    schema_hint = """
{
  "learning_objective": "one sentence",
  "beats": [
    {
      "title": "short beat title",
      "explanation_points": ["point 1", "point 2"],
      "visual_type": "one of: equation, graph, step_by_step, diagram, formula, process, simulation_note, labeled_diagram, timeline, map, code, execution_flow, architecture_diagram, plain_text, analogy_image",
      "visual_spec": "concrete description of exactly what the visual should show",
      "example": "a concrete example or null",
      "check_in_question": "a question to ask the student, or null if this beat has none",
      "question_type": "one of: conceptual, mcq, short_answer, application, explain_back, null",
      "est_seconds": 45
    }
  ]
}
"""

    prompt = f"""Design a lesson plan for teaching this to a {level} student in {language}.

Topic: {topic}
Learning objective (if given by student): {learning_objective or "(not specified — infer a sensible one)"}
Subject area: {subject_area}
Total available time: {total_minutes} minutes
Target beat count: {beat_count}
Depth guidance: {depth_note}

Rules:
- Order beats so each builds on the previous one.
- Roughly every other beat (and always the last one) should end with a check_in_question.
- est_seconds across all beats should sum to roughly {total_minutes * 60} seconds.
- visual_type must be chosen based on what actually fits the content (see section 10 style rubric: \
math->equation/graph/step_by_step, physics->diagram/formula/process/simulation_note, \
biology->labeled_diagram/process, history->timeline/map, programming->code/execution_flow/architecture_diagram).
{grounding_instruction}

Return JSON matching exactly this shape:
{schema_hint}
"""

    data = llm.generate_json(prompt, system=PLANNER_SYSTEM)

    beats = []
    for i, b in enumerate(data.get("beats", [])):
        beats.append(LessonBeat(
            beat_id=i,
            title=b.get("title", f"Beat {i+1}"),
            explanation_points=b.get("explanation_points", []),
            visual_type=b.get("visual_type", "plain_text"),
            visual_spec=b.get("visual_spec", ""),
            example=b.get("example"),
            check_in_question=b.get("check_in_question"),
            question_type=b.get("question_type"),
            est_seconds=int(b.get("est_seconds", 45)),
            grounded=grounded if vector_store is not None else True,
        ))

    return LessonPlan(
        topic=topic,
        level=level,
        language=language,
        total_minutes=total_minutes,
        subject_area=subject_area,
        beats=beats,
        learning_objective=data.get("learning_objective", learning_objective),
    )
