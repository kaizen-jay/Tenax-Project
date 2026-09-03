"""
The heart of the "human-like teaching" requirement (worth 20/100 alone).

This is a state machine, not a chat loop. It walks through the lesson plan
beat by beat. After each beat with a check-in question, it stops and waits
for a student answer. It never just marks right/wrong — it classifies the
answer, and on wrong/partial answers it runs a remediation sub-loop
(explain differently -> new example -> re-ask) before moving on, per
section 12 (Misconception Detection).

State is serializable so a Streamlit app (which reruns the whole script on
every interaction) can persist it in st.session_state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List

from .llm import LocalLLM
from .lesson_planner import LessonPlan, LessonBeat


class TeachingState(Enum):
    EXPLAIN = "explain"                 # deliver the beat's content
    AWAITING_ANSWER = "awaiting_answer"  # waiting on student input to the check-in question
    EVALUATING = "evaluating"           # LLM is classifying the student's answer
    REMEDIATING = "remediating"         # student got it wrong/partial -> re-explain differently
    ADVANCING = "advancing"             # move to next beat
    DONE = "done"                       # lesson complete -> hand off to assessment.py


@dataclass
class AnswerEvaluation:
    correctness: str            # "correct" | "partial" | "incorrect"
    misconception: Optional[str]
    feedback: str                # what to say to the student, in character as the teacher
    should_remediate: bool


@dataclass
class SessionLog:
    beat_id: int
    question: Optional[str]
    student_answer: Optional[str]
    evaluation: Optional[AnswerEvaluation]
    remediation_count: int = 0


@dataclass
class TeachingSession:
    plan: LessonPlan
    current_beat_index: int = 0
    state: TeachingState = TeachingState.EXPLAIN
    logs: List[SessionLog] = field(default_factory=list)
    remediation_attempts_this_beat: int = 0
    max_remediation_attempts: int = 2  # avoid infinite loops if a student keeps missing it

    @property
    def current_beat(self) -> LessonBeat:
        return self.plan.beats[self.current_beat_index]

    @property
    def is_last_beat(self) -> bool:
        return self.current_beat_index >= len(self.plan.beats) - 1


EVAL_SYSTEM = """You are an expert, patient teacher evaluating a student's answer during a live \
lesson. You classify understanding accurately, identify the SPECIFIC misconception behind wrong \
answers (not just "wrong"), and give constructive, encouraging feedback the way a good human \
teacher would — never dismissive."""


class TeachingOrchestrator:
    """Advances a TeachingSession through the state machine. Each method is a
    pure step: call it, get an updated session + something to show the user."""

    def __init__(self, llm: LocalLLM):
        self.llm = llm

    def start(self, plan: LessonPlan) -> TeachingSession:
        return TeachingSession(plan=plan)

    def get_current_beat_for_display(self, session: TeachingSession) -> LessonBeat:
        """The EXPLAIN step: just surface the planned beat. Video/TTS layer
        consumes this to render the segment."""
        return session.current_beat

    def submit_answer(self, session: TeachingSession, student_answer: str) -> TeachingSession:
        """Called when the student answers a check-in question. Runs the
        Evaluate step, then decides Adapt (remediate) vs Continue."""
        beat = session.current_beat
        evaluation = self._evaluate_answer(beat, student_answer, session)

        log = SessionLog(
            beat_id=beat.beat_id,
            question=beat.check_in_question,
            student_answer=student_answer,
            evaluation=evaluation,
            remediation_count=session.remediation_attempts_this_beat,
        )
        session.logs.append(log)

        if evaluation.should_remediate and session.remediation_attempts_this_beat < session.max_remediation_attempts:
            session.remediation_attempts_this_beat += 1
            session.state = TeachingState.REMEDIATING
        else:
            session.remediation_attempts_this_beat = 0
            session.state = TeachingState.ADVANCING
        return session

    def _evaluate_answer(self, beat: LessonBeat, student_answer: str, session: TeachingSession) -> AnswerEvaluation:
        prior_attempts = session.remediation_attempts_this_beat
        prompt = f"""The lesson is teaching: "{beat.title}"
Key points covered: {beat.explanation_points}
Question asked: {beat.check_in_question}
Question type: {beat.question_type}
Student's answer: "{student_answer}"
This is attempt number {prior_attempts + 1} for this question.

Evaluate the answer. If it's wrong or partially wrong, identify the SPECIFIC underlying \
misconception (e.g. "confuses correlation with causation" or "thinks resistance increasing \
increases current, not decreases it") — do not just say "incorrect".

Return JSON:
{{
  "correctness": "correct" | "partial" | "incorrect",
  "misconception": "specific misconception, or null if correct",
  "feedback": "1-3 sentences of feedback to say directly to the student, encouraging and specific",
  "should_remediate": true or false
}}

should_remediate should be true unless correctness is "correct"."""
        data = self.llm.generate_json(prompt, system=EVAL_SYSTEM)
        return AnswerEvaluation(
            correctness=data.get("correctness", "incorrect"),
            misconception=data.get("misconception"),
            feedback=data.get("feedback", ""),
            should_remediate=bool(data.get("should_remediate", True)),
        )

    def generate_remediation(self, session: TeachingSession) -> LessonBeat:
        """Adapt step: build a NEW mini-beat that re-explains the concept
        differently — different analogy, different example — per section 12."""
        beat = session.current_beat
        last_log = session.logs[-1]
        misconception = last_log.evaluation.misconception if last_log.evaluation else "unclear"

        prompt = f"""A student is struggling with this concept: "{beat.title}"
Original explanation points: {beat.explanation_points}
Their specific misconception: {misconception}
Their answer was: "{last_log.student_answer}"

Create a short remediation explanation that:
1. Uses a DIFFERENT analogy than a generic textbook explanation would.
2. Directly addresses their specific misconception.
3. Gives one new concrete example (different from any implied by the original beat).
4. Ends with a new, simpler check-in question to re-check understanding.

Return JSON in this exact shape:
{{
  "title": "short title, e.g. 'Let's look at this differently'",
  "explanation_points": ["point 1", "point 2"],
  "visual_type": "{beat.visual_type}",
  "visual_spec": "what the visual should show for this new explanation",
  "example": "the new example",
  "check_in_question": "a new, simpler question",
  "question_type": "{beat.question_type}",
  "est_seconds": 40
}}"""
        data = self.llm.generate_json(prompt, system=EVAL_SYSTEM)
        return LessonBeat(
            beat_id=beat.beat_id,
            title=data.get("title", "Let's try a different approach"),
            explanation_points=data.get("explanation_points", []),
            visual_type=data.get("visual_type", beat.visual_type),
            visual_spec=data.get("visual_spec", ""),
            example=data.get("example"),
            check_in_question=data.get("check_in_question"),
            question_type=data.get("question_type", beat.question_type),
            est_seconds=int(data.get("est_seconds", 40)),
            grounded=beat.grounded,
        )

    def advance(self, session: TeachingSession) -> TeachingSession:
        """Continue step: move to the next beat or finish the lesson."""
        if session.is_last_beat:
            session.state = TeachingState.DONE
        else:
            session.current_beat_index += 1
            session.state = TeachingState.EXPLAIN
        return session

    def answer_followup_question(self, session: TeachingSession, question: str) -> str:
        """Section 12 / mandatory req 11: answer ad-hoc student questions
        mid-lesson while maintaining lesson context, without derailing the
        planned flow."""
        beat = session.current_beat
        covered_so_far = [b.title for b in session.plan.beats[:session.current_beat_index + 1]]
        prompt = f"""You are mid-lesson on "{session.plan.topic}" ({session.plan.level} level, \
in {session.plan.language}). Covered so far: {covered_so_far}. Currently on: "{beat.title}".

The student has a follow-up question: "{question}"

Answer it clearly and briefly (this is a spoken aside, not a new lesson section), staying \
consistent with what's been taught so far, then note whether it connects to what's coming up."""
        return self.llm.generate(prompt, system=EVAL_SYSTEM)
