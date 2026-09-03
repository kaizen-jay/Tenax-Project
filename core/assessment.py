"""
Section 13: final assessment + learning report. Consumes the SessionLog
list from a completed TeachingSession — no separate re-implementation of
scoring logic, it's derived straight from what actually happened in the
lesson so the report can't drift from reality.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .llm import LocalLLM
from .orchestrator import TeachingSession


@dataclass
class LearningReport:
    topic: str
    score_percent: float
    strong_concepts: List[str]
    weak_concepts: List[str]
    incorrect_concepts: List[str]
    recommendation: str
    suggested_next_topic: str


def generate_report(llm: LocalLLM, session: TeachingSession) -> LearningReport:
    total = len(session.logs)
    correct = sum(1 for log in session.logs if log.evaluation and log.evaluation.correctness == "correct")
    score = round(100 * correct / total, 1) if total else 0.0

    strong = [session.plan.beats[log.beat_id].title for log in session.logs
              if log.evaluation and log.evaluation.correctness == "correct"]
    weak = [session.plan.beats[log.beat_id].title for log in session.logs
            if log.evaluation and log.evaluation.correctness == "partial"]
    incorrect = [session.plan.beats[log.beat_id].title for log in session.logs
                 if log.evaluation and log.evaluation.correctness == "incorrect"]
    misconceptions = [log.evaluation.misconception for log in session.logs
                       if log.evaluation and log.evaluation.misconception]

    prompt = f"""A student just finished a lesson on "{session.plan.topic}" ({session.plan.level} level).
Score: {score}%
Concepts they got right: {strong}
Concepts they struggled with (partial): {weak}
Concepts they got wrong: {incorrect}
Specific misconceptions observed: {misconceptions}

Write:
1. A short (2-3 sentence) actionable recommendation for what to revise and how.
2. One suggested next topic to learn, given they've now covered "{session.plan.topic}".

Return JSON: {{"recommendation": "...", "suggested_next_topic": "..."}}"""
    data = llm.generate_json(prompt)

    return LearningReport(
        topic=session.plan.topic,
        score_percent=score,
        strong_concepts=strong,
        weak_concepts=weak,
        incorrect_concepts=incorrect,
        recommendation=data.get("recommendation", ""),
        suggested_next_topic=data.get("suggested_next_topic", ""),
    )
