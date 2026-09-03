"""
Section 14: persistent learner profile across sessions, stored as one JSON
file per student. Deliberately simple (no DB server) so it stays "local
only" per the team's constraint and needs zero setup.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict

from .assessment import LearningReport


@dataclass
class LearnerProfile:
    student_id: str
    topics_studied: List[str] = field(default_factory=list)
    learning_history: List[Dict] = field(default_factory=list)  # one entry per completed lesson
    strong_concepts: List[str] = field(default_factory=list)
    weak_concepts: List[str] = field(default_factory=list)
    current_learning_path: List[str] = field(default_factory=list)
    current_path_index: int = 0


class ProfileStore:
    def __init__(self, profiles_dir: str):
        self.profiles_dir = profiles_dir
        os.makedirs(profiles_dir, exist_ok=True)

    def _path(self, student_id: str) -> str:
        return os.path.join(self.profiles_dir, f"{student_id}.json")

    def load(self, student_id: str) -> LearnerProfile:
        path = self._path(student_id)
        if not os.path.exists(path):
            return LearnerProfile(student_id=student_id)
        with open(path, "r") as f:
            data = json.load(f)
        return LearnerProfile(**data)

    def save(self, profile: LearnerProfile):
        with open(self._path(profile.student_id), "w") as f:
            json.dump(asdict(profile), f, indent=2)

    def record_lesson(self, student_id: str, report: LearningReport):
        profile = self.load(student_id)
        if report.topic not in profile.topics_studied:
            profile.topics_studied.append(report.topic)
        profile.learning_history.append({
            "topic": report.topic,
            "score": report.score_percent,
            "date": datetime.utcnow().isoformat(),
            "weak_concepts": report.weak_concepts + report.incorrect_concepts,
        })
        # dedupe while preserving recency
        profile.strong_concepts = list(dict.fromkeys(report.strong_concepts + profile.strong_concepts))
        profile.weak_concepts = list(dict.fromkeys(report.weak_concepts + report.incorrect_concepts + profile.weak_concepts))
        self.save(profile)
        return profile
