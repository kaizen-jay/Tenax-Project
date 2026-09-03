"""
AI Teacher — main app. Run with: streamlit run app.py

Flow implemented here:
  Setup (upload or topic + preferences)
    -> Lesson plan generation (core/lesson_planner.py)
    -> Beat-by-beat teaching loop (core/orchestrator.py):
         render video for current beat (core/video.py)
         -> show it, wait for answer if there's a check-in question
         -> evaluate -> remediate or advance
    -> Final assessment + learning report (core/assessment.py)
    -> Learner profile updated (core/learner_profile.py)

Kept as a single file for hackathon judging clarity — see README for how
to split into pages if you want a nicer multi-step UI.
"""
import os
import tempfile

import streamlit as st

from core.llm import LocalLLM, LLMConfig
from core.rag import VectorStore
from core.lesson_planner import build_lesson_plan
from core.orchestrator import TeachingOrchestrator, TeachingState
from core.assessment import generate_report
from core.learner_profile import ProfileStore
from core import video as video_mod

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
VECTOR_DIR = os.path.join(BASE_DIR, "data", "vector_store")
PROFILE_DIR = os.path.join(BASE_DIR, "data", "profiles")
MODELS_DIR = os.path.join(BASE_DIR, "models", "piper")
WORK_DIR = os.path.join(BASE_DIR, "outputs")

st.set_page_config(page_title="AI Teacher", layout="wide")


@st.cache_resource
def get_llm():
    return LocalLLM(LLMConfig())


@st.cache_resource
def get_orchestrator():
    return TeachingOrchestrator(get_llm())


@st.cache_resource
def get_profile_store():
    return ProfileStore(PROFILE_DIR)


def init_state():
    defaults = {
        "stage": "setup",       # setup -> teaching -> report
        "session": None,
        "vector_store": None,
        "student_id": "student_1",
        "video_enabled": True,  # allow disabling video for fast text-only iteration/testing
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def setup_screen():
    st.title("🎓 AI Teacher")
    st.caption("Upload material or give a topic. The AI Teacher plans, explains, questions, and adapts — it doesn't just answer questions.")

    st.session_state.student_id = st.text_input("Your name / student ID", st.session_state.student_id)

    mode = st.radio("Teach from:", ["A topic", "Uploaded material"], horizontal=True)

    material_paths = []
    topic = ""
    if mode == "Uploaded material":
        uploaded = st.file_uploader(
            "Upload book / textbook / notes / slides / paper",
            type=["pdf", "docx", "pptx", "txt"],
            accept_multiple_files=True,
        )
        if uploaded:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            for f in uploaded:
                path = os.path.join(UPLOAD_DIR, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                material_paths.append(path)
        topic = st.text_input("What should the lesson focus on within this material?",
                               placeholder="e.g. Chapter 4, or 'the section on Ohm's Law'")
    else:
        topic = st.text_input("Topic", placeholder="e.g. Newton's Laws of Motion")

    col1, col2, col3 = st.columns(3)
    with col1:
        level = st.selectbox("Learner level", ["beginner", "intermediate", "advanced"])
    with col2:
        language = st.selectbox("Language", ["english", "hindi", "hinglish"])
    with col3:
        minutes = st.selectbox("Available time (minutes)", [5, 20, 60], index=1)

    objective = st.text_input("Learning objective (optional)", placeholder="e.g. prepare for a technical interview")
    st.session_state.video_enabled = st.checkbox(
        "Generate video+voice per beat (uncheck for fast text-only testing of the teaching logic)",
        value=st.session_state.video_enabled,
    )

    if st.button("Start lesson", type="primary", disabled=not topic):
        with st.spinner("Understanding the request and planning the lesson..."):
            llm = get_llm()
            vector_store = None
            if material_paths:
                vector_store = VectorStore(VECTOR_DIR)
                vector_store.build_from_files(material_paths)
                st.session_state.vector_store = vector_store

            plan = build_lesson_plan(
                llm, topic=topic, level=level, language=language,
                total_minutes=minutes, vector_store=vector_store,
                learning_objective=objective,
            )
            orchestrator = get_orchestrator()
            st.session_state.session = orchestrator.start(plan)
            st.session_state.stage = "teaching"
        st.rerun()


def render_beat_media(beat, language):
    beat_dir = os.path.join(WORK_DIR, f"session_beat_{beat.beat_id}_{hash(beat.title) % 10_000}")
    with st.spinner("Rendering teaching video for this beat (visuals + voice + avatar)..."):
        try:
            path = video_mod.render_beat_video(beat, language, WORK_DIR, MODELS_DIR)
            st.video(path)
        except Exception as e:
            st.warning(
                f"Video rendering unavailable right now ({e}). Showing the lesson content as text instead — "
                "the teaching logic below still runs normally."
            )
            render_beat_text(beat)


def render_beat_text(beat):
    st.subheader(beat.title)
    for point in beat.explanation_points:
        st.markdown(f"- {point}")
    if beat.example:
        st.info(f"**Example:** {beat.example}")
    if not beat.grounded:
        st.caption("⚠️ No matching content found in the uploaded material for this beat — flagged rather than invented.")


def teaching_screen():
    orchestrator = get_orchestrator()
    session = st.session_state.session
    plan = session.plan

    st.progress((session.current_beat_index + (0 if session.state == TeachingState.EXPLAIN else 1)) / len(plan.beats))
    st.caption(f"Topic: {plan.topic}  |  Beat {session.current_beat_index + 1} of {len(plan.beats)}  |  {plan.level} · {plan.language}")

    beat = session.current_beat

    if session.state == TeachingState.EXPLAIN:
        if st.session_state.video_enabled:
            render_beat_media(beat, plan.language)
        else:
            render_beat_text(beat)

        if beat.check_in_question:
            st.markdown(f"**Check-in:** {beat.check_in_question}")
            answer = st.text_input("Your answer", key=f"answer_{beat.beat_id}_{session.remediation_attempts_this_beat}")
            if st.button("Submit answer") and answer:
                with st.spinner("Evaluating your answer..."):
                    st.session_state.session = orchestrator.submit_answer(session, answer)
                st.rerun()
        else:
            if st.button("Continue"):
                st.session_state.session = orchestrator.advance(session)
                st.rerun()

    elif session.state == TeachingState.REMEDIATING:
        last_eval = session.logs[-1].evaluation
        st.warning(f"**Feedback:** {last_eval.feedback}")
        st.caption(f"Misconception identified: {last_eval.misconception}")
        with st.spinner("Preparing a different explanation..."):
            new_beat = orchestrator.generate_remediation(session)
        # temporarily swap in the remediation beat for display, keep same beat_id
        session.plan.beats[session.current_beat_index] = new_beat
        session.state = TeachingState.EXPLAIN
        st.session_state.session = session
        st.rerun()

    elif session.state == TeachingState.ADVANCING:
        last_eval = session.logs[-1].evaluation
        st.success(f"**Feedback:** {last_eval.feedback}")
        if st.button("Continue to next part"):
            st.session_state.session = orchestrator.advance(session)
            st.rerun()

    elif session.state == TeachingState.DONE:
        st.session_state.stage = "report"
        st.rerun()

    with st.expander("Ask a question about this lesson"):
        followup = st.text_input("Your question", key="followup")
        if st.button("Ask") and followup:
            with st.spinner("Thinking..."):
                answer = orchestrator.answer_followup_question(session, followup)
            st.info(answer)


def report_screen():
    llm = get_llm()
    session = st.session_state.session
    profile_store = get_profile_store()

    with st.spinner("Generating your learning report..."):
        report = generate_report(llm, session)
        profile_store.record_lesson(st.session_state.student_id, report)

    st.title("📊 Learning Report")
    st.metric("Score", f"{report.score_percent}%")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Strong areas**")
        st.write(report.strong_concepts or "—")
    with col2:
        st.markdown("**Needs improvement**")
        st.write(list(dict.fromkeys(report.weak_concepts + report.incorrect_concepts)) or "—")

    st.markdown(f"**Recommendation:** {report.recommendation}")
    st.markdown(f"**Suggested next topic:** {report.suggested_next_topic}")

    if st.button("Start a new lesson"):
        st.session_state.stage = "setup"
        st.session_state.session = None
        st.rerun()


def main():
    init_state()
    if st.session_state.stage == "setup":
        setup_screen()
    elif st.session_state.stage == "teaching":
        teaching_screen()
    elif st.session_state.stage == "report":
        report_screen()


if __name__ == "__main__":
    main()
