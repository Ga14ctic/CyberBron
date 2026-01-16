"""
Quiz Tab UI Component
Quiz interface with scoring and review.
"""
import streamlit as st
from services.quiz_service import QuizService
from generators.quiz_generator import QuizGenerator


def render_quiz_tab(quiz_service: QuizService, quiz_generator: QuizGenerator, notes_service=None):
    """
    Render the quiz tab.
    
    Args:
        quiz_service: QuizService instance
        quiz_generator: QuizGenerator instance
        notes_service: Optional NotesService for selecting notes
    """
    st.header("📊 Quiz Mode")
    
    # Tab navigation
    tab1, tab2, tab3 = st.tabs(["🎯 Take Quiz", "➕ Generate Quiz", "📈 Results"])
    
    with tab1:
        render_take_quiz(quiz_service)
    
    with tab2:
        render_generate_quiz(quiz_service, quiz_generator, notes_service)
    
    with tab3:
        render_quiz_results(quiz_service)


def render_take_quiz(quiz_service: QuizService):
    """Render quiz-taking interface."""
    st.subheader("Take a Quiz")
    
    quizzes = quiz_service.get_all_quizzes()
    
    if not quizzes:
        st.info("No quizzes available. Generate one in the 'Generate Quiz' tab!")
        return
    
    # Select quiz
    quiz_options = {f"{q['title']} ({q.get('difficulty', 'medium')})": q['id'] for q in quizzes}
    selected_quiz_name = st.selectbox("Select Quiz", list(quiz_options.keys()))
    
    if not selected_quiz_name:
        return
    
    quiz_id = quiz_options[selected_quiz_name]
    quiz = quiz_service.get_quiz(quiz_id)
    
    if not quiz:
        st.error("Quiz not found!")
        return
    
    st.markdown(f"**Topic:** {quiz.get('topic', 'General')}")
    st.markdown(f"**Difficulty:** {quiz.get('difficulty', 'medium').title()}")
    st.markdown(f"**Questions:** {len(quiz['questions'])}")
    
    if st.button("▶️ Start Quiz"):
        st.session_state.active_quiz = quiz_id
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False
        st.rerun()
    
    # Active quiz taking
    if st.session_state.get("active_quiz") == quiz_id and not st.session_state.get("quiz_submitted", False):
        st.divider()
        st.markdown("### Answer the following questions:")
        
        with st.form("quiz_form"):
            answers = {}
            
            for i, question in enumerate(quiz['questions']):
                st.markdown(f"**Question {i+1}:**")
                st.markdown(question['question'])
                
                q_type = question.get('type', 'multiple_choice')
                
                if q_type == 'multiple_choice':
                    options = question.get('options', [])
                    answer = st.radio(
                        f"Select answer for Q{i+1}",
                        options,
                        key=f"q_{i}",
                        label_visibility="collapsed"
                    )
                    answers[i] = answer
                
                elif q_type == 'true_false':
                    answer = st.radio(
                        f"Select answer for Q{i+1}",
                        ["True", "False"],
                        key=f"q_{i}",
                        label_visibility="collapsed"
                    )
                    answers[i] = answer.lower()
                
                elif q_type == 'short_answer':
                    answer = st.text_input(
                        f"Your answer for Q{i+1}",
                        key=f"q_{i}",
                        label_visibility="collapsed"
                    )
                    answers[i] = answer
                
                st.divider()
            
            submit = st.form_submit_button("Submit Quiz")
            
            if submit:
                st.session_state.quiz_answers = answers
                st.session_state.quiz_submitted = True
                st.rerun()
    
    # Show results after submission
    if st.session_state.get("quiz_submitted", False) and st.session_state.get("active_quiz") == quiz_id:
        display_quiz_results(quiz_service, quiz, st.session_state.quiz_answers)


def display_quiz_results(quiz_service: QuizService, quiz: dict, user_answers: dict):
    """Display quiz results with scoring."""
    st.divider()
    st.subheader("📊 Quiz Results")
    
    questions = quiz['questions']
    score = 0
    results_detail = []
    
    for i, question in enumerate(questions):
        user_answer = user_answers.get(i, "")
        correct_answer = question.get('correct_answer', '')
        q_type = question.get('type', 'multiple_choice')
        
        is_correct = quiz_service.grade_answer(q_type, user_answer, correct_answer)
        
        if is_correct:
            score += 1
        
        results_detail.append({
            'question': question['question'],
            'user_answer': user_answer,
            'correct_answer': correct_answer,
            'is_correct': is_correct,
            'explanation': question.get('explanation', '')
        })
    
    # Display score
    percentage = (score / len(questions) * 100) if questions else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score", f"{score}/{len(questions)}")
    with col2:
        st.metric("Percentage", f"{percentage:.1f}%")
    with col3:
        if percentage >= 80:
            st.metric("Grade", "A", delta="Excellent!")
        elif percentage >= 60:
            st.metric("Grade", "B", delta="Good!")
        else:
            st.metric("Grade", "C", delta="Keep studying!")
    
    # Save result
    quiz_service.submit_quiz_result(
        quiz_id=quiz['id'],
        answers=user_answers,
        score=score,
        total_questions=len(questions)
    )
    
    # Detailed results
    st.markdown("### Detailed Review")
    
    for i, detail in enumerate(results_detail, 1):
        with st.expander(f"Question {i} - {'✅ Correct' if detail['is_correct'] else '❌ Incorrect'}"):
            st.markdown(f"**Question:** {detail['question']}")
            st.markdown(f"**Your Answer:** {detail['user_answer']}")
            if not detail['is_correct']:
                st.markdown(f"**Correct Answer:** {detail['correct_answer']}")
            if detail['explanation']:
                st.info(f"**Explanation:** {detail['explanation']}")
    
    # Reset button
    if st.button("Take Another Quiz"):
        st.session_state.active_quiz = None
        st.session_state.quiz_answers = {}
        st.session_state.quiz_submitted = False
        st.rerun()


def render_generate_quiz(quiz_service: QuizService, quiz_generator: QuizGenerator, notes_service=None):
    """Render quiz generation interface."""
    st.subheader("Generate New Quiz")
    
    st.markdown("Create a quiz from your notes or study materials using AI.")
    
    # Check if coming from notes tab
    prefill_note = None
    if st.session_state.get("generate_quiz_from_note") and notes_service:
        note_id = st.session_state.generate_quiz_from_note
        prefill_note = notes_service.get_note(note_id)
        if prefill_note:
            st.info(f"📝 Ready to generate quiz from note: **{prefill_note['title']}**")
    
    with st.form("generate_quiz"):
        title = st.text_input("Quiz Title*", placeholder="e.g., Network Security Basics")
        
        # Content source selection
        default_source = "Select from Notes" if prefill_note else "Paste Text"
        content_source = st.radio(
            "Content Source",
            ["Paste Text", "Select from Notes"],
            index=1 if prefill_note else 0,
            horizontal=True
        )
        
        text_input = ""
        if content_source == "Paste Text":
            text_input = st.text_area(
                "Paste content to generate quiz from*",
                height=200,
                placeholder="Paste your study notes or textbook content here..."
            )
        else:
            if notes_service:
                notes = notes_service.get_all_notes()
                if notes:
                    note_options = {f"{note['title']} ({note.get('folder', 'General')})": note for note in notes}
                    default_idx = 0
                    if prefill_note:
                        prefill_key = f"{prefill_note['title']} ({prefill_note.get('folder', 'General')})"
                        if prefill_key in note_options:
                            default_idx = list(note_options.keys()).index(prefill_key)
                    
                    selected_note_name = st.selectbox("Select Note*", list(note_options.keys()), index=default_idx)
                    if selected_note_name:
                        selected_note = note_options[selected_note_name]
                        text_input = selected_note['content']
                        st.info(f"📝 Using note: **{selected_note['title']}**")
                else:
                    st.warning("No notes available. Create notes first or paste text instead.")
            else:
                st.warning("Notes service not available. Please paste text instead.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            num_questions = st.number_input("Number of Questions", min_value=5, max_value=20, value=10)
        with col2:
            difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"])
        with col3:
            topic = st.text_input("Topic (optional)")
        
        generate = st.form_submit_button("🤖 Generate Quiz")
        
        if generate and title and text_input:
            with st.spinner("Generating quiz questions..."):
                questions = quiz_generator.generate_quiz(
                    text=text_input,
                    num_questions=num_questions,
                    difficulty=difficulty,
                    topic=topic if topic else None
                )
                
                if questions:
                    quiz = quiz_service.create_quiz(
                        title=title,
                        questions=questions,
                        topic=topic if topic else None,
                        difficulty=difficulty
                    )
                    st.success(f"✅ Quiz '{title}' created with {len(questions)} questions!")
                    # Clear the prefill flag
                    if "generate_quiz_from_note" in st.session_state:
                        st.session_state.generate_quiz_from_note = None
                else:
                    st.error("Failed to generate quiz. Please try again with different content.")


def render_quiz_results(quiz_service: QuizService):
    """Render quiz results history."""
    st.subheader("Quiz Results History")
    
    results = quiz_service.get_quiz_results()
    
    if not results:
        st.info("No quiz results yet. Take a quiz to see your progress!")
        return
    
    # Sort by date (most recent first)
    results = sorted(results, key=lambda x: x.get('completed_at', ''), reverse=True)
    
    st.caption(f"Total quizzes taken: {len(results)}")
    
    # Overall stats
    if results:
        avg_percentage = sum(r.get('percentage', 0) for r in results) / len(results)
        best_score = max(r.get('percentage', 0) for r in results)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Score", f"{avg_percentage:.1f}%")
        with col2:
            st.metric("Best Score", f"{best_score:.1f}%")
    
    st.divider()
    
    # Individual results
    for result in results:
        quiz = quiz_service.get_quiz(result['quiz_id'])
        if not quiz:
            continue
        
        with st.expander(f"📊 {quiz['title']} - {result['percentage']:.1f}%", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Score", f"{result['score']}/{result['total_questions']}")
            with col2:
                st.metric("Percentage", f"{result['percentage']:.1f}%")
            with col3:
                st.caption(f"Completed: {result['completed_at'][:10]}")
