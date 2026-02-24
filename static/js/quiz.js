/**
 * quiz.js — Quiz flow + scoring
 */
(function () {
  'use strict';

  // ── State ────────────────────────────────────────────────────────────────
  let allQuizzes = [];
  let activeQuiz = null;
  let currentQIndex = 0;
  let answers = {};
  let score = 0;
  let selectedAnswer = null;
  let answerSubmitted = false;

  // ── Views ────────────────────────────────────────────────────────────────
  const listView    = document.getElementById('quiz-list-view');
  const takeView    = document.getElementById('quiz-take-view');
  const resultsView = document.getElementById('quiz-results-view');
  const emptyState  = document.getElementById('quiz-empty');
  const container   = document.getElementById('quizzes-container');

  // ── Take-quiz elements ───────────────────────────────────────────────────
  const quizTitle    = document.getElementById('quiz-title');
  const quizProgress = document.getElementById('quiz-progress');
  const quizScoreLive= document.getElementById('quiz-score-live');
  const qNum         = document.getElementById('q-num');
  const qText        = document.getElementById('q-text');
  const mcOptions    = document.getElementById('mc-options');
  const tfOptions    = document.getElementById('tf-options');
  const saInput      = document.getElementById('sa-input');
  const saAnswer     = document.getElementById('sa-answer');
  const submitBtn    = document.getElementById('submit-answer-btn');
  const feedback     = document.getElementById('answer-feedback');
  const feedbackRes  = document.getElementById('feedback-result');
  const feedbackExp  = document.getElementById('feedback-explanation');
  const nextBtn      = document.getElementById('next-question-btn');
  const quitBtn      = document.getElementById('quit-quiz-btn');

  // ── Results elements ─────────────────────────────────────────────────────
  const resultsScore    = document.getElementById('results-score');
  const resultsBreakdown= document.getElementById('results-breakdown');
  const resultsIcon     = document.getElementById('results-icon');
  const retakeBtn       = document.getElementById('retake-quiz-btn');
  const backBtn         = document.getElementById('back-to-list-btn');

  // ── Generate modal ───────────────────────────────────────────────────────
  const genQuizBtn    = document.getElementById('generate-quiz-btn');
  const genModal      = document.getElementById('gen-quiz-modal');
  const closeGenBtn   = document.getElementById('close-quiz-gen-btn');
  const cancelGenBtn  = document.getElementById('cancel-quiz-gen-btn');
  const startGenBtn   = document.getElementById('start-quiz-gen-btn');

  // ── Load quizzes ─────────────────────────────────────────────────────────
  function loadQuizzes() {
    apiFetch('/api/quizzes')
      .then(r => r.json())
      .then(quizzes => {
        allQuizzes = quizzes;
        renderQuizList();
      })
      .catch(() => showNotification('Failed to load quizzes', 'error'));
  }

  function renderQuizList() {
    showView('list');
    if (!container) return;
    if (!allQuizzes.length) {
      container.innerHTML = '';
      if (emptyState) emptyState.classList.remove('hidden');
      return;
    }
    if (emptyState) emptyState.classList.add('hidden');
    container.innerHTML = allQuizzes
      .map(q => `
        <div class="quiz-card">
          <div class="quiz-card-info">
            <div class="quiz-card-title">${escHtml(q.title)}</div>
            <div class="quiz-card-meta">${q.questions ? q.questions.length : 0} questions · ${escHtml(q.difficulty || 'medium')} · Taken ${q.times_taken || 0}×</div>
          </div>
          <div class="quiz-card-actions">
            <button class="btn btn-sm btn-primary start-quiz-btn" data-id="${escHtml(q.id)}">▶ Start</button>
            <button class="btn btn-sm btn-danger del-quiz-btn" data-id="${escHtml(q.id)}">🗑️</button>
          </div>
        </div>`)
      .join('');

    container.querySelectorAll('.start-quiz-btn').forEach(btn => {
      btn.addEventListener('click', () => startQuiz(btn.dataset.id));
    });
    container.querySelectorAll('.del-quiz-btn').forEach(btn => {
      btn.addEventListener('click', () => deleteQuiz(btn.dataset.id));
    });
  }

  function deleteQuiz(id) {
    if (!confirm('Delete this quiz?')) return;
    apiFetch('/api/quizzes/' + encodeURIComponent(id) + '/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ answers: {}, score: 0, total_questions: 0 }),
    }).catch(() => {});
    // Optimistic removal
    allQuizzes = allQuizzes.filter(q => q.id !== id);
    renderQuizList();
    showNotification('Quiz deleted', 'info');
  }

  // ── Start quiz ───────────────────────────────────────────────────────────
  function startQuiz(id) {
    const quiz = allQuizzes.find(q => q.id === id);
    if (!quiz) return;
    activeQuiz = quiz;
    currentQIndex = 0;
    answers = {};
    score = 0;
    selectedAnswer = null;
    answerSubmitted = false;

    if (quizTitle) quizTitle.textContent = quiz.title;
    showView('take');
    renderQuestion();
  }

  function renderQuestion() {
    if (!activeQuiz || currentQIndex >= activeQuiz.questions.length) {
      showResults();
      return;
    }

    const q = activeQuiz.questions[currentQIndex];
    selectedAnswer = null;
    answerSubmitted = false;

    if (quizProgress) quizProgress.textContent = `Question ${currentQIndex + 1} of ${activeQuiz.questions.length}`;
    if (quizScoreLive) quizScoreLive.textContent = `Score: ${score}`;
    if (qNum) qNum.textContent = `Q${currentQIndex + 1}`;
    if (qText) qText.textContent = q.question;
    if (feedback) feedback.classList.add('hidden');
    if (submitBtn) { submitBtn.disabled = false; submitBtn.textContent = 'Submit Answer'; }

    // Hide all input types
    if (mcOptions) { mcOptions.classList.add('hidden'); mcOptions.innerHTML = ''; }
    if (tfOptions) {
      tfOptions.classList.add('hidden');
      tfOptions.querySelectorAll('.tf-btn').forEach(b => {
        b.disabled = false;
        b.classList.remove('selected');
      });
    }
    if (saInput) { saInput.classList.add('hidden'); if (saAnswer) saAnswer.value = ''; }

    const type = q.type || 'multiple_choice';
    if (type === 'multiple_choice') {
      if (mcOptions) {
        mcOptions.classList.remove('hidden');
        const opts = q.options || [];
        mcOptions.innerHTML = opts
          .map(o => `<button class="mc-option" data-val="${escHtml(o)}">${escHtml(o)}</button>`)
          .join('');
        mcOptions.querySelectorAll('.mc-option').forEach(btn => {
          btn.addEventListener('click', () => {
            if (answerSubmitted) return;
            mcOptions.querySelectorAll('.mc-option').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            selectedAnswer = btn.dataset.val;
          });
        });
      }
    } else if (type === 'true_false') {
      if (tfOptions) {
        tfOptions.classList.remove('hidden');
        tfOptions.querySelectorAll('.tf-btn').forEach(btn => {
          btn.onclick = () => {
            if (answerSubmitted) return;
            tfOptions.querySelectorAll('.tf-btn').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            selectedAnswer = btn.dataset.val;
          };
        });
      }
    } else {
      // short answer
      if (saInput) saInput.classList.remove('hidden');
    }
  }

  // ── Submit answer ────────────────────────────────────────────────────────
  if (submitBtn) {
    submitBtn.addEventListener('click', () => {
      if (answerSubmitted) return;

      const q = activeQuiz.questions[currentQIndex];
      const type = q.type || 'multiple_choice';

      let userAnswer = selectedAnswer;
      if (type === 'short_answer') {
        userAnswer = saAnswer ? saAnswer.value.trim() : '';
      }

      if (!userAnswer && userAnswer !== false) {
        showNotification('Please select or enter an answer', 'warn');
        return;
      }

      answerSubmitted = true;
      answers[currentQIndex] = userAnswer;

      // Grade answer
      let correct = false;
      const correctAnswer = q.correct_answer;

      if (type === 'short_answer') {
        correct = String(userAnswer).toLowerCase().includes(String(correctAnswer).toLowerCase()) ||
                  String(correctAnswer).toLowerCase().includes(String(userAnswer).toLowerCase());
      } else {
        correct = String(userAnswer).trim().toLowerCase() === String(correctAnswer).trim().toLowerCase();
      }

      if (correct) score++;

      // Show feedback
      if (feedbackRes) {
        feedbackRes.textContent = correct ? '✅ Correct!' : '❌ Incorrect';
        feedbackRes.style.color = correct ? '#2ed573' : '#ff4757';
      }
      if (feedbackExp) {
        let expText = '';
        if (!correct) expText = 'Correct answer: ' + correctAnswer + '. ';
        expText += (q.explanation || '');
        feedbackExp.textContent = expText;
      }
      if (feedback) feedback.classList.remove('hidden');
      if (submitBtn) submitBtn.disabled = true;

      // Highlight MC/TF
      if (type === 'multiple_choice' && mcOptions) {
        mcOptions.querySelectorAll('.mc-option').forEach(btn => {
          btn.disabled = true;
          if (btn.dataset.val === String(correctAnswer)) btn.classList.add('correct');
          else if (btn.dataset.val === String(userAnswer) && !correct) btn.classList.add('wrong');
        });
      }
      if (type === 'true_false' && tfOptions) {
        tfOptions.querySelectorAll('.tf-btn').forEach(btn => {
          btn.disabled = true;
        });
      }

      if (quizScoreLive) quizScoreLive.textContent = `Score: ${score}`;
    });
  }

  // ── Next question ────────────────────────────────────────────────────────
  if (nextBtn) {
    nextBtn.addEventListener('click', () => {
      currentQIndex++;
      if (currentQIndex >= (activeQuiz ? activeQuiz.questions.length : 0)) {
        showResults();
      } else {
        renderQuestion();
      }
    });
  }

  // ── Results ──────────────────────────────────────────────────────────────
  function showResults() {
    showView('results');
    const total = activeQuiz ? activeQuiz.questions.length : 0;
    const pct = total ? Math.round((score / total) * 100) : 0;

    if (resultsIcon) {
      resultsIcon.textContent = pct >= 80 ? '🏆' : pct >= 60 ? '👍' : '📚';
    }
    if (resultsScore) resultsScore.textContent = `${score} / ${total} (${pct}%)`;
    if (resultsBreakdown) {
      resultsBreakdown.textContent = `${pct >= 80 ? 'Excellent!' : pct >= 60 ? 'Good job!' : 'Keep studying!'} You scored ${score} out of ${total} questions.`;
    }

    // Submit to server
    if (activeQuiz) {
      apiFetch('/api/quizzes/' + encodeURIComponent(activeQuiz.id) + '/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ answers, score, total_questions: total }),
      }).catch(() => {});
    }
  }

  if (retakeBtn) {
    retakeBtn.addEventListener('click', () => {
      if (activeQuiz) startQuiz(activeQuiz.id);
    });
  }

  if (backBtn) {
    backBtn.addEventListener('click', () => {
      activeQuiz = null;
      loadQuizzes();
    });
  }

  if (quitBtn) {
    quitBtn.addEventListener('click', () => {
      if (confirm('Quit quiz?')) {
        activeQuiz = null;
        loadQuizzes();
      }
    });
  }

  // ── View switcher ────────────────────────────────────────────────────────
  function showView(view) {
    if (listView)    listView.classList.toggle('hidden',    view !== 'list');
    if (takeView)    takeView.classList.toggle('hidden',    view !== 'take');
    if (resultsView) resultsView.classList.toggle('hidden', view !== 'results');
  }

  // ── Generate quiz modal ──────────────────────────────────────────────────
  if (genQuizBtn) genQuizBtn.addEventListener('click', () => { if (genModal) genModal.classList.remove('hidden'); });
  function closeGenModal() { if (genModal) genModal.classList.add('hidden'); }
  if (closeGenBtn) closeGenBtn.addEventListener('click', closeGenModal);
  if (cancelGenBtn) cancelGenBtn.addEventListener('click', closeGenModal);

  if (startGenBtn) {
    startGenBtn.addEventListener('click', () => {
      const title  = document.getElementById('gen-quiz-title').value.trim() || 'Generated Quiz';
      const text   = document.getElementById('gen-quiz-text').value.trim();
      const count  = parseInt(document.getElementById('gen-quiz-count').value) || 10;
      const diff   = document.getElementById('gen-quiz-diff').value;
      const topic  = document.getElementById('gen-quiz-topic').value.trim() || null;

      if (!text) { showNotification('Please paste some text', 'error'); return; }

      startGenBtn.disabled = true;
      startGenBtn.textContent = '⏳ Generating…';

      apiFetch('/api/generate/quiz', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, text, num_questions: count, difficulty: diff, topic }),
      })
        .then(r => r.json())
        .then(data => {
          if (data.error) { showNotification(data.error, 'error'); return; }
          showNotification(`Quiz "${escHtml(data.title)}" generated ✅`, 'success');
          closeGenModal();
          loadQuizzes();
        })
        .catch(() => showNotification('Generation failed', 'error'))
        .finally(() => {
          startGenBtn.disabled = false;
          startGenBtn.textContent = '🤖 Generate';
        });
    });
  }

  // ── Init ─────────────────────────────────────────────────────────────────
  loadQuizzes();
})();
