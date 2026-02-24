/**
 * flashcards.js — Flashcard management and study mode
 */
(function () {
  'use strict';

  // ── State ────────────────────────────────────────────────────────────────
  let allCards = [];
  let allDecks = [];
  let studyDeck = [];
  let studyIndex = 0;
  let studyFlipped = false;

  // ── Elements ─────────────────────────────────────────────────────────────
  const deckSelect   = document.getElementById('deck-select');
  const deckCount    = document.getElementById('deck-count');
  const cardsGrid    = document.getElementById('cards-grid');
  const studyBtn     = document.getElementById('study-btn');
  const studyOverlay = document.getElementById('study-overlay');
  const endStudyBtn  = document.getElementById('end-study-btn');
  const flashcard    = document.getElementById('flashcard');
  const flipBtn      = document.getElementById('flip-btn');
  const cardQuestion = document.getElementById('card-question');
  const cardAnswer   = document.getElementById('card-answer');
  const studyProgress= document.getElementById('study-progress');

  const createCardBtn  = document.getElementById('create-card-btn');
  const cardModal      = document.getElementById('card-modal');
  const closeModalBtn  = document.getElementById('close-modal-btn');
  const cancelCardBtn  = document.getElementById('cancel-card-btn');
  const saveCardBtn    = document.getElementById('save-card-btn');

  const generateCardsBtn = document.getElementById('generate-cards-btn');
  const genModal         = document.getElementById('gen-modal');
  const closeGenBtn      = document.getElementById('close-gen-btn');
  const cancelGenBtn     = document.getElementById('cancel-gen-btn');
  const startGenBtn      = document.getElementById('start-gen-btn');

  let editingCardId = null;

  // ── Load flashcards ──────────────────────────────────────────────────────
  function loadCards() {
    const deck = deckSelect ? deckSelect.value : '';
    const url = deck ? '/api/flashcards?deck=' + encodeURIComponent(deck) : '/api/flashcards';
    apiFetch(url)
      .then(r => r.json())
      .then(data => {
        allCards = data.cards || [];
        allDecks = data.decks || [];
        renderDeckSelect();
        renderCards();
      })
      .catch(() => showNotification('Failed to load flashcards', 'error'));
  }

  function renderDeckSelect() {
    if (!deckSelect) return;
    const current = deckSelect.value;
    deckSelect.innerHTML = '<option value="">All Decks</option>';
    allDecks.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d;
      opt.textContent = d;
      if (d === current) opt.selected = true;
      deckSelect.appendChild(opt);
    });
  }

  function renderCards() {
    if (deckCount) deckCount.textContent = allCards.length + ' cards';
    if (!cardsGrid) return;
    if (!allCards.length) {
      cardsGrid.innerHTML = '<div class="empty-text">No flashcards yet. Create or generate some!</div>';
      return;
    }
    cardsGrid.innerHTML = allCards
      .map(c => `
        <div class="card-item">
          <div class="card-question">${escHtml(c.question)}</div>
          <div class="card-answer">${escHtml(c.answer.slice(0, 100))}${c.answer.length > 100 ? '…' : ''}</div>
          <div class="card-actions">
            <button class="btn btn-sm btn-secondary edit-card-btn" data-id="${escHtml(c.id)}">✏️ Edit</button>
            <button class="btn btn-sm btn-danger del-card-btn" data-id="${escHtml(c.id)}">🗑️</button>
          </div>
        </div>`)
      .join('');

    cardsGrid.querySelectorAll('.edit-card-btn').forEach(btn => {
      btn.addEventListener('click', () => openEditModal(btn.dataset.id));
    });
    cardsGrid.querySelectorAll('.del-card-btn').forEach(btn => {
      btn.addEventListener('click', () => deleteCard(btn.dataset.id));
    });
  }

  // ── Study mode ───────────────────────────────────────────────────────────
  if (studyBtn) {
    studyBtn.addEventListener('click', () => {
      if (!allCards.length) { showNotification('No cards to study', 'warn'); return; }
      studyDeck = [...allCards].sort(() => Math.random() - 0.5);
      studyIndex = 0;
      showStudyCard();
      if (studyOverlay) studyOverlay.classList.remove('hidden');
    });
  }

  if (endStudyBtn) {
    endStudyBtn.addEventListener('click', () => {
      if (studyOverlay) studyOverlay.classList.add('hidden');
    });
  }

  if (flipBtn) {
    flipBtn.addEventListener('click', () => {
      if (flashcard) flashcard.classList.add('flipped');
      studyFlipped = true;
    });
  }

  function showStudyCard() {
    if (!studyDeck.length) return;
    const card = studyDeck[studyIndex];
    if (cardQuestion) cardQuestion.textContent = card.question;
    if (cardAnswer) cardAnswer.textContent = card.answer;
    if (flashcard) flashcard.classList.remove('flipped');
    studyFlipped = false;
    if (studyProgress) {
      studyProgress.textContent = `Card ${studyIndex + 1} of ${studyDeck.length}`;
    }
  }

  // Difficulty buttons
  document.querySelectorAll('.diff-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const diff = btn.dataset.diff;
      const card = studyDeck[studyIndex];
      if (card) {
        apiFetch('/api/flashcards/' + encodeURIComponent(card.id) + '/review', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ difficulty: diff }),
        }).catch(() => {});
      }
      studyIndex++;
      if (studyIndex >= studyDeck.length) {
        if (studyOverlay) studyOverlay.classList.add('hidden');
        showNotification('Study session complete! 🎉', 'success');
        loadCards();
      } else {
        showStudyCard();
      }
    });
  });

  // ── Create / Edit modal ──────────────────────────────────────────────────
  function openEditModal(id) {
    const card = allCards.find(c => c.id === id);
    if (!card) return;
    editingCardId = id;
    document.getElementById('modal-title').textContent = 'Edit Flashcard';
    document.getElementById('modal-question').value = card.question;
    document.getElementById('modal-answer').value = card.answer;
    document.getElementById('modal-deck').value = card.deck || 'General';
    document.getElementById('modal-topic').value = card.topic || '';
    if (cardModal) cardModal.classList.remove('hidden');
  }

  if (createCardBtn) {
    createCardBtn.addEventListener('click', () => {
      editingCardId = null;
      document.getElementById('modal-title').textContent = 'Create Flashcard';
      document.getElementById('modal-question').value = '';
      document.getElementById('modal-answer').value = '';
      document.getElementById('modal-deck').value = deckSelect ? deckSelect.value || 'General' : 'General';
      document.getElementById('modal-topic').value = '';
      if (cardModal) cardModal.classList.remove('hidden');
    });
  }

  function closeCardModal() {
    if (cardModal) cardModal.classList.add('hidden');
    editingCardId = null;
  }

  if (closeModalBtn) closeModalBtn.addEventListener('click', closeCardModal);
  if (cancelCardBtn) cancelCardBtn.addEventListener('click', closeCardModal);

  if (saveCardBtn) {
    saveCardBtn.addEventListener('click', () => {
      const question = document.getElementById('modal-question').value.trim();
      const answer   = document.getElementById('modal-answer').value.trim();
      const deck     = document.getElementById('modal-deck').value.trim() || 'General';
      const topic    = document.getElementById('modal-topic').value.trim() || null;

      if (!question || !answer) {
        showNotification('Question and answer are required', 'error');
        return;
      }
      const body = JSON.stringify({ question, answer, deck, topic });

      if (editingCardId) {
        apiFetch('/api/flashcards/' + encodeURIComponent(editingCardId), {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body,
        })
          .then(() => { showNotification('Card updated ✅', 'success'); closeCardModal(); loadCards(); })
          .catch(() => showNotification('Update failed', 'error'));
      } else {
        apiFetch('/api/flashcards', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body,
        })
          .then(() => { showNotification('Card created ✅', 'success'); closeCardModal(); loadCards(); })
          .catch(() => showNotification('Create failed', 'error'));
      }
    });
  }

  // ── Delete card ──────────────────────────────────────────────────────────
  function deleteCard(id) {
    if (!confirm('Delete this flashcard?')) return;
    apiFetch('/api/flashcards/' + encodeURIComponent(id), { method: 'DELETE' })
      .then(() => { showNotification('Card deleted', 'info'); loadCards(); })
      .catch(() => showNotification('Delete failed', 'error'));
  }

  // ── Generate flashcards modal ────────────────────────────────────────────
  if (generateCardsBtn) generateCardsBtn.addEventListener('click', () => {
    if (genModal) genModal.classList.remove('hidden');
  });
  if (closeGenBtn) closeGenBtn.addEventListener('click', () => { if (genModal) genModal.classList.add('hidden'); });
  if (cancelGenBtn) cancelGenBtn.addEventListener('click', () => { if (genModal) genModal.classList.add('hidden'); });

  if (startGenBtn) {
    startGenBtn.addEventListener('click', () => {
      const text  = document.getElementById('gen-text').value.trim();
      const count = parseInt(document.getElementById('gen-count').value) || 10;
      const deck  = document.getElementById('gen-deck').value.trim() || 'Generated';
      const topic = document.getElementById('gen-topic').value.trim() || null;

      if (!text) { showNotification('Please paste some text', 'error'); return; }

      startGenBtn.disabled = true;
      startGenBtn.textContent = '⏳ Generating…';

      apiFetch('/api/generate/flashcards', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, num_cards: count, deck, topic }),
      })
        .then(r => r.json())
        .then(data => {
          if (data.error) { showNotification(data.error, 'error'); return; }
          showNotification(`Generated ${data.count} flashcards ✅`, 'success');
          if (genModal) genModal.classList.add('hidden');
          if (deckSelect) deckSelect.value = deck;
          loadCards();
        })
        .catch(() => showNotification('Generation failed', 'error'))
        .finally(() => {
          startGenBtn.disabled = false;
          startGenBtn.textContent = '🤖 Generate';
        });
    });
  }

  // ── Deck change ──────────────────────────────────────────────────────────
  if (deckSelect) deckSelect.addEventListener('change', loadCards);

  // ── Init ─────────────────────────────────────────────────────────────────
  loadCards();
})();
