/**
 * notes.js — Notes CRUD via fetch API
 */
(function () {
  'use strict';

  const notesList = document.getElementById('notes-list');
  const noteSearch = document.getElementById('note-search');
  const newNoteBtn = document.getElementById('new-note-btn');
  const noteEditor = document.getElementById('note-editor');
  const noteEmpty = document.getElementById('note-empty');
  const noteTitle = document.getElementById('note-title');
  const noteContent = document.getElementById('note-content');
  const noteFolder = document.getElementById('note-folder');
  const noteTags = document.getElementById('note-tags');
  const saveNoteBtn = document.getElementById('save-note-btn');
  const deleteNoteBtn = document.getElementById('delete-note-btn');
  const cancelNoteBtn = document.getElementById('cancel-note-btn');

  let currentNoteId = null;
  let allNotes = [];
  let searchTimer = null;

  // ── Load notes ───────────────────────────────────────────────────────────
  function loadNotes(query) {
    const url = query ? '/api/notes?q=' + encodeURIComponent(query) : '/api/notes';
    apiFetch(url)
      .then(r => r.json())
      .then(notes => {
        allNotes = notes;
        renderNotesList(notes);
      })
      .catch(() => showNotification('Failed to load notes', 'error'));
  }

  function renderNotesList(notes) {
    if (!notesList) return;
    if (!notes.length) {
      notesList.innerHTML = '<div class="empty-text">No notes found.</div>';
      return;
    }
    notesList.innerHTML = notes
      .map(n => `
        <div class="note-item ${n.id === currentNoteId ? 'active' : ''}" data-id="${escHtml(n.id)}">
          <div class="note-title">${escHtml(n.title)}</div>
          <div class="note-meta">${escHtml(n.folder)} · ${fmtDate(n.updated_at)}</div>
        </div>`)
      .join('');

    notesList.querySelectorAll('.note-item').forEach(item => {
      item.addEventListener('click', () => openNote(item.dataset.id));
    });
  }

  // ── Open / create note ───────────────────────────────────────────────────
  function openNote(id) {
    const note = allNotes.find(n => n.id === id);
    if (!note) return;
    currentNoteId = id;
    noteTitle.value = note.title || '';
    noteContent.value = note.content || '';
    noteFolder.value = note.folder || 'General';
    noteTags.value = (note.tags || []).join(', ');
    showEditor();
    notesList.querySelectorAll('.note-item').forEach(i => {
      i.classList.toggle('active', i.dataset.id === id);
    });
  }

  function showEditor() {
    if (noteEditor) noteEditor.classList.remove('hidden');
    if (noteEmpty) noteEmpty.classList.add('hidden');
    if (deleteNoteBtn) deleteNoteBtn.style.display = currentNoteId ? '' : 'none';
  }

  function hideEditor() {
    if (noteEditor) noteEditor.classList.add('hidden');
    if (noteEmpty) noteEmpty.classList.remove('hidden');
    currentNoteId = null;
  }

  // ── New note ─────────────────────────────────────────────────────────────
  if (newNoteBtn) {
    newNoteBtn.addEventListener('click', () => {
      currentNoteId = null;
      noteTitle.value = '';
      noteContent.value = '';
      noteFolder.value = 'General';
      noteTags.value = '';
      showEditor();
      noteTitle.focus();
    });
  }

  // ── Save note ────────────────────────────────────────────────────────────
  if (saveNoteBtn) {
    saveNoteBtn.addEventListener('click', saveNote);
  }

  function saveNote() {
    const title = noteTitle.value.trim();
    if (!title) { showNotification('Title is required', 'error'); return; }
    const content = noteContent.value;
    const folder = noteFolder.value.trim() || 'General';
    const tags = noteTags.value
      .split(',')
      .map(t => t.trim())
      .filter(Boolean);

    const body = JSON.stringify({ title, content, folder, tags });

    if (currentNoteId) {
      apiFetch('/api/notes/' + encodeURIComponent(currentNoteId), {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body,
      })
        .then(r => r.json())
        .then(note => {
          showNotification('Note saved ✅', 'success');
          loadNotes();
        })
        .catch(() => showNotification('Save failed', 'error'));
    } else {
      apiFetch('/api/notes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      })
        .then(r => r.json())
        .then(note => {
          currentNoteId = note.id;
          showNotification('Note created ✅', 'success');
          loadNotes();
          showEditor();
        })
        .catch(() => showNotification('Create failed', 'error'));
    }
  }

  // ── Delete note ──────────────────────────────────────────────────────────
  if (deleteNoteBtn) {
    deleteNoteBtn.addEventListener('click', () => {
      if (!currentNoteId) return;
      if (!confirm('Delete this note?')) return;
      apiFetch('/api/notes/' + encodeURIComponent(currentNoteId), { method: 'DELETE' })
        .then(() => {
          showNotification('Note deleted', 'info');
          hideEditor();
          loadNotes();
        })
        .catch(() => showNotification('Delete failed', 'error'));
    });
  }

  // ── Cancel ───────────────────────────────────────────────────────────────
  if (cancelNoteBtn) {
    cancelNoteBtn.addEventListener('click', hideEditor);
  }

  // ── Search ───────────────────────────────────────────────────────────────
  if (noteSearch) {
    noteSearch.addEventListener('input', () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => loadNotes(noteSearch.value.trim()), 300);
    });
  }

  // ── Init ─────────────────────────────────────────────────────────────────
  loadNotes();
})();
