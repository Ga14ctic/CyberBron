/**
 * chat.js — Streaming chat via fetch + SSE
 * Handles: conversation management, message rendering, SSE streaming.
 */
(function () {
  'use strict';

  // ── State ────────────────────────────────────────────────────────────────
  let currentSessionId = null;
  let isStreaming = false;
  let lastAssistantMsgEl = null;

  const chatMessages = document.getElementById('chat-messages');
  const chatInput = document.getElementById('chat-input');
  const sendBtn = document.getElementById('send-btn');
  const newChatBtn = document.getElementById('new-chat-btn');
  const convList = document.getElementById('conv-list');
  const searchIndicator = document.getElementById('search-indicator');

  // ── Init: pick first conversation or start fresh ─────────────────────────
  (function init() {
    const firstItem = convList ? convList.querySelector('.conv-item') : null;
    if (firstItem) {
      currentSessionId = firstItem.dataset.id;
      loadConversation(currentSessionId);
      firstItem.classList.add('active');
    } else {
      currentSessionId = newSessionId();
    }
    if (convList) setupConvListeners();
  })();

  function newSessionId() {
    return Date.now() + '.json';
  }

  // ── Conversation list ────────────────────────────────────────────────────
  function setupConvListeners() {
    convList.addEventListener('click', function (e) {
      // Delete button
      if (e.target.classList.contains('conv-del')) {
        e.stopPropagation();
        const id = e.target.dataset.id;
        if (confirm('Delete this conversation?')) deleteConversation(id);
        return;
      }
      // Load conversation
      const item = e.target.closest('.conv-item');
      if (item) {
        const id = item.dataset.id;
        convList.querySelectorAll('.conv-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');
        currentSessionId = id;
        loadConversation(id);
      }
    });
  }

  function loadConversation(id) {
    apiFetch('/api/conversations/' + encodeURIComponent(id))
      .then(r => r.json())
      .then(messages => {
        clearMessages();
        if (messages && messages.length) {
          messages.forEach(msg => appendMessage(msg.role, msg.content));
        } else {
          showWelcome();
        }
        scrollToBottom();
      })
      .catch(() => showWelcome());
  }

  function deleteConversation(id) {
    apiFetch('/api/conversations/' + encodeURIComponent(id), { method: 'DELETE' })
      .then(() => {
        // Remove from DOM
        const item = convList.querySelector(`.conv-item[data-id="${CSS.escape(id)}"]`);
        if (item) item.remove();
        if (currentSessionId === id) {
          currentSessionId = newSessionId();
          clearMessages();
          showWelcome();
        }
      })
      .catch(() => showNotification('Failed to delete conversation', 'error'));
  }

  if (newChatBtn) {
    newChatBtn.addEventListener('click', function () {
      currentSessionId = newSessionId();
      convList.querySelectorAll('.conv-item').forEach(i => i.classList.remove('active'));
      clearMessages();
      showWelcome();
    });
  }

  // ── Message rendering ────────────────────────────────────────────────────
  function clearMessages() {
    chatMessages.innerHTML = '';
  }

  function showWelcome() {
    chatMessages.innerHTML = `
      <div class="chat-welcome">
        <div class="welcome-icon">🛡️</div>
        <h2>CyberBron</h2>
        <p>Your AI-powered cybersecurity study assistant.</p>
        <p class="welcome-hint">Ask me anything about cybersecurity.</p>
      </div>`;
  }

  function appendMessage(role, content) {
    // Remove welcome if present
    const welcome = chatMessages.querySelector('.chat-welcome');
    if (welcome) welcome.remove();

    const msgDiv = document.createElement('div');
    msgDiv.className = 'msg ' + role;

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    // Content is sanitised server-side; render as text to prevent XSS
    bubble.textContent = content;

    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.textContent = role === 'user' ? 'You' : 'CyberBron';

    msgDiv.appendChild(meta);
    msgDiv.appendChild(bubble);

    if (role === 'assistant') {
      // Quick-action buttons
      const actions = buildMsgActions(content);
      msgDiv.appendChild(actions);
    }

    chatMessages.appendChild(msgDiv);
    return msgDiv;
  }

  function buildMsgActions(content) {
    const div = document.createElement('div');
    div.className = 'msg-actions';

    const saveBtn = document.createElement('button');
    saveBtn.className = 'msg-action-btn';
    saveBtn.textContent = '📝 Save to Notes';
    saveBtn.addEventListener('click', () => saveToNotes(content));

    div.appendChild(saveBtn);
    return div;
  }

  function saveToNotes(content) {
    const title = 'Chat Note ' + new Date().toLocaleDateString();
    apiFetch('/api/notes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title, content, folder: 'Chat Notes' }),
    })
      .then(r => (r.ok ? showNotification('Saved to notes ✅', 'success') : showNotification('Save failed', 'error')))
      .catch(() => showNotification('Save failed', 'error'));
  }

  // ── Streaming response ───────────────────────────────────────────────────
  function appendStreamingMessage() {
    const welcome = chatMessages.querySelector('.chat-welcome');
    if (welcome) welcome.remove();

    const msgDiv = document.createElement('div');
    msgDiv.className = 'msg assistant';

    const meta = document.createElement('div');
    meta.className = 'msg-meta';
    meta.textContent = 'CyberBron';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.innerHTML =
      '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';

    msgDiv.appendChild(meta);
    msgDiv.appendChild(bubble);
    chatMessages.appendChild(msgDiv);
    lastAssistantMsgEl = bubble;
    scrollToBottom();
    return bubble;
  }

  function appendSearchResults(results) {
    if (!results || !results.length) return;
    const div = document.createElement('div');
    div.className = 'search-results';
    div.innerHTML =
      '<div class="source-badge">🌐 Web Search Results</div>' +
      results
        .map(
          r => `<div class="search-result-item">
          <a href="${escHtml(r.link)}" target="_blank" rel="noopener noreferrer">${escHtml(r.title)}</a>
          <div class="snippet">${escHtml(r.snippet)}</div>
        </div>`
        )
        .join('');
    chatMessages.appendChild(div);
  }

  // ── Send message ─────────────────────────────────────────────────────────
  function sendMessage() {
    const msg = chatInput.value.trim();
    if (!msg || isStreaming) return;

    isStreaming = true;
    sendBtn.disabled = true;
    sendBtn.textContent = '…';

    appendMessage('user', msg);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    scrollToBottom();

    const streamBubble = appendStreamingMessage();
    let accumulated = '';
    let initialized = false;

    apiFetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, session_id: currentSessionId }),
    })
      .then(response => {
        if (!response.ok) {
          return response.json().then(err => {
            throw new Error(err.error || 'Request failed');
          });
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        function read() {
          return reader.read().then(({ done, value }) => {
            if (done) {
              finishStream();
              return;
            }
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // incomplete line

            for (const line of lines) {
              if (!line.startsWith('data: ')) continue;
              const rawJson = line.slice(6).trim();
              if (!rawJson) continue;
              try {
                const data = JSON.parse(rawJson);
                if (data.chunk) {
                  if (!initialized) {
                    streamBubble.textContent = '';
                    initialized = true;
                  }
                  accumulated += data.chunk;
                  streamBubble.textContent = accumulated;
                  scrollToBottom();
                }
                if (data.done) {
                  currentSessionId = data.session_id || currentSessionId;
                  if (data.search_results) {
                    appendSearchResults(data.search_results);
                  }
                  updateConvList(currentSessionId, msg);
                  finishStream();
                  return;
                }
                if (data.error) {
                  streamBubble.textContent = '❌ ' + data.error;
                  finishStream();
                  return;
                }
              } catch (_) {}
            }
            return read();
          });
        }

        return read();
      })
      .catch(err => {
        streamBubble.textContent = '❌ ' + (err.message || 'Failed to connect');
        finishStream();
      });
  }

  function finishStream() {
    isStreaming = false;
    sendBtn.disabled = false;
    sendBtn.textContent = 'Send';
    scrollToBottom();
    // Add action buttons to last message
    if (lastAssistantMsgEl) {
      const msgDiv = lastAssistantMsgEl.closest('.msg');
      if (msgDiv && !msgDiv.querySelector('.msg-actions')) {
        const actions = buildMsgActions(lastAssistantMsgEl.textContent || '');
        msgDiv.appendChild(actions);
      }
    }
  }

  function updateConvList(id, preview) {
    if (!convList) return;
    let item = convList.querySelector(`.conv-item[data-id="${CSS.escape(id)}"]`);
    if (!item) {
      item = document.createElement('div');
      item.className = 'conv-item';
      item.dataset.id = id;
      item.innerHTML =
        `<span class="conv-preview"></span>` +
        `<button class="conv-del" data-id="${escHtml(id)}" title="Delete">✕</button>`;
      convList.insertBefore(item, convList.firstChild);
    }
    convList.querySelectorAll('.conv-item').forEach(i => i.classList.remove('active'));
    item.classList.add('active');
    const previewEl = item.querySelector('.conv-preview');
    if (previewEl) previewEl.textContent = preview.slice(0, 60) + (preview.length > 60 ? '…' : '');
  }

  // ── Input event handlers ─────────────────────────────────────────────────
  if (sendBtn) sendBtn.addEventListener('click', sendMessage);
  if (chatInput) {
    chatInput.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
    // Auto-resize
    chatInput.addEventListener('input', function () {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 150) + 'px';
    });
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
})();
