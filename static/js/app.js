/**
 * app.js — Shared utilities for CyberBron
 * Provides: CSRF token injection, token auth headers, notification system,
 * HTML escaping, and apiFetch wrapper.
 */
(function () {
  'use strict';

  // ── CSRF token ──────────────────────────────────────────────────────────
  function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute('content') : '';
  }

  // ── Access token (stored in sessionStorage for network mode) ────────────
  function getAccessToken() {
    return sessionStorage.getItem('cb_token') || '';
  }

  // ── Escape HTML to prevent XSS ──────────────────────────────────────────
  function escHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }
  window.escHtml = escHtml;

  // ── apiFetch: fetch wrapper with CSRF + token headers ───────────────────
  function apiFetch(url, options) {
    options = options || {};
    const headers = options.headers ? new Headers(options.headers) : new Headers();

    // CSRF token for state-changing requests
    const method = (options.method || 'GET').toUpperCase();
    if (method !== 'GET' && method !== 'HEAD') {
      headers.set('X-CSRFToken', getCsrfToken());
    }

    // Access token for network mode
    const token = getAccessToken();
    if (token) {
      headers.set('X-Access-Token', token);
    }

    // Keep existing Content-Type if set
    options.headers = headers;
    return fetch(url, options);
  }
  window.apiFetch = apiFetch;

  // ── Notification system ─────────────────────────────────────────────────
  function showNotification(message, type) {
    type = type || 'info';
    const container = getOrCreateNotifContainer();
    const notif = document.createElement('div');
    notif.className = 'notif notif-' + type;
    notif.textContent = message;  // textContent is already XSS-safe
    container.appendChild(notif);
    setTimeout(() => notif.classList.add('show'), 10);
    setTimeout(() => {
      notif.classList.remove('show');
      setTimeout(() => notif.remove(), 300);
    }, 3500);
  }
  window.showNotification = showNotification;

  function getOrCreateNotifContainer() {
    let c = document.getElementById('notif-container');
    if (!c) {
      c = document.createElement('div');
      c.id = 'notif-container';
      c.style.cssText =
        'position:fixed;top:64px;right:16px;z-index:9999;display:flex;flex-direction:column;gap:8px;';
      // Inline styles for notifications (no extra CSS dependency)
      const style = document.createElement('style');
      style.textContent = `
        .notif{padding:.55rem .9rem;border-radius:6px;font-size:.85rem;font-weight:600;
          opacity:0;transform:translateX(20px);transition:all .25s;max-width:320px;}
        .notif.show{opacity:1;transform:translateX(0);}
        .notif-info{background:#21262d;border:1px solid rgba(0,212,255,.4);color:#00d4ff;}
        .notif-success{background:#1a2e1f;border:1px solid rgba(46,213,115,.4);color:#2ed573;}
        .notif-error{background:#2e1a1a;border:1px solid rgba(255,71,87,.4);color:#ff4757;}
        .notif-warn{background:#2e2a1a;border:1px solid rgba(255,165,2,.4);color:#ffa502;}
      `;
      document.head.appendChild(style);
      document.body.appendChild(c);
    }
    return c;
  }

  // ── GPU queue status poller ─────────────────────────────────────────────
  function pollQueueStatus() {
    const badge = document.getElementById('gpu-status');
    if (!badge) return;
    apiFetch('/api/queue/status')
      .then(r => (r.ok ? r.json() : null))
      .then(data => {
        if (!data) return;
        badge.textContent = data.active
          ? `GPU busy${data.queue_depth ? ' (+' + data.queue_depth + ')' : ''}`
          : 'GPU free';
        badge.style.color = data.active ? '#ffa502' : '';
      })
      .catch(() => {});
  }
  setInterval(pollQueueStatus, 4000);
  pollQueueStatus();

  // ── Utility: format date ────────────────────────────────────────────────
  function fmtDate(isoStr) {
    if (!isoStr) return '';
    try {
      return new Date(isoStr).toLocaleDateString(undefined, {
        year: 'numeric', month: 'short', day: 'numeric',
      });
    } catch (_) { return isoStr; }
  }
  window.fmtDate = fmtDate;

})();
