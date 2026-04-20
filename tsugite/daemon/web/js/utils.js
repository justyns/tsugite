import { marked } from './vendor/marked.esm.min.js';

marked.setOptions({ gfm: true, breaks: true });

export function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

export function renderMarkdown(text) {
  const html = marked.parse(text ?? '');
  return html.replace(/<pre>/g, '<pre><button class="copy-code" type="button" aria-label="Copy code" title="Copy code">\u29c9</button>');
}

export function fmtTokens(n) {
  if (!n) return '-';
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return String(n);
}

export function fmtCost(c) {
  if (c == null) return '-';
  if (c < 0.01) return '$' + c.toFixed(4);
  return '$' + c.toFixed(2);
}

export function formatFileSize(bytes) {
  if (bytes == null) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatDate(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export function contentBlockHtml(name, content) {
  return `<details class="content-block"><summary><code>${escapeHtml(name)}</code> (content block)</summary><pre><code>${escapeHtml(content)}</code></pre></details>`;
}

export function truncate(s, max = 500) {
  return s.length > max ? s.slice(0, max) + '...' : s;
}

export function scrollToBottom(el) {
  el.scrollTop = el.scrollHeight;
}

export function toast(text, kind = 'success', duration = 2200) {
  window.dispatchEvent(new CustomEvent('tsugite:toast', { detail: { text, kind, duration } }));
}

function fallbackCopy(text) {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.setAttribute('readonly', '');
  ta.style.position = 'fixed';
  ta.style.top = '-1000px';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  let ok = false;
  try { ok = document.execCommand('copy'); } catch { ok = false; }
  document.body.removeChild(ta);
  return ok;
}

export async function copyText(text) {
  const s = text ?? '';
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(s);
    } else if (!fallbackCopy(s)) {
      throw new Error('copy failed');
    }
    toast('Copied');
    return true;
  } catch {
    if (fallbackCopy(s)) {
      toast('Copied');
      return true;
    }
    toast('Copy failed', 'error');
    return false;
  }
}

const _stateBadgeMap = {
  pending: 'badge-muted', active: 'badge-ok', running: 'badge-accent',
  completed: 'badge-ok', failed: 'badge-error', error: 'badge-error',
  cancelled: 'badge-muted',
};

export function stateBadgeClass(state) {
  return _stateBadgeMap[state] || 'badge-muted';
}

