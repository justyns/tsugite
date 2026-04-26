import Alpine from 'https://cdn.jsdelivr.net/npm/alpinejs@3/dist/module.esm.js';
import { get, post, patch, connectEvents, onAuthRequired } from './api.js';
import conversationsView from './views/conversations.js';
import scheduleView from './views/schedules.js';
import webhookView from './views/webhooks.js';
import fileEditorView from './views/file-editor.js';
import workspaceView from './views/workspace.js';
import usageView from './views/usage.js';

window.Alpine = Alpine;
window.tsugiteApi = { get, post, patch };

function parseHash(hash) {
  const [view, query] = hash.split('?');
  const params = new URLSearchParams(query || '');
  return { view: view || '', sessionId: params.get('session') || '' };
}

const legacyViews = { chat: 'conversations', sessions: 'conversations', history: 'conversations' };
const initialParsed = parseHash(location.hash.slice(1));
const initialView = legacyViews[initialParsed.view] || initialParsed.view || localStorage.getItem('tsugite-view') || 'conversations';

Alpine.store('app', {
  tabs: ['conversations','workspace','agents','skills','schedules','webhooks','usage'],
  agents: [],
  selectedAgent: localStorage.getItem('tsugite-agent') || null,
  view: initialView,
  theme: localStorage.getItem('tsugite_theme') || 'frappe',
  userId: localStorage.getItem('tsugite_user_id') || 'web-user-1',
  authRequired: !localStorage.getItem('tsugite_token'),
  showSettings: false,
  menuOpen: false,
  lastEvent: null,
  viewSessionId: initialParsed.sessionId || null,
  pendingWorkspaceFiles: [],
  autoFollow: localStorage.getItem('tsugite_auto_follow') !== 'false',
  skillIssues: [],
  tokensTotal: null,  // updated by usageView.load() so the keystrip shows daily token count
});

Alpine.data('conversationsView', conversationsView);
Alpine.data('scheduleView', scheduleView);
Alpine.data('webhookView', webhookView);
Alpine.data('agentFileView', fileEditorView('agents', 'agent-files'));
Alpine.data('skillFileView', fileEditorView('skills', 'skill-files'));
Alpine.data('workspaceView', workspaceView);
Alpine.data('usageView', usageView);

window.addEventListener('hashchange', () => {
  const { view, sessionId } = parseHash(location.hash.slice(1));
  const store = Alpine.store('app');
  if (view) store.view = legacyViews[view] || view;
  if (sessionId) store.viewSessionId = sessionId;
});

Alpine.start();

// Persist view and agent selection to localStorage
Alpine.effect(() => {
  const store = Alpine.store('app');
  localStorage.setItem('tsugite-view', store.view);
  const currentView = parseHash(location.hash.slice(1)).view;
  if (currentView !== store.view) location.hash = store.view;
  if (store.selectedAgent) localStorage.setItem('tsugite-agent', store.selectedAgent);
});

// Keep <meta name="theme-color"> in sync with the active theme's --crust so
// the OS/browser chrome (PWA status bar, mobile address bar) matches the IDE
// tab strip instead of flashing the previous theme on switch.
Alpine.effect(() => {
  const theme = Alpine.store('app').theme;
  void theme;  // reactive dep
  const meta = document.querySelector('meta[name="theme-color"]');
  if (!meta) return;
  // Defer one frame so the new --crust is resolved against the new data-theme.
  requestAnimationFrame(() => {
    const crust = getComputedStyle(document.body).getPropertyValue('--crust').trim();
    if (crust) meta.setAttribute('content', crust);
  });
});

let _es = null;

onAuthRequired(() => {
  const store = Alpine.store('app');
  if (store.authRequired) return;
  store.authRequired = true;
  if (_es) { _es.close(); _es = null; }
});

async function loadAgents() {
  const data = await get('/api/agents');
  const store = Alpine.store('app');
  store.agents = data.agents || [];
  store.authRequired = false;
  if (store.agents.length && !store.selectedAgent) {
    store.selectedAgent = store.agents[0].name;
  }
  connectSSE();
  loadSkillIssues().catch(() => {});
}
window.tsugiteLoadAgents = loadAgents;

async function loadSkillIssues() {
  try {
    const data = await get('/api/skills/issues');
    Alpine.store('app').skillIssues = data.issues || [];
  } catch {
    /* keep prior state on transient failure */
  }
}
window.tsugiteLoadSkillIssues = loadSkillIssues;

function connectSSE() {
  if (_es) return;
  _es = connectEvents((event) => {
    if (event.type === 'reconnect') {
      loadAgents().catch(() => {});
    }
    Alpine.store('app').lastEvent = { ...event, _ts: Date.now() };
  });
}

if (localStorage.getItem('tsugite_token')) {
  loadAgents().catch(() => {});
}

// When the PWA resumes from background (mobile suspend, tab switch),
// reconnect SSE and reload agents to ensure fresh state.
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState !== 'visible') return;
  if (!localStorage.getItem('tsugite_token')) return;
  if (_es) { _es.close(); _es = null; }
  loadAgents().catch(() => {});
});

// In standalone PWA mode, force external links to open in the system browser
// target="_blank" alone is unreliable across platforms (especially iOS Safari)
if (window.matchMedia('(display-mode: standalone)').matches || navigator.standalone) {
  document.addEventListener('click', (e) => {
    const anchor = e.target.closest('a[href]');
    if (!anchor) return;
    const url = anchor.getAttribute('href');
    if (!url || url.startsWith('#') || url.startsWith('javascript:')) return;
    try {
      const linkUrl = new URL(url, location.href);
      if (linkUrl.origin !== location.origin) {
        e.preventDefault();
        window.open(linkUrl.href, '_blank');
      }
    } catch { /* malformed URL, let browser handle it */ }
  });
}

// Service worker + push notifications
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/sw.js').then((reg) => {
    const notifyIfWaiting = (worker) => {
      if (worker && worker.state === 'installed' && navigator.serviceWorker.controller) {
        window.dispatchEvent(new CustomEvent('tsugite:update-available'));
      }
    };
    notifyIfWaiting(reg.waiting);
    reg.addEventListener('updatefound', () => {
      const nw = reg.installing;
      if (!nw) return;
      nw.addEventListener('statechange', () => notifyIfWaiting(nw));
    });
    let reloading = false;
    navigator.serviceWorker.addEventListener('controllerchange', () => {
      if (reloading) return;
      reloading = true;
      location.reload();
    });
    window.tsugiteApplyUpdate = () => {
      const w = reg.waiting || reg.installing;
      if (w) w.postMessage({ type: 'SKIP_WAITING' });
      else location.reload();
    };
  }).catch(() => {});
}

window.tsugiteSubscribePush = async function() {
  const reg = await navigator.serviceWorker.ready;
  const { public_key } = await get('/api/push/vapid-key');
  const sub = await reg.pushManager.subscribe({
    userVisibleOnly: true,
    applicationServerKey: Uint8Array.from(atob(public_key.replace(/-/g, '+').replace(/_/g, '/')), c => c.charCodeAt(0)),
  });
  await post('/api/push/subscribe', sub.toJSON());
};

window.tsugiteUnsubscribePush = async function() {
  const reg = await navigator.serviceWorker.ready;
  const sub = await reg.pushManager.getSubscription();
  if (sub) {
    const endpoint = sub.endpoint;
    await sub.unsubscribe();
    await post('/api/push/unsubscribe', { endpoint });
  }
};
