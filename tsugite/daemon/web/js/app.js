import Alpine from 'https://cdn.jsdelivr.net/npm/alpinejs@3/dist/module.esm.js';
import { get, post, patch, connectEvents } from './api.js';
import conversationsView from './views/conversations.js';
import dashboardView from './views/dashboard.js';
import scheduleView from './views/schedules.js';
import webhookView from './views/webhooks.js';
import fileEditorView from './views/file-editor.js';
import workspaceView from './views/workspace.js';
import kvstoreView from './views/kvstore.js';

window.Alpine = Alpine;
window.tsugiteApi = { get, post, patch };

const legacyViews = { chat: 'conversations', sessions: 'conversations', history: 'conversations' };
const initialHash = location.hash.slice(1);
const initialView = legacyViews[initialHash] || initialHash || localStorage.getItem('tsugite-view') || 'dashboard';

Alpine.store('app', {
  tabs: ['dashboard','conversations','workspace','agents','skills','schedules','webhooks','kvstore'],
  agents: [],
  selectedAgent: localStorage.getItem('tsugite-agent') || null,
  view: initialView,
  theme: localStorage.getItem('tsugite_theme') || 'frappe',
  userId: localStorage.getItem('tsugite_user_id') || 'web-user-1',
  showSettings: false,
  menuOpen: false,
  lastEvent: null,
  viewSessionId: null,
  pendingWorkspaceFiles: [],
  autoFollow: localStorage.getItem('tsugite_auto_follow') !== 'false',
});

Alpine.data('conversationsView', conversationsView);
Alpine.data('dashboardView', dashboardView);
Alpine.data('scheduleView', scheduleView);
Alpine.data('webhookView', webhookView);
Alpine.data('agentFileView', fileEditorView('agents', 'agent-files'));
Alpine.data('skillFileView', fileEditorView('skills', 'skill-files'));
Alpine.data('workspaceView', workspaceView);
Alpine.data('kvstoreView', kvstoreView);

window.addEventListener('hashchange', () => {
  const hash = location.hash.slice(1);
  if (hash) Alpine.store('app').view = legacyViews[hash] || hash;
});

Alpine.start();

// Persist view and agent selection to localStorage
Alpine.effect(() => {
  const store = Alpine.store('app');
  localStorage.setItem('tsugite-view', store.view);
  if (location.hash.slice(1) !== store.view) location.hash = store.view;
  if (store.selectedAgent) localStorage.setItem('tsugite-agent', store.selectedAgent);
});

async function loadAgents() {
  try {
    const data = await get('/api/agents');
    const store = Alpine.store('app');
    store.agents = data.agents || [];
    if (store.agents.length && !store.selectedAgent) {
      store.selectedAgent = store.agents[0].name;
    }
  } catch { /* ignore */ }
}

loadAgents();

// SSE event stream — push updates to Alpine store
const _es = connectEvents((event) => {
  Alpine.store('app').lastEvent = { ...event, _ts: Date.now() };
});
_es.onopen = () => {
  // On reconnect, refresh agents to catch up
  loadAgents();
  Alpine.store('app').lastEvent = { type: 'reconnect', _ts: Date.now() };
};

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
  navigator.serviceWorker.register('/static/sw.js').catch(() => {});
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
