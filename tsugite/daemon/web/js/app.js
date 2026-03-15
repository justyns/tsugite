import Alpine from 'https://cdn.jsdelivr.net/npm/alpinejs@3/dist/module.esm.js';
import { get, post, patch, connectEvents } from './api.js';
import chatView from './views/chat.js';
import dashboardView from './views/dashboard.js';
import scheduleView from './views/schedules.js';
import sessionView from './views/sessions.js';
import historyView from './views/history.js';
import webhookView from './views/webhooks.js';
import agentFileView from './views/agent-files.js';
import skillFileView from './views/skills.js';

window.Alpine = Alpine;
window.tsugiteApi = { get, post, patch };

Alpine.store('app', {
  agents: [],
  selectedAgent: localStorage.getItem('tsugite-agent') || null,
  view: location.hash.slice(1) || localStorage.getItem('tsugite-view') || 'dashboard',
  theme: localStorage.getItem('tsugite_theme') || 'frappe',
  userId: localStorage.getItem('tsugite_user_id') || 'web-user-1',
  showSettings: false,
  lastEvent: null,
  viewSessionId: null,
});

Alpine.data('chatView', chatView);
Alpine.data('dashboardView', dashboardView);
Alpine.data('scheduleView', scheduleView);
Alpine.data('sessionView', sessionView);
Alpine.data('historyView', historyView);
Alpine.data('webhookView', webhookView);
Alpine.data('agentFileView', agentFileView);
Alpine.data('skillFileView', skillFileView);

window.addEventListener('hashchange', () => {
  const hash = location.hash.slice(1);
  if (hash) Alpine.store('app').view = hash;
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
