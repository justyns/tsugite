import Alpine from 'https://cdn.jsdelivr.net/npm/alpinejs@3/dist/module.esm.js';
import { get } from './api.js';
import chatView from './views/chat.js';
import dashboardView from './views/dashboard.js';
import scheduleView from './views/schedules.js';
import historyView from './views/history.js';
import webhookView from './views/webhooks.js';

window.Alpine = Alpine;

Alpine.store('app', {
  agents: [],
  selectedAgent: null,
  view: location.hash.slice(1) || 'dashboard',
  theme: localStorage.getItem('tsugite_theme') || 'frappe',
  userId: localStorage.getItem('tsugite_user_id') || 'web-user-1',
  showSettings: false,
});

Alpine.data('chatView', chatView);
Alpine.data('dashboardView', dashboardView);
Alpine.data('scheduleView', scheduleView);
Alpine.data('historyView', historyView);
Alpine.data('webhookView', webhookView);

window.addEventListener('hashchange', () => {
  const hash = location.hash.slice(1);
  if (hash) Alpine.store('app').view = hash;
});

Alpine.start();

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
