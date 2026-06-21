import { get } from '../api.js';
import { fmtTokens, fmtCost } from '../utils.js';

function daysAgo(n) {
  const d = new Date();
  d.setDate(d.getDate() - n);
  return d.toISOString().split('T')[0];
}

export default () => ({
  totals: null,
  dailySummary: [],
  topAgents: [],
  topModels: [],
  period: 'day',
  sinceDays: 30,
  loading: true,
  error: null,

  init() {
    this.$watch(() => this.$store.app.view, v => { if (v === 'usage' && !this.totals) this.load(); });
    if (this.$store.app.view === 'usage') this.load();
  },

  async load() {
    this.loading = true;
    this.error = null;
    const since = daysAgo(this.sinceDays);
    try {
      const [totals, summary, agents, models] = await Promise.all([
        get(`/api/usage/total?since=${since}`),
        get(`/api/usage/summary?period=${this.period}&since=${since}`),
        get(`/api/usage/agents?since=${since}&limit=10`),
        get(`/api/usage/models?since=${since}&limit=10`),
      ]);
      this.totals = totals;
      this.dailySummary = summary;
      this.topAgents = agents;
      this.topModels = models;
      const next = totals?.total_tokens ?? null;
      if (this.$store.app.tokensTotal !== next) this.$store.app.tokensTotal = next;
    } catch (e) {
      this.error = e.message || 'Failed to load usage data';
    }
    this.loading = false;
  },

  maxCost() {
    if (!this.dailySummary.length) return 1;
    return Math.max(...this.dailySummary.map(r => r.total_cost || 0), 0.001);
  },

  barPct(cost) {
    return Math.max(((cost || 0) / this.maxCost()) * 100, 2) + '%';
  },

  barLabel(row) {
    if (this.period === 'day') {
      const d = new Date(row.period + 'T00:00:00');
      return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    }
    return row.period;
  },

  fmtTokens,
  fmtCost,
});
