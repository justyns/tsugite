import { get } from '../api.js';
import { formatDate } from '../utils.js';

export default () => ({
  agentCards: [],
  upcomingSchedules: [],
  usageSummary: null,
  usageByAgent: [],
  loading: true,
  error: null,
  _debounceTimer: null,

  init() {
    this.load();
    this.$watch('$store.app.lastEvent', (ev) => {
      if (!ev) return;
      if (ev.type === 'agent_status' || ev.type === 'schedule_update') {
        this._debouncedLoad();
      }
    });
  },

  destroy() {
    if (this._debounceTimer) clearTimeout(this._debounceTimer);
  },

  _debouncedLoad() {
    if (this._debounceTimer) clearTimeout(this._debounceTimer);
    this._debounceTimer = setTimeout(() => this.load(), 200);
  },

  async load() {
    this.error = null;
    try {
      const [agentsData, schedulesData] = await Promise.all([
        get('/api/agents'),
        get('/api/schedules').catch(() => ({ schedules: [] })),
      ]);

      const cards = [];
      for (const agent of agentsData.agents || []) {
        try {
          const userId = encodeURIComponent(this.$store.app.userId);
          const status = await get(`/api/agents/${agent.name}/status?user_id=${userId}`);
          cards.push({ ...agent, ...status });
        } catch {
          cards.push({ ...agent, model: 'unknown', tokens: 0, context_limit: 0, message_count: 0 });
        }
      }
      this.agentCards = cards;

      // Sort schedules by next_run, take first 5
      const schedules = (schedulesData.schedules || [])
        .filter(s => s.enabled && s.next_run)
        .sort((a, b) => new Date(a.next_run) - new Date(b.next_run))
        .slice(0, 5);
      this.upcomingSchedules = schedules;

      // Load usage data
      await this._loadUsage(agentsData.agents || []);
    } catch (e) {
      this.error = e.message;
    } finally {
      this.loading = false;
    }
  },

  async _loadUsage(agents) {
    try {
      // Get 30-day summary
      const since = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString();
      this.usageSummary = await get(`/api/usage/summary?since=${since}`);

      // Get per-agent usage for the top agents
      const agentUsages = [];
      for (const agent of agents.slice(0, 10)) {
        try {
          const data = await get(`/api/usage/summary?agent=${agent.name}&since=${since}`);
          if (data.run_count > 0) {
            agentUsages.push({ name: agent.name, ...data });
          }
        } catch { /* skip */ }
      }
      agentUsages.sort((a, b) => b.total_cost - a.total_cost);
      this.usageByAgent = agentUsages;
    } catch {
      this.usageSummary = null;
      this.usageByAgent = [];
    }
  },

  contextPct(card) {
    if (!card.context_limit) return 0;
    return ((card.tokens / card.context_limit) * 100).toFixed(0);
  },

  contextDisplay(card) {
    if (!card.context_limit) return 'N/A';
    const tk = (card.tokens / 1000).toFixed(1);
    const lk = (card.context_limit / 1000).toFixed(0);
    return `${tk}k / ${lk}k (${this.contextPct(card)}%)`;
  },

  formatCost(cost) {
    if (!cost || cost === 0) return '$0.00';
    if (cost < 0.01) return '$' + cost.toFixed(4);
    return '$' + cost.toFixed(2);
  },

  formatTokens(tokens) {
    if (!tokens) return '0';
    if (tokens >= 1000000) return (tokens / 1000000).toFixed(1) + 'M';
    if (tokens >= 1000) return (tokens / 1000).toFixed(1) + 'k';
    return tokens.toString();
  },

  formatDate,
});
