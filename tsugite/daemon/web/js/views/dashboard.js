import { get } from '../api.js';
import { formatDate } from '../utils.js';

export default () => ({
  agentCards: [],
  upcomingSchedules: [],
  loading: true,
  error: null,
  _interval: null,

  init() {
    this.load();
    this._interval = setInterval(() => this.load(), 30000);
  },

  destroy() {
    if (this._interval) clearInterval(this._interval);
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
    } catch (e) {
      this.error = e.message;
    } finally {
      this.loading = false;
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

  formatDate,
});
