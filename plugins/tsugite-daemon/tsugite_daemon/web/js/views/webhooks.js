import { get, post, del } from '../api.js';

export default () => ({
  webhooks: [],
  revealedTokens: new Set(),
  loading: true,
  showForm: false,
  form: { agent: '', source: '' },
  error: null,

  init() {
    this.load();
  },

  async load() {
    this.error = null;
    try {
      const data = await get('/api/webhooks');
      this.webhooks = data.webhooks || [];
    } catch { /* ignore */ }
    this.loading = false;
  },

  async create() {
    this.error = null;
    if (!this.form.agent || !this.form.source) {
      this.error = 'Agent and source are required';
      return;
    }
    try {
      await post('/api/webhooks', this.form);
      this.showForm = false;
      this.form = { agent: '', source: '' };
      await this.load();
    } catch (e) {
      this.error = e.message;
    }
  },

  async remove(w) {
    if (!confirm(`Delete webhook for "${w.source}"?`)) return;
    try {
      await del(`/api/webhooks/${w.token}`);
      await this.load();
    } catch (e) {
      this.error = e.message;
    }
  },

  toggleReveal(token) {
    if (this.revealedTokens.has(token)) {
      this.revealedTokens.delete(token);
    } else {
      this.revealedTokens.add(token);
    }
    this.revealedTokens = new Set(this.revealedTokens);
  },

  isRevealed(token) {
    return this.revealedTokens.has(token);
  },

  maskedToken(token) {
    if (token.length <= 6) return '***';
    return token.slice(0, 3) + '***' + token.slice(-3);
  },

  webhookUrl(token) {
    return `${window.location.origin}/webhook/${token}`;
  },
});
