import { get } from '../api.js';

export default () => ({
  namespaces: [],
  selectedNamespace: null,
  keys: [],
  selectedKey: null,
  keyData: null,
  prefix: '',
  loading: false,
  error: null,
  sidebarOpen: false,

  init() {
    this.$watch('$store.app.view', (view) => {
      if (view === 'kvstore') this.loadNamespaces();
    });
  },

  async loadNamespaces() {
    this.loading = true;
    this.error = null;
    try {
      const data = await get('/api/kv/namespaces');
      this.namespaces = data.namespaces || [];
    } catch (e) {
      this.error = e.message;
    } finally {
      this.loading = false;
    }
  },

  async selectNamespace(ns) {
    this.selectedNamespace = ns;
    this.selectedKey = null;
    this.keyData = null;
    this.prefix = '';
    this.sidebarOpen = false;
    await this.loadKeys();
  },

  async loadKeys() {
    if (!this.selectedNamespace) return;
    this.error = null;
    try {
      const params = this.prefix ? `?prefix=${encodeURIComponent(this.prefix)}` : '';
      const data = await get(`/api/kv/${encodeURIComponent(this.selectedNamespace)}/keys${params}`);
      this.keys = data.keys || [];
    } catch (e) {
      this.error = e.message;
    }
  },

  async selectKey(key) {
    this.selectedKey = key;
    this.error = null;
    try {
      const data = await get(`/api/kv/${encodeURIComponent(this.selectedNamespace)}/keys/${encodeURIComponent(key)}`);
      this.keyData = data;
    } catch (e) {
      this.error = e.message;
      this.keyData = null;
    }
  },

  formatValue(val) {
    if (!val) return '';
    try {
      return JSON.stringify(JSON.parse(val), null, 2);
    } catch {
      return val;
    }
  },

  formatExpiry(ts) {
    if (!ts) return 'Never';
    return new Date(ts * 1000).toLocaleString();
  },
});
