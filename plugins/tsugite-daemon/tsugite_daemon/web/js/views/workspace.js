import { get, put, post } from '../api.js';
import { formatDate, formatFileSize } from '../utils.js';

export default () => ({
  sidebarOpen: false,
  entries: [],
  currentDir: '',
  loading: true,
  error: null,
  selectedFile: null,
  content: '',
  originalContent: '',
  saving: false,
  workspaceDir: '',

  _keyHandler: null,

  init() {
    this.$watch('$store.app.selectedAgent', () => {
      if (this.$store.app.view === 'workspace') { this.currentDir = ''; this.load(); }
    });
    this.$watch('$store.app.view', (view) => {
      if (view === 'workspace' && this.$store.app.selectedAgent) this.load();
    });
    this._keyHandler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 's' && this.$store.app.view === 'workspace' && this.selectedFile) {
        e.preventDefault();
        this.save();
      }
    };
    document.addEventListener('keydown', this._keyHandler);
    if (this.$store.app.view === 'workspace' && this.$store.app.selectedAgent) this.load();
  },

  destroy() {
    if (this._keyHandler) document.removeEventListener('keydown', this._keyHandler);
  },

  async load() {
    const agent = this.$store.app.selectedAgent;
    if (!agent) return;
    this.loading = true;
    this.error = null;
    try {
      let url = `/api/agents/${agent}/workspace`;
      if (this.currentDir) url += `?subdir=${encodeURIComponent(this.currentDir)}`;
      const data = await get(url);
      this.entries = data.entries || [];
      this.workspaceDir = data.workspace_dir || '';
    } catch (e) {
      this.error = e.message;
    } finally {
      this.loading = false;
    }
  },

  get breadcrumbs() {
    if (!this.currentDir) return [];
    const parts = this.currentDir.split('/').filter(Boolean);
    return parts.map((name, i) => ({
      name,
      path: parts.slice(0, i + 1).join('/'),
    }));
  },

  _navigateTo(dir) {
    if (this.isDirty && !confirm('You have unsaved changes. Discard?')) return;
    this.selectedFile = null;
    this.content = '';
    this.originalContent = '';
    this.currentDir = dir;
    this.load();
  },

  openDir(entry) { this._navigateTo(entry.path); },
  goToRoot() { this._navigateTo(''); },
  goToBreadcrumb(crumb) { this._navigateTo(crumb.path); },

  get isDirty() {
    return this.content !== this.originalContent;
  },

  formatSize: formatFileSize,

  async selectFile(file) {
    if (this.isDirty && !confirm('You have unsaved changes. Discard?')) return;
    this.sidebarOpen = false;
    this.selectedFile = file;
    this.content = '';
    this.originalContent = '';
    this.error = null;

    const agent = this.$store.app.selectedAgent;
    try {
      const data = await get(`/api/agents/${agent}/workspace/content?path=${encodeURIComponent(file.path)}`);
      this.content = data.content;
      this.originalContent = data.content;
    } catch (e) {
      this.error = e.message;
    }
  },

  async save() {
    if (!this.selectedFile || this.saving) return;
    this.saving = true;
    this.error = null;
    try {
      const agent = this.$store.app.selectedAgent;
      await put(`/api/agents/${agent}/workspace/content`, { path: this.selectedFile.path, content: this.content });
      this.originalContent = this.content;
    } catch (e) {
      this.error = e.message;
    } finally {
      this.saving = false;
    }
  },

  async attachToChat(file) {
    const agent = this.$store.app.selectedAgent;
    if (!agent) return;
    try {
      const data = await post(`/api/agents/${agent}/workspace/attach?path=${encodeURIComponent(file.path)}`);
      const uploaded = (data.files || [])[0];
      if (uploaded) {
        const store = this.$store.app;
        store.pendingWorkspaceFiles = [...store.pendingWorkspaceFiles, uploaded];
        store.view = 'conversations';
        location.hash = 'conversations';
      }
    } catch (e) {
      this.error = e.message;
    }
  },

  formatDate,
});
