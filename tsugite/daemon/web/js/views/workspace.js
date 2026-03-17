import { get, put, post } from '../api.js';
import { formatDate, formatFileSize } from '../utils.js';

export default () => ({
  files: [],
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
      if (this.$store.app.view === 'workspace') this.load();
    });
    this.$watch('$store.app.view', (view) => {
      if (view === 'workspace' && this.$store.app.selectedAgent) this.load();
    });
    this._keyHandler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 's' && this.$store.app.view === 'workspace' && this.selectedFile && this.selectedFile.is_text) {
        e.preventDefault();
        this.save();
      }
    };
    document.addEventListener('keydown', this._keyHandler);
  },

  destroy() {
    if (this._keyHandler) document.removeEventListener('keydown', this._keyHandler);
  },

  async load() {
    const agent = this.$store.app.selectedAgent;
    if (!agent) return;
    this.loading = true;
    this.error = null;
    this.selectedFile = null;
    this.content = '';
    this.originalContent = '';
    try {
      const data = await get(`/api/agents/${agent}/workspace`);
      this.files = data.files || [];
      this.workspaceDir = data.workspace_dir || '';
    } catch (e) {
      this.error = e.message;
    } finally {
      this.loading = false;
    }
  },

  get isDirty() {
    return this.content !== this.originalContent;
  },

  formatSize: formatFileSize,

  async selectFile(file) {
    if (this.isDirty && !confirm('You have unsaved changes. Discard?')) return;
    this.selectedFile = file;
    this.content = '';
    this.originalContent = '';
    this.error = null;

    if (!file.is_text) return;

    const agent = this.$store.app.selectedAgent;
    try {
      const data = await get(`/api/agents/${agent}/workspace/content?path=${encodeURIComponent(file.path)}`);
      if (!data.is_text) {
        this.selectedFile = { ...file, is_text: false };
        return;
      }
      this.content = data.content;
      this.originalContent = data.content;
    } catch (e) {
      this.error = e.message;
    }
  },

  async save() {
    if (!this.selectedFile || !this.selectedFile.is_text || this.saving) return;
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
