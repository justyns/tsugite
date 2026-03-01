import { get, put } from '../api.js';

export default function fileEditorView(viewName, apiPrefix) {
  return () => ({
    files: [],
    loading: true,
    error: null,
    selectedFile: null,
    content: '',
    originalContent: '',
    saving: false,
    filter: 'all',

    init() {
      this.$watch('$store.app.view', () => {
        if (this.$store.app.view === viewName) this.load();
      });
      document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 's' && this.$store.app.view === viewName && this.selectedFile && !this.selectedFile.readonly) {
          e.preventDefault();
          this.save();
        }
      });
    },

    async load() {
      this.loading = true;
      this.error = null;
      try {
        const data = await get(`/api/${apiPrefix}`);
        this.files = data.files || [];
      } catch (e) {
        this.error = e.message;
      } finally {
        this.loading = false;
      }
    },

    get filteredFiles() {
      if (this.filter === 'all') return this.files;
      return this.files.filter(f => f.source === this.filter);
    },

    get isDirty() {
      return this.content !== this.originalContent;
    },

    async selectFile(file) {
      if (this.isDirty && !confirm('You have unsaved changes. Discard?')) return;
      this.selectedFile = file;
      this.content = '';
      this.originalContent = '';
      try {
        const data = await get(`/api/${apiPrefix}/content?path=${encodeURIComponent(file.path)}`);
        this.content = data.content;
        this.originalContent = data.content;
      } catch (e) {
        this.error = e.message;
      }
    },

    async save() {
      if (!this.selectedFile || this.selectedFile.readonly || this.saving) return;
      this.saving = true;
      this.error = null;
      try {
        await put(`/api/${apiPrefix}/content`, { path: this.selectedFile.path, content: this.content });
        this.originalContent = this.content;
      } catch (e) {
        this.error = e.message;
      }
      this.saving = false;
    },
  });
}
