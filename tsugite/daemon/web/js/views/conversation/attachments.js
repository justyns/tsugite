export const attachmentsMixin = {
  pendingFiles: [],
  isDragging: false,
  showPasteBanner: false,
  pendingPasteText: '',
  showPasteModal: false,
  pasteModalText: '',
  pasteModalFilename: '',

  addFiles(fileList) {
    for (const file of fileList) {
      const entry = { file, name: file.name, size: file.size, type: file.type, previewUrl: null };
      if (file.type.startsWith('image/')) entry.previewUrl = URL.createObjectURL(file);
      this.pendingFiles.push(entry);
    }
  },

  removeFile(index) {
    const entry = this.pendingFiles[index];
    if (entry?.previewUrl) URL.revokeObjectURL(entry.previewUrl);
    this.pendingFiles.splice(index, 1);
  },

  openFilePicker() {
    this.$refs.fileInput?.click();
  },

  onFileInputChange(e) {
    if (e.target.files?.length) this.addFiles(e.target.files);
    e.target.value = '';
  },

  onDragEnter(e) { this.isDragging = true; },
  onDragLeave(e) {
    if (!e.currentTarget.contains(e.relatedTarget)) this.isDragging = false;
  },
  onDrop(e) {
    this.isDragging = false;
    if (e.dataTransfer?.files?.length) this.addFiles(e.dataTransfer.files);
  },

  _pasteTimestamp() {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
  },

  pasteAsAttachment(text, filename) {
    if (!filename) filename = `pasted-${this._pasteTimestamp()}.txt`;
    this.addFiles([new File([text], filename, { type: 'text/plain' })]);
  },

  _resetPasteState() {
    this.showPasteBanner = false;
    this.showPasteModal = false;
    this.pendingPasteText = '';
    this.pasteModalText = '';
    this.pasteModalFilename = '';
  },

  onPaste(e) {
    const text = e.clipboardData?.getData('text/plain');
    if (!text || (text.length <= 500 && text.split('\n').length <= 11)) return;
    e.preventDefault();
    this.pendingPasteText = text;
    this.showPasteBanner = true;
  },

  acceptPasteAsFile() {
    this.pasteAsAttachment(this.pendingPasteText);
    this._resetPasteState();
  },

  dismissPasteBanner() {
    this.messageText += this.pendingPasteText;
    this._resetPasteState();
  },

  openPasteModal() {
    this._resetPasteState();
    this.pasteModalFilename = `pasted-${this._pasteTimestamp()}.txt`;
    this.showPasteModal = true;
    this.$nextTick(() => this.$refs.pasteModalText?.focus());
  },

  confirmPasteModal() {
    if (!this.pasteModalText.trim()) return;
    const filename = this.pasteModalFilename.trim() || `pasted-${this._pasteTimestamp()}.txt`;
    this.pasteAsAttachment(this.pasteModalText, filename);
    this._resetPasteState();
  },

  closePasteModal() {
    this._resetPasteState();
  },
};
