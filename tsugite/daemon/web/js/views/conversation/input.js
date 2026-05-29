import { get, post } from '../../api.js';

export const inputMixin = {
  _mobileQuery: null,
  _draftTimer: null,
  showExpandModal: false,
  expandModalText: '',
  messageText: '',
  availableCommands: [],
  showCommandSuggestions: false,
  commandSelectedIndex: 0,

  async _loadCommands() {
    try {
      const data = await get('/api/commands');
      this.availableCommands = data.commands || [];
    } catch {
      this.availableCommands = [];
    }
  },

  _draftKey() {
    return `tsugite_draft_${this.selectedSessionId || 'new'}`;
  },

  _saveDraft() {
    clearTimeout(this._draftTimer);
    this._draftTimer = setTimeout(() => this._saveDraftNow(), 300);
  },

  _saveDraftNow() {
    clearTimeout(this._draftTimer);
    const key = this._draftKey();
    if (this.messageText) {
      localStorage.setItem(key, this.messageText);
    } else {
      localStorage.removeItem(key);
    }
  },

  _restoreDraft() {
    const saved = localStorage.getItem(this._draftKey());
    this.messageText = saved || '';
  },

  _clearDraft() {
    clearTimeout(this._draftTimer);
    localStorage.removeItem(this._draftKey());
  },

  onInputChange() {
    this.showCommandSuggestions = this.messageText.startsWith('/') && !this.messageText.includes(' ') && this.filteredCommands.length > 0;
    this.commandSelectedIndex = 0;
    this._saveDraft();
  },

  selectCommand(cmd) {
    this.messageText = `/${cmd.name} `;
    this.showCommandSuggestions = false;
    this.$nextTick(() => this.$refs.messageInput?.focus());
  },

  _parseCommand(text) {
    const match = text.match(/^\/(\S+)\s*(.*)/s);
    if (!match) return null;
    const name = match[1];
    const rest = match[2].trim();
    const cmd = this.availableCommands.find(c => c.name === name);
    if (!cmd) return null;
    return { command: cmd, args: rest };
  },

  async _runCommand(cmd, argsText) {
    const agent = this.$store.app.selectedAgent;
    const kwargs = {};
    const hasUserId = cmd.params.some(p => p.name === 'user_id');
    const hasSessionId = cmd.params.some(p => p.name === 'session_id');
    const hiddenNames = new Set(['user_id', 'session_id']);
    const visibleParams = cmd.params.filter(p => !hiddenNames.has(p.name));
    const requiredVisible = visibleParams.filter(p => p.required);

    // Pull `--key value` flags off the tail of argsText so commands like
    // `/job do thing --ac "tests pass" --repo /path` route those values to
    // the right params instead of stuffing the whole string into prompt.
    const validFlagNames = new Set(visibleParams.map(p => p.name));
    const { positional, flags } = this._extractFlags(argsText, validFlagNames);
    Object.assign(kwargs, flags);

    if (requiredVisible.length === 1 && positional) {
      kwargs[requiredVisible[0].name] = positional;
    } else if (requiredVisible.length > 1) {
      const parts = positional.split(/\s+/);
      for (let i = 0; i < visibleParams.length && i < parts.length; i++) {
        if (!(visibleParams[i].name in kwargs)) {
          kwargs[visibleParams[i].name] = parts[i];
        }
      }
    } else if (visibleParams.length === 1 && !visibleParams[0].required && positional) {
      kwargs[visibleParams[0].name] = positional;
    }
    if (hasUserId) kwargs.user_id = this.userId;
    // Forward the active chat's session id so commands anchor on the user's
    // current conversation, not their default/primary session.
    if (hasSessionId && this.selectedSessionId) kwargs.session_id = this.selectedSessionId;
    try {
      const data = await post(`/api/agents/${agent}/commands/${cmd.name}`, kwargs);
      return data.result || JSON.stringify(data);
    } catch (e) {
      return `Command failed: ${e.message}`;
    }
  },

  // Split argsText into a positional remainder + a flags dict. Recognises
  // `--key value` and `--key "quoted value"` only for keys in validFlagNames;
  // unrecognised --flags stay in the positional string so prompts can contain
  // `--foo` text without being eaten. Repeatable flags are joined with `|`.
  _extractFlags(argsText, validFlagNames) {
    if (!argsText) return { positional: '', flags: {} };
    const tokens = argsText.match(/"[^"]*"|\S+/g) || [];
    const positionalParts = [];
    const flags = {};
    for (let i = 0; i < tokens.length; i++) {
      const tok = tokens[i];
      const flagMatch = /^--([a-zA-Z_][a-zA-Z0-9_-]*)$/.exec(tok);
      const name = flagMatch && flagMatch[1].replace(/-/g, '_');
      if (name && validFlagNames.has(name) && i + 1 < tokens.length) {
        let value = tokens[i + 1];
        if (value.startsWith('"') && value.endsWith('"')) value = value.slice(1, -1);
        flags[name] = name in flags ? `${flags[name]}|${value}` : value;
        i += 1;
      } else {
        positionalParts.push(tok.startsWith('"') && tok.endsWith('"') ? tok.slice(1, -1) : tok);
      }
    }
    return { positional: positionalParts.join(' '), flags };
  },

  onInputKeydown(e) {
    if (this.showCommandSuggestions) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        this.commandSelectedIndex = Math.min(this.commandSelectedIndex + 1, this.filteredCommands.length - 1);
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        this.commandSelectedIndex = Math.max(this.commandSelectedIndex - 1, 0);
        return;
      }
      if (e.key === 'Tab' || (e.key === 'Enter' && !e.shiftKey)) {
        e.preventDefault();
        const selected = this.filteredCommands[this.commandSelectedIndex];
        if (selected) this.selectCommand(selected);
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        this.showCommandSuggestions = false;
        return;
      }
    }

    if (e.key === 'Escape' && this.sending) {
      e.preventDefault();
      this.cancelChat();
      return;
    }
    if (e.key === 'Enter' && !e.shiftKey && !this._mobileQuery.matches) {
      e.preventDefault();
      this.sendMessage();
    }
  },

  _resizeTextarea(ta) {
    const maxH = parseInt(getComputedStyle(ta).maxHeight) || 150;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, maxH) + 'px';
  },

  autoResize(e) {
    this._resizeTextarea(e.target);
  },

  _resetInputHeight() {
    this.$nextTick(() => {
      const ta = this.$refs.messageInput;
      if (ta) ta.style.height = 'auto';
    });
  },

  openExpandModal() {
    this.expandModalText = this.messageText;
    this.showExpandModal = true;
    this.$nextTick(() => this.$refs.expandModalText?.focus());
  },

  closeExpandModal() {
    this.showExpandModal = false;
    this.expandModalText = '';
  },

  confirmExpandModal() {
    this.messageText = this.expandModalText;
    this.expandModalText = '';
    this.showExpandModal = false;
    this._saveDraft();
    this.$nextTick(() => {
      const ta = this.$refs.messageInput;
      if (ta) {
        this._resizeTextarea(ta);
        ta.focus();
      }
    });
  },
};
