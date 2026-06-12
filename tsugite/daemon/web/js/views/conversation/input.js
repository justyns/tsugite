import { get, post } from '../../api.js';

export const inputMixin = {
  _mobileQuery: null,
  _draftTimer: null,
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
    // /run is wired directly to the terminals API rather than the adapter
    // command registry - it spawns a PTY-backed sub-session in the sidebar,
    // not an agent reply. We synthesise a virtual command so the existing
    // sendMessage path dispatches it without changes elsewhere.
    if (name === 'run') {
      return { command: { name: 'run', _terminal: true, params: [] }, args: rest };
    }
    const cmd = this.availableCommands.find(c => c.name === name);
    if (!cmd) return null;
    return { command: cmd, args: rest };
  },

  async _runCommand(cmd, argsText) {
    // /run synthesised by _parseCommand bypasses the agent command registry -
    // POST straight to /api/terminals and auto-select the new PTY in the
    // sidebar. Failures fall back to a string the chat thread shows verbatim.
    if (cmd && cmd._terminal) {
      const command = (argsText || '').trim();
      if (!command) return 'Usage: /run <command>';
      const newId = await this.$store.terminals?.runTerminal(command);
      if (newId) {
        return `Terminal session started - see “${command}” in the sidebar.`;
      }
      return 'Failed to start terminal - see toast.';
    }
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
    // The single required positional (e.g. prompt) is never flag-addressable -
    // free text ending in `--prompt foo` must stay part of the prompt.
    const validFlagNames = new Set(visibleParams.map(p => p.name));
    if (requiredVisible.length === 1) validFlagNames.delete(requiredVisible[0].name);
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

  // Split argsText into a verbatim positional head + a flags dict parsed from
  // the TAIL only. Consuming trailing `--key value` / `--key "quoted value"`
  // pairs (key = exact param name or unambiguous prefix, so `--ac` routes to
  // acceptance_criteria) and stopping at the first non-flag means the body of
  // the prompt is never re-tokenised: its quotes, newlines, spacing, and any
  // mid-text `--words` survive byte-for-byte. Repeated flags join with `|`
  // (which `cmd_job` already splits on for AC lists).
  _extractFlags(argsText, validFlagNames) {
    if (!argsText) return { positional: '', flags: {} };
    const names = Array.from(validFlagNames);
    const resolveName = (raw) => {
      const norm = raw.replace(/-/g, '_');
      if (validFlagNames.has(norm)) return norm;
      const candidates = names.filter(n => n.startsWith(norm));
      return candidates.length === 1 ? candidates[0] : null;
    };
    const tailFlag = /(?:^|\s)--([a-zA-Z_][a-zA-Z0-9_-]*)\s+("([^"]*)"|\S+)\s*$/;
    let rest = argsText;
    const flags = {};
    while (true) {
      const m = tailFlag.exec(rest);
      if (!m) break;
      const name = resolveName(m[1]);
      if (!name) break;
      const value = m[3] !== undefined ? m[3] : m[2];
      // Scanning tail-first reverses order, so prepend on repeats.
      flags[name] = name in flags ? `${value}|${flags[name]}` : value;
      rest = rest.slice(0, m.index);
    }
    return { positional: rest.trim(), flags };
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
    this.$store.tsu.open('expand');
    this.$nextTick(() => this.$refs.expandModalText?.focus());
  },

  closeExpandModal() {
    this.$store.tsu.close('expand');
    this.expandModalText = '';
  },

  confirmExpandModal() {
    this.messageText = this.expandModalText;
    this.expandModalText = '';
    this.$store.tsu.close('expand');
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
