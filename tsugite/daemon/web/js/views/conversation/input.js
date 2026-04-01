import { get, post } from '../../api.js';

export const inputMixin = {
  _mobileQuery: null,
  expandedInput: false,
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

  onInputChange() {
    this.showCommandSuggestions = this.messageText.startsWith('/') && !this.messageText.includes(' ') && this.filteredCommands.length > 0;
    this.commandSelectedIndex = 0;
  },

  selectCommand(cmd) {
    this.messageText = `/${cmd.name} `;
    this.showCommandSuggestions = false;
    this.$nextTick(() => {
      const input = document.getElementById('message-input');
      if (input) input.focus();
    });
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
    const visibleParams = cmd.params.filter(p => p.name !== 'user_id');
    const requiredVisible = visibleParams.filter(p => p.required);
    if (requiredVisible.length === 1 && argsText) {
      kwargs[requiredVisible[0].name] = argsText;
    } else if (requiredVisible.length > 1) {
      const parts = argsText.split(/\s+/);
      for (let i = 0; i < visibleParams.length && i < parts.length; i++) {
        kwargs[visibleParams[i].name] = parts[i];
      }
    } else if (visibleParams.length === 1 && !visibleParams[0].required && argsText) {
      kwargs[visibleParams[0].name] = argsText;
    }
    if (hasUserId) kwargs.user_id = this.userId;
    try {
      const data = await post(`/api/agents/${agent}/commands/${cmd.name}`, kwargs);
      return data.result || JSON.stringify(data);
    } catch (e) {
      return `Command failed: ${e.message}`;
    }
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

  autoResize(e) {
    const maxH = parseInt(getComputedStyle(e.target).maxHeight) || 150;
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, maxH) + 'px';
  },

  _resetInputHeight() {
    this.expandedInput = false;
    this.$nextTick(() => {
      const ta = document.getElementById('message-input');
      if (ta) ta.style.height = 'auto';
    });
  },
};
