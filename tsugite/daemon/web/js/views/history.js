import { get } from '../api.js';
import { formatDate, renderMarkdown } from '../utils.js';

export default () => ({
  conversations: [],
  selectedConv: null,
  turns: [],
  loading: true,

  init() {
    const maybeLoad = () => {
      if (this.$store.app.view === 'history' && this.$store.app.selectedAgent) this.load();
    };
    this.$watch('$store.app.selectedAgent', maybeLoad);
    this.$watch('$store.app.view', maybeLoad);
  },

  async load() {
    this.loading = true;
    this.conversations = [];
    this.selectedConv = null;
    this.turns = [];

    const agent = this.$store.app.selectedAgent;
    if (!agent) { this.loading = false; return; }

    try {
      const data = await get(`/api/agents/${agent}/sessions`);
      this.conversations = (data.sessions || []).map(s => ({
        ...s,
        agent,
      }));
    } catch { /* ignore */ }
    this.loading = false;

    // Auto-select a session if navigated from another view
    const targetId = this.$store.app.viewSessionId;
    if (targetId) {
      this.$store.app.viewSessionId = null;
      const match = this.conversations.find(c => c.conversation_id === targetId);
      if (match) {
        await this.selectConversation(match);
      } else {
        // Session not in per-agent list — load directly by session_id
        await this.selectConversation({ agent, conversation_id: targetId });
      }
    }
  },

  async selectConversation(conv) {
    this.selectedConv = conv;
    this.turns = [];
    try {
      let url = `/api/agents/${conv.agent}/history?detail=true`;
      if (conv.conversation_id) {
        url += `&session_id=${encodeURIComponent(conv.conversation_id)}`;
      } else if (conv.user_id) {
        url += `&user_id=${encodeURIComponent(conv.user_id)}`;
      }
      const data = await get(url);
      this.turns = data.turns || [];
    } catch { /* ignore */ }
  },

  renderHtml(text) { return renderMarkdown(text); },

  formatDate,

  extractMessages(turn) {
    if (!turn.messages) return [];
    const items = [];
    for (const msg of turn.messages) {
      if (msg.role === 'assistant') {
        const codeMatch = msg.content?.match(/```(?:python)?\n([\s\S]*?)```/);
        if (codeMatch) {
          items.push({ type: 'tool_call', name: 'code', args: codeMatch[1] });
        }
        if (msg.tool_calls) {
          for (const tc of msg.tool_calls) {
            const fn = tc.function || {};
            items.push({ type: 'tool_call', name: fn.name || 'unknown', args: fn.arguments || '{}' });
          }
        }
      } else if (msg.role === 'user' && msg.content?.includes('<tsugite_execution_result>')) {
        const content = msg.content.replace(/<\/?tsugite_execution_result>/g, '').trim();
        const truncated = content.length > 500 ? content.slice(0, 500) + '...' : content;
        items.push({ type: 'tool_result', name: 'result', content: truncated });
      } else if (msg.role === 'tool') {
        const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
        items.push({ type: 'tool_result', name: msg.name || '', content: content.length > 500 ? content.slice(0, 500) + '...' : content });
      }
    }
    return items;
  },
});
