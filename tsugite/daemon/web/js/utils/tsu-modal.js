/* ============================================================================
   tsu-modal - Alpine.js controller for the reusable modal shell.
   ----------------------------------------------------------------------------
   Registered by app.js as Alpine.store('tsu') + Alpine.data('tsuModal').

   Open a modal from anywhere:
     $store.tsu.open('settings')        // declarative, inside Alpine markup
     window.tsu.open('settings')        // imperative, from plain JS

   Close the top one:
     $store.tsu.close()  /  Esc  /  backdrop click (when dismissable)

   Each modal root uses  x-data="tsuModal('<name>', { ...opts })".
   Opts:  dismissable (bool, default true)
   ============================================================================ */

export const tsuStore = {
  stack: [],
  get top() { return this.stack[this.stack.length - 1] || null; },
  get count() { return this.stack.length; },
  isOpen(name) { return this.stack.includes(name); },
  isTop(name) { return this.top === name; },
  open(name) {
    if (!name) return this;
    if (!this.isOpen(name)) this.stack.push(name);
    return this;
  },
  close(name) {
    const target = name || this.top;
    if (!target) return this;
    this.stack = this.stack.filter((id) => id !== target);
    return this;
  },
  toggle(name) { return this.isOpen(name) ? this.close(name) : this.open(name); },
  closeAll() { this.stack = []; return this; },
};

export const tsuModal = (name, opts = {}) => ({
  name,
  dismissable: opts.dismissable !== false,

  get open() { return this.$store.tsu.isOpen(this.name); },
  get isTop() { return this.$store.tsu.isTop(this.name); },

  close() { this.$store.tsu.close(this.name); },

  onEscape() {
    if (!this.open || !this.isTop || !this.dismissable) return;
    this.close();
  },
  onBackdrop() {
    if (!this.dismissable) return;
    this.close();
  },
});

// Imperative handle for non-Alpine call sites. The actual store mutation is
// dispatched as a CustomEvent so this file doesn't need a live Alpine handle
// at import time - app.js registers the listener after Alpine.start().
export function installWindowHandle() {
  if (typeof window === 'undefined' || window.tsu) return;
  window.tsu = {
    open(name) { document.dispatchEvent(new CustomEvent('tsu:open', { detail: name })); },
    close(name) { document.dispatchEvent(new CustomEvent('tsu:close', { detail: name })); },
  };
}
