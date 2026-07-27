/**
 * Toast queue backing `<Toasts>` (the stack host).
 * Push from anywhere (API error handlers, job updates, ...); `<Toasts>` renders
 * whatever is queued. Exported as a class instance - never a reassigned $state binding.
 */

export type ToastVariant = 'ok' | 'warn' | 'err' | 'info';
export type ToastIcon = 'check' | 'alert' | 'x' | 'dot' | 'q';

export interface ToastOptions {
  body?: string;
  /** Toasts announce state changes; anything requiring action links straight
   * to the owning surface via this button. */
  actionLabel?: string;
  onAction?: () => void;
  /** Overrides the variant's default icon (e.g. a job-question toast uses `q`). */
  icon?: ToastIcon;
  /** Suppresses the 6s auto-dismiss. `err` is always sticky regardless of this flag. */
  sticky?: boolean;
}

export interface ToastEntry extends ToastOptions {
  id: number;
  variant: ToastVariant;
  title: string;
  sticky: boolean;
}

/** Oldest toast is dropped once the stack exceeds this many at once. */
export const MAX_TOASTS = 4;
export const AUTO_DISMISS_MS = 6000;
/** Auto-dismiss delay reinstated after the pointer/focus leaves a hovered toast. */
export const RESUME_DISMISS_MS = 2500;
/** Matches the `.t-toast.is-out` fade transition duration before unmount. */
export const EXIT_DURATION_MS = 320;

/** Errors persist until dismissed; every other variant honors the caller's flag. */
export function isSticky(variant: ToastVariant, sticky?: boolean): boolean {
  return variant === 'err' || Boolean(sticky);
}

class ToastStore {
  items = $state<ToastEntry[]>([]);
  private nextId = 1;

  push(variant: ToastVariant, title: string, options: ToastOptions = {}): number {
    const id = this.nextId++;
    this.items.push({ id, variant, title, ...options, sticky: isSticky(variant, options.sticky) });
    if (this.items.length > MAX_TOASTS) this.items.shift();
    return id;
  }

  dismiss(id: number): void {
    const idx = this.items.findIndex((t) => t.id === id);
    if (idx !== -1) this.items.splice(idx, 1);
  }
}

export const toasts = new ToastStore();
