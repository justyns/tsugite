/**
 * File-attach + paste controller for the composer: the generic/camera file
 * pickers, client-side re-encode + multipart upload (POST .../upload) of picked,
 * pasted, or dropped files, and the large-text-paste chooser (attach as .txt vs
 * paste inline). A pasted screenshot routes to attach; a pasted reference marker
 * routes to the context controller via the `attachRef` dep.
 *
 * A mutated $state class instance, never a reassigned binding (AGENTS.md): the
 * component instantiates it, binds the hidden inputs + paste banner to its
 * `$state` element fields, and wires the paste-dismiss effect to it.
 */
import { tick } from 'svelte';
import type { Attachment } from '$lib/components/composer/types';
import { api } from '$lib/api/client';
import { loadImageConfig } from '$lib/api/serverConfig';
import { reencodeImage } from '$lib/media/imageEncode';
import { toasts } from '$lib/components/feedback/toast-store.svelte';
import { extractFiles } from './dropFiles';
import { parseRefMarker } from './attachRecord';
import { writeDraft } from './draft';

// A large text paste offers a choice instead of dumping a wall of text into the
// draft. Thresholds mirror the legacy UI: large past 500 chars OR 11 lines.
const PASTE_MAX_CHARS = 500;
const PASTE_MAX_LINES = 11;

export interface FileAttachDeps {
  /** The composer's bindable input text (read + written by the paste-inline path). */
  value: string;
  readonly sessionId: string | null;
  /** Route a pasted reference marker to the context controller. */
  attachRef: (kind: string, id: string) => void;
}

export class FileAttach {
  #deps: FileAttachDeps;
  #pasteTa: HTMLTextAreaElement | null = null;

  attachments = $state<Attachment[]>([]);
  fileInput = $state<HTMLInputElement>();
  cameraInput = $state<HTMLInputElement>();
  pastePrompt = $state<{ text: string; start: number; end: number } | null>(null);
  pasteBannerEl = $state<HTMLElement>();

  constructor(deps: FileAttachDeps) {
    this.#deps = deps;
  }

  openFilePicker = (): void => {
    this.fileInput?.click();
  };

  openCamera = (): void => {
    this.cameraInput?.click();
  };

  onFilesChosen = (e: Event): void => {
    const input = e.currentTarget as HTMLInputElement;
    const files = input.files ? Array.from(input.files) : [];
    input.value = '';
    void this.upload(files);
  };

  #isLargePaste(text: string): boolean {
    return text.length > PASTE_MAX_CHARS || text.split('\n').length > PASTE_MAX_LINES;
  }

  // A pasted screenshot (Firefox exposes it as an image File on clipboardData)
  // routes to attach; a large text paste opens the chooser; anything smaller is
  // left to the browser's native paste. Files always win over accompanying text.
  onPaste = (e: ClipboardEvent): void => {
    const files = extractFiles(e.clipboardData);
    if (files.length > 0) {
      e.preventDefault();
      void this.upload(files);
      return;
    }
    // A "copy reference" paste carries an html marker naming a record; attach that
    // record instead of pasting its text. A normal paste has no marker and falls
    // through to the unchanged image/large-text handling below.
    const ref = parseRefMarker(e.clipboardData?.getData('text/html') ?? '');
    if (ref) {
      e.preventDefault();
      this.#deps.attachRef(ref.kind, ref.id);
      return;
    }
    const text = e.clipboardData?.getData('text/plain') ?? '';
    if (!this.#isLargePaste(text)) return;
    const ta = e.target as HTMLTextAreaElement;
    e.preventDefault();
    this.#pasteTa = ta;
    this.pastePrompt = {
      text,
      start: ta.selectionStart ?? this.#deps.value.length,
      end: ta.selectionEnd ?? this.#deps.value.length,
    };
  };

  #pasteFilename(): string {
    const d = new Date();
    const pad = (n: number) => String(n).padStart(2, '0');
    const ts = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
    return `pasted-${ts}.txt`;
  }

  #resetPastePrompt(): void {
    this.pastePrompt = null;
    this.#pasteTa = null;
  }

  pasteAsFile = (): void => {
    const p = this.pastePrompt;
    if (!p) return;
    this.#resetPastePrompt();
    void this.upload([new File([p.text], this.#pasteFilename(), { type: 'text/plain' })]);
  };

  pasteInline = (): void => {
    const p = this.pastePrompt;
    if (!p) return;
    const ta = this.#pasteTa;
    this.#resetPastePrompt();
    const next = this.#deps.value.slice(0, p.start) + p.text + this.#deps.value.slice(p.end);
    this.#deps.value = next;
    writeDraft(this.#deps.sessionId, next);
    const caret = p.start + p.text.length;
    void tick().then(() => ta?.setSelectionRange(caret, caret));
  };

  // While the chooser is open, Escape or a click outside it defaults to inline -
  // never discard the pasted text. Returns a cleanup; call from a component
  // `$effect` so it re-subscribes as the prompt opens/closes.
  installPasteDismiss(): (() => void) | void {
    if (!this.pastePrompt) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        this.pasteInline();
      }
    };
    const onDown = (e: MouseEvent) => {
      if (this.pasteBannerEl && !this.pasteBannerEl.contains(e.target as Node)) this.pasteInline();
    };
    window.addEventListener('keydown', onKey, true);
    window.addEventListener('mousedown', onDown, true);
    return () => {
      window.removeEventListener('keydown', onKey, true);
      window.removeEventListener('mousedown', onDown, true);
    };
  }

  async upload(files: File[]): Promise<void> {
    if (!files.length) return;
    try {
      // Re-encode photos client-side (downscale + JPEG) before upload; non-images
      // and svg/gif pass through untouched. Config comes from /api/health.
      const cfg = await loadImageConfig();
      const processed = await Promise.all(files.map((f) => reencodeImage(f, cfg)));
      const res = await api.uploadFiles<{ files: { name: string; size?: number }[] }>(
        `/api/chat/upload`,
        processed,
      );
      this.attachments = [
        ...this.attachments,
        ...res.files.map((f) => ({
          id: f.name,
          name: f.name,
          ...(f.size ? { size: `${Math.round(f.size / 1024)} KB` } : {}),
        })),
      ];
    } catch (err) {
      toasts.push('err', 'Upload failed', {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }

  removeAttachment = (id: string): void => {
    this.attachments = this.attachments.filter((a) => a.id !== id);
  };
}
