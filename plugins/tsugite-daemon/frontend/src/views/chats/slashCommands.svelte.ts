/**
 * Slash-command controller for the composer: the `/token` menu (GET
 * /api/commands), its second-stage argument choices (static `choices`, or a
 * model-fetched `effort` list), inline-menu keyboard navigation, and dispatch to
 * the command endpoint. Reads the composer's bindable `value` and the open
 * session/agent through its deps so its `$derived` menus track the live input.
 *
 * A mutated $state class instance, never a reassigned binding (AGENTS.md): the
 * component instantiates it, wires the effects that load its lists, and reads its
 * derived menus in the markup.
 */
import { api } from '$lib/api/client';
import { auth } from '$lib/stores/auth.svelte';
import { clearDraft, writeDraft } from './draft';
import { modelPickerRequest } from './modelPickerSignal.svelte';
import { commandArgParam } from '$lib/shell/palette-sources';

export interface CommandParam {
  name: string;
  type: string;
  required: boolean;
  /** Rich-input hint: a dedicated control (`model`, `effort`) for this arg. */
  widget?: string;
  /** A fixed set of valid values, offered as an inline choices list. */
  choices?: string[];
}
export interface Command {
  name: string;
  description: string;
  params: CommandParam[];
}

export interface SlashDeps {
  /** The composer's bindable input text (read + written). */
  value: string;
  readonly agent: string;
  readonly sessionId: string | null;
  /** Deliver an argument-picked command through the same send path as a typed one. */
  handleSend: (text: string) => void;
  /** A slash-command finished: surface its result as an inline conversation echo. */
  onCommandResult?: (
    command: string,
    output: string,
    ok: boolean,
    action?: { label: string; href: string },
  ) => void;
}

export class SlashCommands {
  #deps: SlashDeps;
  #effortFetchKey = '';

  commands = $state<Command[]>([]);
  slashActive = $state(0);
  argActive = $state(0);
  // Slash menu: dismissed by Escape until the next keystroke re-opens it.
  slashDismissed = $state(false);
  // Effort levels are model-dependent, so they're fetched per session when the
  // current arg wants them; null leaves the arg a plain text field.
  effortLevels = $state<string[] | null>(null);

  // Slash menu: open while the value is a lone `/token` (before any argument).
  readonly slashQuery: string | null = $derived.by(() => {
    const m = /^\s*\/([^\s]*)$/.exec(this.#deps.value);
    return m ? (m[1] ?? '').toLowerCase() : null;
  });
  readonly slashMatches: Command[] = $derived.by(() => {
    const q = this.slashQuery;
    return q === null ? [] : this.commands.filter((c) => c.name.toLowerCase().startsWith(q));
  });
  readonly slashOpen: boolean = $derived(!this.slashDismissed && this.slashMatches.length > 0);

  // Second stage: once the command is typed and a space begins its argument
  // (`/<cmd> <partial>`), resolve which command and which arg param is in play so
  // its `choices`/`widget` hint can drive an inline options list (kept to one line
  // - a slash command isn't multi-line).
  readonly argContext = $derived.by(() => {
    const m = /^\s*\/(\S+)[ \t]+([^\n]*)$/.exec(this.#deps.value);
    if (!m) return null;
    const [, name = '', partial = ''] = m;
    const cmd = this.commands.find((c) => c.name.toLowerCase() === name.toLowerCase());
    const arg = cmd && commandArgParam(cmd);
    return cmd && arg ? { cmd, arg, partial } : null;
  });

  // The choices to offer for the current argument, filtered by what's typed. A
  // `widget:"model"` arg has no inline list (picking the command opens the header
  // picker instead), so it's excluded here.
  readonly argChoices: string[] | null = $derived.by(() => {
    const ctx = this.argContext;
    if (!ctx || ctx.arg.widget === 'model') return null;
    const all = ctx.arg.choices ?? (ctx.arg.widget === 'effort' ? this.effortLevels : null);
    if (!all || all.length === 0) return null;
    const q = ctx.partial.trim().toLowerCase();
    return q ? all.filter((c) => c.toLowerCase().includes(q)) : all;
  });
  readonly argOpen: boolean = $derived(
    !this.slashDismissed && !this.slashOpen && !!this.argChoices && this.argChoices.length > 0,
  );

  constructor(deps: SlashDeps) {
    this.#deps = deps;
  }

  /** Load the command list (best-effort - the menu just stays empty on 404). */
  loadCommands(): void {
    api
      .get<{ commands: Command[] }>('/api/commands')
      .then((res) => (this.commands = res.commands))
      .catch(() => (this.commands = []));
  }

  /** Fetch the model's effort levels when the current arg wants them (a
   *  `widget:"effort"` param). Deduped per agent#session; a null result (model has
   *  no effort levels) leaves the arg a plain text field. Called from a component
   *  `$effect` so it re-runs as the argument in play changes. */
  syncEffortLevels(): void {
    const sessionId = this.#deps.sessionId;
    if (this.argContext?.arg.widget !== 'effort' || !sessionId) return;
    const agent = this.#deps.agent;
    const key = `${agent}#${sessionId}`;
    if (key === this.#effortFetchKey) return;
    this.#effortFetchKey = key;
    this.effortLevels = null;
    api
      .get<{ supported_effort_levels: string[] | null }>(
        `/api/agents/${encodeURIComponent(agent)}/effort-levels?session_id=${encodeURIComponent(sessionId)}`,
      )
      .then((r) => {
        if (this.#effortFetchKey === key) this.effortLevels = r.supported_effort_levels;
      })
      .catch(() => {});
  }

  onInput = (next: string): void => {
    writeDraft(this.#deps.sessionId, next);
    this.slashActive = 0;
    this.argActive = 0;
    this.slashDismissed = false;
  };

  pickSlash = (cmd: Command): void => {
    // A model command routes to the header picker rather than a text field.
    const sessionId = this.#deps.sessionId;
    if (commandArgParam(cmd)?.widget === 'model' && sessionId) {
      modelPickerRequest.request(sessionId);
      this.#deps.value = '';
      this.slashActive = 0;
      clearDraft(sessionId);
      return;
    }
    this.#deps.value = `/${cmd.name} `;
    this.slashActive = 0;
  };

  // Submit the command with the chosen argument value, running it through the same
  // dispatch (and inline echo) path as a typed-then-sent command.
  pickArgChoice = (choice: string): void => {
    const ctx = this.argContext;
    if (ctx) this.#deps.handleSend(`/${ctx.cmd.name} ${choice}`);
  };

  // Shared menu navigation for both stages: Tab/Enter picks the highlighted item,
  // arrows move the highlight (wrapping), Escape dismisses. Returns true when it
  // consumed the key so the composer doesn't also act on it.
  #navMenu<T>(
    e: KeyboardEvent,
    items: T[],
    active: number,
    setActive: (i: number) => void,
    choose: (item: T) => void,
  ): boolean {
    if (e.key === 'Tab' || e.key === 'Enter') {
      e.preventDefault();
      const it = items[active] ?? items[0];
      if (it !== undefined) choose(it);
      return true;
    }
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      setActive((active + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length);
      return true;
    }
    if (e.key === 'Escape') {
      e.preventDefault();
      this.slashDismissed = true;
      return true;
    }
    return false;
  }

  onComposerKeydown = (e: KeyboardEvent): boolean => {
    if (this.slashOpen)
      return this.#navMenu(
        e,
        this.slashMatches,
        this.slashActive,
        (i) => (this.slashActive = i),
        this.pickSlash,
      );
    if (this.argOpen && this.argChoices)
      return this.#navMenu(
        e,
        this.argChoices,
        this.argActive,
        (i) => (this.argActive = i),
        this.pickArgChoice,
      );
    return false;
  };

  async dispatchCommand(line: string): Promise<void> {
    const trimmed = line.trim().slice(1);
    const gap = trimmed.indexOf(' ');
    const name = (gap === -1 ? trimmed : trimmed.slice(0, gap)).toLowerCase();
    const rest = gap === -1 ? '' : trimmed.slice(gap + 1).trim();
    // Echo under the exact line the user ran (Claude-Code style), args and all.
    const label = `/${name}${rest ? ` ${rest}` : ''}`;
    const cmd = this.commands.find((c) => c.name === name);
    if (!cmd) {
      this.#deps.onCommandResult?.(label, `Unknown command /${name}`, false);
      return;
    }
    const body: Record<string, unknown> = {};
    if (cmd.params.some((p) => p.name === 'user_id')) body.user_id = auth.userId;
    const sessionId = this.#deps.sessionId;
    if (sessionId && cmd.params.some((p) => p.name === 'session_id')) body.session_id = sessionId;
    const primary = commandArgParam(cmd)?.name;
    if (primary && rest) body[primary] = rest;
    // The /job affordance moves from the toast's action button into the echo: a
    // link to the jobs board, matching the header's `#jobs` chip.
    const action = name === 'job' ? { label: 'Open jobs', href: '#jobs' } : undefined;
    try {
      const res = await api.post<{ result?: string }>(
        `/api/agents/${encodeURIComponent(this.#deps.agent)}/commands/${encodeURIComponent(name)}`,
        body,
      );
      this.#deps.onCommandResult?.(
        label,
        typeof res.result === 'string' ? res.result : '',
        true,
        action,
      );
    } catch (err) {
      this.#deps.onCommandResult?.(label, err instanceof Error ? err.message : String(err), false);
    }
  }
}
