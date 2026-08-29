<script lang="ts">
  // Create / edit a schedule, plus its recent-run timeline. Wraps the shared
  // <Drawer> (inspection panel); the form seeds from the target schedule on open
  // and writes back through the schedules store. Enable/disable is a live toggle
  // (its own endpoint) - the daemon's PATCH doesn't accept `enabled` - so it acts
  // immediately rather than waiting for Save.
  import { untrack } from 'svelte';
  import Drawer from '$lib/components/overlays/Drawer.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Field from '$lib/components/inputs/Field.svelte';
  import Input from '$lib/components/inputs/Input.svelte';
  import Select from '$lib/components/inputs/Select.svelte';
  import Seg from '$lib/components/inputs/Seg.svelte';
  import PaneState from '$lib/components/connstates/PaneState.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { schedules, type Schedule, type ScheduleSession } from '$lib/stores/schedules.svelte';
  import { TESTID } from '$lib/testids';
  import { describeCron } from './cron';
  import { deriveRunStatus, formatDuration, formatStamp, recentRuns } from './schedulesView';
  import RunStatusPill from './RunStatusPill.svelte';
  import EnableSwitch from './EnableSwitch.svelte';

  let {
    open = false,
    schedule = null,
    agents = [],
    onclose,
    onchanged,
    onRequestDelete,
  }: {
    open?: boolean;
    /** null => create mode. */
    schedule?: Schedule | null;
    agents?: string[];
    onclose?: () => void;
    /** The set changed (create / update / run / enable / disable); parent reloads. */
    onchanged?: () => void;
    onRequestDelete?: (schedule: Schedule) => void;
  } = $props();

  interface FormState {
    id: string;
    // Kept as plain strings so the shared <Seg> (a string-bindable) can two-way
    // bind them directly; narrowed with `===` reads and cast at the API boundary.
    execution_type: string;
    schedule_type: string;
    cron_expr: string;
    run_at: string; // datetime-local value
    prompt: string;
    command: string;
    agent: string;
    timezone: string;
  }

  const browserTz = (() => {
    try {
      return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC';
    } catch {
      return 'UTC';
    }
  })();

  function pad2(n: number): string {
    return String(n).padStart(2, '0');
  }
  /** UTC ISO -> "YYYY-MM-DDTHH:MM" in local time for a datetime-local input. */
  function toLocalInput(iso: string | null): string {
    if (!iso) return '';
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '';
    return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}T${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
  }
  /** datetime-local value (local wall time) -> absolute UTC ISO instant. */
  function fromLocalInput(value: string): string | null {
    if (!value) return null;
    const d = new Date(value);
    return Number.isNaN(d.getTime()) ? null : d.toISOString();
  }

  function seed(s: Schedule | null): FormState {
    if (!s) {
      return {
        id: '',
        execution_type: 'agent',
        schedule_type: 'cron',
        cron_expr: '',
        run_at: '',
        prompt: '',
        command: '',
        agent: agents[0] ?? '',
        timezone: browserTz,
      };
    }
    return {
      id: s.id,
      execution_type: s.execution_type,
      schedule_type: s.schedule_type === 'once' ? 'once' : 'cron',
      cron_expr: s.cron_expr ?? '',
      run_at: toLocalInput(s.run_at),
      prompt: s.prompt ?? '',
      command: s.command ?? '',
      agent: s.agent,
      timezone: s.timezone || 'UTC',
    };
  }

  let form = $state<FormState>(seed(null));
  let enabled = $state(true);
  let saving = $state(false);

  const isEdit = $derived(schedule != null);

  // Reseed exactly once per open (or when the target schedule changes while open),
  // never on every keystroke. The key flips only on those transitions.
  const seedKey = $derived(open ? (schedule?.id ?? '__new__') : '__closed__');
  $effect(() => {
    seedKey; // track
    if (!open) return;
    untrack(() => {
      form = seed(schedule);
      enabled = schedule ? schedule.enabled : true;
    });
  });

  const cronHelp = $derived(describeCron(form.cron_expr));

  const canSave = $derived.by(() => {
    if (!form.id.trim() || !form.agent || !form.prompt.trim()) return false;
    if (form.schedule_type === 'cron' && !form.cron_expr.trim()) return false;
    if (form.schedule_type === 'once' && !form.run_at) return false;
    if (form.execution_type === 'script' && !form.command.trim()) return false;
    return true;
  });

  // --- recent runs timeline ---
  let runs = $state<ScheduleSession[]>([]);
  let runsLoading = $state(false);
  let runsError = $state<string | null>(null);

  async function loadRuns(id: string): Promise<void> {
    runsLoading = true;
    runsError = null;
    try {
      runs = recentRuns(await schedules.sessions(id));
    } catch (err) {
      runsError = err instanceof Error ? err.message : String(err);
    } finally {
      runsLoading = false;
    }
  }

  const runsKey = $derived(open && schedule ? schedule.id : '');
  $effect(() => {
    const id = runsKey;
    if (!id) {
      runs = [];
      return;
    }
    void loadRuns(id);
  });

  /* Node tone for a run status; the tone -> token mapping lives in CSS. */
  function runTone(status: string): 'ok' | 'err' | 'run' | 'mute' {
    const s = status.toLowerCase();
    if (s.includes('complete') || s.includes('success') || s === 'done') return 'ok';
    if (s.includes('fail') || s.includes('error')) return 'err';
    if (s.includes('active') || s.includes('running')) return 'run';
    return 'mute';
  }
  function isRunning(status: string): boolean {
    const s = status.toLowerCase();
    return s.includes('active') || s.includes('running');
  }
  function runDuration(r: ScheduleSession): string | null {
    if (!r.created_at || !r.last_active) return null;
    const a = Date.parse(r.created_at);
    const b = Date.parse(r.last_active);
    if (Number.isNaN(a) || Number.isNaN(b) || b < a) return null;
    return formatDuration(b - a);
  }
  // --- actions ---
  async function save(): Promise<void> {
    if (!canSave || saving) return;
    saving = true;
    try {
      if (isEdit && schedule) {
        const fields: Partial<Schedule> = {
          prompt: form.prompt,
          agent: form.agent,
          timezone: form.timezone,
          schedule_type: form.schedule_type as Schedule['schedule_type'],
          execution_type: form.execution_type as Schedule['execution_type'],
        };
        if (form.schedule_type === 'cron') fields.cron_expr = form.cron_expr.trim();
        else fields.run_at = fromLocalInput(form.run_at);
        if (form.execution_type === 'script') fields.command = form.command.trim();
        await schedules.update(schedule.id, fields);
        toasts.push('ok', 'Schedule saved', { body: `${form.id} · next run recalculated` });
      } else {
        const body: Partial<Schedule> & {
          id: string;
          agent: string;
          prompt: string;
          schedule_type: string;
        } = {
          id: form.id.trim(),
          agent: form.agent,
          prompt: form.prompt,
          schedule_type: form.schedule_type as Schedule['schedule_type'],
          execution_type: form.execution_type as Schedule['execution_type'],
          timezone: form.timezone,
          enabled,
        };
        if (form.schedule_type === 'cron') body.cron_expr = form.cron_expr.trim();
        else body.run_at = fromLocalInput(form.run_at) ?? undefined;
        if (form.execution_type === 'script') body.command = form.command.trim();
        await schedules.create(body);
        toasts.push('ok', 'Schedule created', { body: form.id.trim() });
      }
      onchanged?.();
      onclose?.();
    } catch (err) {
      toasts.push('err', 'Could not save schedule', {
        body: err instanceof Error ? err.message : String(err),
      });
    } finally {
      saving = false;
    }
  }

  async function toggleEnabled(next: boolean): Promise<void> {
    enabled = next; // optimistic
    if (!isEdit || !schedule) return; // create mode: folded into the create body
    try {
      if (next) await schedules.enable(schedule.id);
      else await schedules.disable(schedule.id);
      onchanged?.();
    } catch (err) {
      enabled = !next; // revert
      toasts.push('err', 'Could not update schedule', {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }

  async function runNow(): Promise<void> {
    if (!schedule) return;
    try {
      await schedules.run(schedule.id);
      toasts.push('info', 'Run queued', { body: `${schedule.id} starts on the next tick` });
      onchanged?.();
    } catch (err) {
      toasts.push('err', 'Could not run schedule', {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }
</script>

<Drawer
  {open}
  {onclose}
  title={isEdit ? (schedule?.id ?? '') : 'New schedule'}
  label="Schedule detail"
>
  {#snippet status()}
    {#if isEdit && schedule}
      <RunStatusPill status={deriveRunStatus(schedule)} />
    {/if}
  {/snippet}

  {#snippet children()}
    <form
      class="d-sec"
      data-testid={TESTID.scheduleForm}
      onsubmit={(e) => {
        e.preventDefault();
        void save();
      }}
    >
      <h4>definition</h4>
      <div class="fields">
        <Field id="sd-name" label="name">
          {#snippet children(describedBy)}
            <Input
              id="sd-name"
              bind:value={form.id}
              mono
              placeholder="nightly-backup"
              disabled={isEdit}
              ariaDescribedby={describedBy}
            />
          {/snippet}
        </Field>

        <div class="row">
          <Seg
            options={isEdit ? ['agent', 'script', 'session_message'] : ['agent', 'script']}
            bind:value={form.execution_type}
            ariaLabel="Execution type"
          />
          <Seg
            options={['cron', 'once']}
            bind:value={form.schedule_type}
            ariaLabel="Schedule kind"
          />
        </div>

        {#if form.schedule_type === 'cron'}
          <Field id="sd-cron" label="cadence (cron)">
            {#snippet children(describedBy)}
              <Input
                id="sd-cron"
                bind:value={form.cron_expr}
                mono
                placeholder="0 3 * * *"
                ariaDescribedby={describedBy}
              />
            {/snippet}
          </Field>
          <span class="cronh" aria-live="polite">
            {cronHelp ?? 'unrecognized expression — will run on the raw schedule'}
          </span>
        {:else}
          <div class="t-field">
            <label for="sd-runat">run at</label>
            <input
              id="sd-runat"
              class="t-input mono"
              type="datetime-local"
              bind:value={form.run_at}
            />
          </div>
        {/if}

        <div class="t-field">
          <label for="sd-prompt"
            >task prompt <span class="req" aria-hidden="true">required</span></label
          >
          <textarea id="sd-prompt" class="t-input" rows="2" bind:value={form.prompt}></textarea>
        </div>

        {#if form.execution_type === 'script'}
          <Field id="sd-cmd" label="command" hint="runs via /bin/sh -c">
            {#snippet children(describedBy)}
              <Input
                id="sd-cmd"
                bind:value={form.command}
                mono
                placeholder="rsync -a ~/workspace/ backup:snap/"
                ariaDescribedby={describedBy}
              />
            {/snippet}
          </Field>
        {/if}

        <div class="row wrap">
          <label class="stack">
            <span class="mlabel">agent</span>
            <Select options={agents} bind:value={form.agent} ariaLabel="Agent" />
          </label>
          <label class="stack">
            <span class="mlabel">timezone</span>
            <Input bind:value={form.timezone} mono ariaLabel="Timezone" />
          </label>
          <span class="switchrow">
            <EnableSwitch
              checked={enabled}
              ariaLabel="Schedule enabled"
              onToggle={(next) => void toggleEnabled(next)}
            />
            enabled
          </span>
        </div>
      </div>
    </form>

    {#if isEdit}
      <div class="d-sec">
        <h4>recent runs</h4>
        {#if runsLoading}
          <PaneState kind="loading" lines={3} />
        {:else if runsError}
          <PaneState kind="error" title="Couldn't load run history">
            {#snippet hint()}<span>{runsError}</span>{/snippet}
            {#snippet actions()}
              <Button size="sm" onclick={() => schedule && loadRuns(schedule.id)}>Retry</Button>
            {/snippet}
          </PaneState>
        {:else if runs.length === 0}
          <PaneState kind="empty" title="No runs yet">
            {#snippet icon()}<Icon name="sched" />{/snippet}
            {#snippet hint()}<span>This schedule hasn't fired. Use Run now to trigger it.</span
              >{/snippet}
          </PaneState>
        {:else}
          <div class="t-tl">
            {#each runs as r (r.id)}
              {@const dur = runDuration(r)}
              <div class="t-tl-it" class:is-now={isRunning(r.status)}>
                <span class="nd"><i data-tone={runTone(r.status)}></i></span>
                <span class="lb">
                  {r.status}
                  {#if dur}<span class="d">· {dur}</span>{/if}
                  {#if r.error}<span class="d err">· {r.error}</span>{/if}
                </span>
                <span class="at">{formatStamp(r.created_at)}</span>
              </div>
            {/each}
          </div>
        {/if}
      </div>
    {/if}
  {/snippet}

  {#snippet footer()}
    <Button
      variant="pri"
      size="sm"
      loading={saving}
      disabled={!canSave}
      data-testid={TESTID.scheduleSave}
      onclick={() => void save()}
    >
      {isEdit ? 'Save changes' : 'Create schedule'}
    </Button>
    {#if isEdit}
      <Button size="sm" data-testid={TESTID.scheduleDrawerRun} onclick={() => void runNow()}>
        {#snippet icon()}<Icon name="play" />{/snippet}
        Run now
      </Button>
      <div class="grow"></div>
      <Button
        variant="danger"
        size="sm"
        data-testid={TESTID.scheduleDelete}
        onclick={() => schedule && onRequestDelete?.(schedule)}
      >
        Delete
      </Button>
    {/if}
  {/snippet}
</Drawer>

<style>
  .d-sec > h4 {
    margin: 0 0 7px;
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .fields {
    display: grid;
    gap: 10px;
  }
  .row {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .row.wrap {
    flex-wrap: wrap;
  }
  .stack {
    display: grid;
    gap: 5px;
  }
  .mlabel,
  .t-field label {
    font: 600 var(--fs-2xs) var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--tx3);
  }
  .switchrow {
    display: flex;
    align-items: center;
    gap: 6px;
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx2);
    align-self: end;
    padding-bottom: 4px;
  }
  .t-field {
    display: grid;
    gap: 5px;
  }
  .cronh {
    font: 400 var(--fs-2xs) var(--font-ui);
    color: var(--tx3);
    margin-top: -4px;
  }

  /* raw controls the shared Input/Field don't cover (textarea, datetime-local): .t-input */
  .t-input {
    width: 100%;
    background: var(--bg1);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    padding: 6px 9px;
    color: var(--tx0);
    font: 400 var(--fs-md) var(--font-ui);
    transition:
      border-color var(--t-1),
      box-shadow var(--t-1);
  }
  textarea.t-input {
    resize: vertical;
    min-height: 44px;
    line-height: 1.5;
  }
  input.t-input {
    height: 28px;
  }
  .t-input.mono {
    font-family: var(--font-mono);
  }
  .t-input::placeholder {
    color: var(--tx3);
  }
  .t-input:focus {
    outline: none;
    border-color: var(--acc);
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--acc) 22%, transparent);
  }

  /* timeline (.t-tl) */
  .t-tl {
    display: grid;
    gap: 0;
    font-size: var(--fs-xs);
  }
  .t-tl-it {
    display: grid;
    grid-template-columns: 14px 1fr auto;
    gap: 0 9px;
    position: relative;
    padding: 0 0 12px;
  }
  .t-tl-it::before {
    content: '';
    position: absolute;
    left: 6.5px;
    top: 13px;
    bottom: 1px;
    width: 1px;
    background: var(--bd1);
  }
  .t-tl-it:last-child::before {
    display: none;
  }
  .t-tl-it .nd {
    width: 14px;
    height: 14px;
    display: grid;
    place-items: center;
    margin-top: 1px;
  }
  .t-tl-it .nd i {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--c, var(--bd1));
    display: block;
  }
  .t-tl-it .nd i[data-tone='ok'] {
    --c: var(--st-ok);
  }
  .t-tl-it .nd i[data-tone='err'] {
    --c: var(--st-err);
  }
  .t-tl-it .nd i[data-tone='run'] {
    --c: var(--st-warn);
  }
  .t-tl-it .nd i[data-tone='mute'] {
    --c: var(--st-mute);
  }
  .t-tl-it.is-now .nd i {
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--c, var(--st-warn)) 22%, transparent);
  }
  .t-tl-it .lb {
    color: var(--tx1);
    font-family: var(--font-mono);
    min-width: 0;
  }
  .t-tl-it .lb .d {
    color: var(--tx3);
  }
  .t-tl-it .lb .d.err {
    color: var(--st-err);
  }
  .t-tl-it .at {
    color: var(--tx3);
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
  }
  .grow {
    flex: 1;
  }

  .req {
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--st-warn);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-left: 4px;
  }
</style>
