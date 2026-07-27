<script lang="ts">
  // Settings drawer (overlays/Drawer) - old-UI parity fields: token, user id,
  // theme, auto-follow, and push subscribe/unsubscribe. Credentials commit on
  // blur/enter (the field's change event) so a live connection never picks up a
  // half-typed token; theme/auto-follow/push apply immediately. Default-agent
  // and model-override are deferred - they need the agents list + config API.
  import Drawer from '$lib/components/overlays/Drawer.svelte';
  import Field from '$lib/components/inputs/Field.svelte';
  import Input from '$lib/components/inputs/Input.svelte';
  import Switch from '$lib/components/inputs/Switch.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import ThemeSeg from './ThemeSeg.svelte';
  import { api } from '$lib/api/client';
  import { auth } from '$lib/stores/auth.svelte';
  import { togglePush } from '$lib/push';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { readLocal, writeLocal } from '$lib/storage';
  import { contextProviders } from '$lib/context/contextProviders';
  import { autoAttachStore } from '$lib/stores/autoAttach.svelte';
  import { TESTID } from '$lib/testids';

  let { open = false, onclose }: { open?: boolean; onclose: () => void } = $props();

  // Frontend-only pref, consumed by the conversation view once wired.
  const AUTO_FOLLOW_KEY = 'tsugite_auto_follow';

  // Context providers that offer a per-device "auto-attach to every send" toggle.
  const autoAttachProviders = contextProviders.filter((p) => p.autoAttachStoreKey);

  let tokenDraft = $state(auth.token);
  let userIdDraft = $state(auth.userId);
  let autoFollow = $state(readLocal(AUTO_FOLLOW_KEY) !== 'false');

  function commitCredentials(event: Event) {
    const target = event.target as HTMLInputElement;
    if (target.id === TESTID.tokenInput) {
      const value = tokenDraft.trim();
      if (value) auth.save(value);
    } else if (target.id === 'settings-userid') {
      const value = userIdDraft.trim();
      if (value) auth.setUserId(value);
    }
  }

  $effect(() => {
    writeLocal(AUTO_FOLLOW_KEY, String(autoFollow));
  });

  let pushBusy = $state(false);
  let pushSubscribed = $state(false);

  // Best-effort read of the current subscription; resolves only where a service
  // worker exists (prod builds), a no-op in dev.
  $effect(() => {
    if (typeof navigator === 'undefined' || !navigator.serviceWorker) return;
    navigator.serviceWorker.ready
      .then((reg) => reg.pushManager.getSubscription())
      .then((sub) => {
        pushSubscribed = !!sub;
      })
      .catch(() => {});
  });

  async function onTogglePush() {
    pushBusy = true;
    try {
      pushSubscribed = await togglePush(pushSubscribed);
      toasts.push('ok', pushSubscribed ? 'Notifications enabled' : 'Notifications disabled');
    } catch (err) {
      toasts.push('err', 'Notification toggle failed', { body: (err as Error).message });
    } finally {
      pushBusy = false;
    }
  }

  let reloading = $state(false);
  async function onReloadConfig() {
    reloading = true;
    try {
      const r = await api.post<{
        added: string[];
        removed: string[];
        updated: string[];
        skipped: string[];
        restart_required: string[];
      }>('/api/daemon/reload-config');
      const changes = [
        r.added.length ? `+${r.added.join(', ')}` : '',
        r.removed.length ? `-${r.removed.join(', ')}` : '',
        r.updated.length ? `~${r.updated.join(', ')}` : '',
      ]
        .filter(Boolean)
        .join(' · ');
      const restart = r.restart_required.length
        ? `restart required for: ${r.restart_required.join(', ')}`
        : '';
      toasts.push(restart ? 'warn' : 'ok', 'Config reloaded', {
        body: [changes || 'no agent changes', restart].filter(Boolean).join(' — '),
      });
    } catch (err) {
      toasts.push('err', 'Config reload failed', { body: (err as Error).message });
    } finally {
      reloading = false;
    }
  }
</script>

<!-- Fixed overlay host: the drawer must never participate in app layout (it
     pushed the whole shell sideways and caused a reflow flicker on open). -->
<div class="drawer-host" class:is-open={open} aria-hidden={!open}>
  <Drawer {open} {onclose} title="Settings">
    <div class="settings" data-testid={TESTID.settingsDrawer}>
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <section class="d-sec" onchange={commitCredentials}>
        <h4>Connection</h4>
        <div class="d-fields">
          <Field label="Access token" id={TESTID.tokenInput}>
            {#snippet children(describedBy)}
              <Input
                type="password"
                bind:value={tokenDraft}
                id={TESTID.tokenInput}
                ariaDescribedby={describedBy}
                placeholder="access token"
                mono
              />
            {/snippet}
          </Field>
          <Field label="User ID" id="settings-userid" hint="Sent with every request as user_id.">
            {#snippet children(describedBy)}
              <Input
                bind:value={userIdDraft}
                id="settings-userid"
                ariaDescribedby={describedBy}
                mono
              />
            {/snippet}
          </Field>
        </div>
      </section>

      <section class="d-sec">
        <h4>Appearance</h4>
        <ThemeSeg testid={TESTID.themeSwitch} />
      </section>

      <section class="d-sec">
        <h4>Behavior</h4>
        <div class="d-toggle">
          <Switch bind:checked={autoFollow} ariaLabel="Auto-follow new output" />
          <div class="d-toggle-lb">
            <span class="tt">Auto-follow</span>
            <span class="sub">Keep the transcript pinned to the newest output.</span>
          </div>
        </div>
      </section>

      {#if autoAttachProviders.length}
        <section class="d-sec">
          <h4>Context</h4>
          {#each autoAttachProviders as provider (provider.key)}
            {@const store = autoAttachStore(provider.autoAttachStoreKey!)}
            <div class="d-toggle" data-testid={TESTID.settingsContextAutoattach(provider.key)}>
              <Switch
                checked={store.enabled}
                onCheckedChange={(v) => store.set(v)}
                ariaLabel={`Auto-attach my ${provider.label.toLowerCase()} to messages`}
              />
              <div class="d-toggle-lb">
                <span class="tt">Auto-attach {provider.label.toLowerCase()}</span>
                <span class="sub">
                  Add your {provider.label.toLowerCase()} to every message you send, so the agent has
                  it. Off by default; you can also attach it per-message from the composer's "context"
                  menu.
                </span>
              </div>
            </div>
          {/each}
        </section>
      {/if}

      <section class="d-sec">
        <h4>Notifications</h4>
        <div class="d-toggle">
          <Button loading={pushBusy} onclick={onTogglePush}>
            {pushSubscribed ? 'Disable' : 'Enable'}
          </Button>
          <div class="d-toggle-lb">
            <span class="tt">Push notifications</span>
            <span class="sub">Job answers and completions arrive as web-push alerts.</span>
          </div>
        </div>
      </section>

      <section class="d-sec">
        <h4>Daemon</h4>
        <div class="d-toggle">
          <Button loading={reloading} onclick={onReloadConfig}>Reload config</Button>
          <div class="d-toggle-lb">
            <span class="tt">Reload daemon config</span>
            <span class="sub">
              Re-reads daemon.yaml and hot-applies the agent list; boot-only sections report as
              restart-required.
            </span>
          </div>
        </div>
      </section>
    </div>
  </Drawer>
</div>

<style>
  .drawer-host {
    position: fixed;
    inset: 0;
    z-index: 60;
    pointer-events: none;
  }
  .drawer-host.is-open {
    pointer-events: auto;
  }
  .settings {
    display: grid;
    gap: 16px;
  }
  /* .d-sec - drawer sections */
  .d-sec > h4 {
    margin: 0 0 7px;
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .d-fields {
    display: grid;
    gap: 12px;
  }
  .d-toggle {
    display: flex;
    align-items: flex-start;
    gap: 10px;
  }
  .d-toggle-lb {
    display: grid;
    gap: 1px;
    min-width: 0;
  }
  .d-toggle-lb .tt {
    font: 600 var(--fs-sm) var(--font-ui);
    color: var(--tx1);
  }
  .d-toggle-lb .sub {
    font: 400 var(--fs-xs) / 1.45 var(--font-ui);
    color: var(--tx3);
    text-wrap: pretty;
  }
</style>
