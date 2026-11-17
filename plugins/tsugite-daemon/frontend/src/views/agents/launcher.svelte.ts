class LauncherState {
  isOpen = $state(false);
  agent = $state<string | null>(null);
  agentId = $state<string | null>(null);

  open(agentPath?: string, agentId?: string): void {
    this.agent = agentPath ?? null;
    this.agentId = agentId ?? null;
    this.isOpen = true;
  }

  close(): void {
    this.isOpen = false;
  }
}

export const launcher = new LauncherState();
