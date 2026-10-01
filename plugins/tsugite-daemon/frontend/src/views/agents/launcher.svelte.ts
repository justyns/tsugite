class LauncherState {
  isOpen = $state(false);
  agent = $state<string | null>(null);

  open(agentPath?: string): void {
    this.agent = agentPath ?? null;
    this.isOpen = true;
  }

  close(): void {
    this.isOpen = false;
  }
}

export const launcher = new LauncherState();
