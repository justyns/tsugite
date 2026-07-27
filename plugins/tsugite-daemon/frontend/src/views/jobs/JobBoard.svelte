<script lang="ts">
  import Icon from '$lib/components/icon/Icon.svelte';
  import Badge from '$lib/components/buttons/Badge.svelte';
  import type { IconName } from '$lib/components/icon/icons';
  import { TESTID } from '$lib/testids';
  import type { Job } from '$lib/stores/jobs.svelte';
  import JobCard from './JobCard.svelte';
  import {
    ATTENTION_COL,
    BOARD_COLS,
    BOARD_COL_LABEL,
    type BoardCol,
    groupByColumn,
  } from './board';

  let {
    jobs,
    now,
    selectedId,
    onOpen,
  }: {
    jobs: Job[];
    now: number;
    selectedId: string | null;
    onOpen: (job: Job) => void;
  } = $props();

  const grouped = $derived(groupByColumn(jobs));

  const COL_ICON: Record<BoardCol, IconName> = {
    queued: 'clock',
    active: 'play',
    'needs-you': 'alert',
    resolved: 'check',
  };
  const COL_COLOR: Record<BoardCol, string | undefined> = {
    queued: 'var(--st-queue)',
    active: 'var(--st-ok)',
    'needs-you': undefined,
    resolved: 'var(--st-mute)',
  };
</script>

<div class="t-board" data-testid={TESTID.jobsBoard} aria-label="Jobs by state group">
  {#each BOARD_COLS as col (col)}
    {@const items = grouped[col]}
    <section
      class="t-col"
      class:t-col--attn={col === ATTENTION_COL}
      data-col={col}
      data-testid={TESTID.jobsColumn(col)}
    >
      <h3 class="t-col-hd">
        <Icon name={COL_ICON[col]} size={11} color={COL_COLOR[col]} />
        {BOARD_COL_LABEL[col]}
        <span class="hd-count">
          <Badge
            variant={col === ATTENTION_COL && items.length > 0 ? 'action' : 'info'}
            label={`${items.length} ${BOARD_COL_LABEL[col]}`}
          >
            {items.length}
          </Badge>
        </span>
      </h3>
      <div class="t-col-bd">
        {#each items as job (job.job_id)}
          <JobCard {job} {now} selected={job.job_id === selectedId} onOpen={() => onOpen(job)} />
        {/each}
      </div>
    </section>
  {/each}
</div>

<style>
  .t-board {
    display: grid;
    grid-auto-flow: column;
    grid-auto-columns: minmax(236px, 1fr);
    gap: 10px;
    overflow-x: auto;
    padding: 12px;
    height: 100%;
    align-items: stretch;
    scroll-snap-type: x proximity;
  }
  .t-col {
    background: var(--bg1);
    border: 1px solid var(--bd0);
    border-radius: var(--r-lg);
    display: flex;
    flex-direction: column;
    min-height: 0;
    scroll-snap-align: start;
  }
  .t-col-hd {
    display: flex;
    align-items: center;
    gap: 7px;
    margin: 0;
    padding: 8px 10px;
    border-bottom: 1px solid var(--bd0);
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--tx2);
    flex: none;
  }
  .hd-count {
    margin-left: auto;
  }
  .t-col--attn {
    border-color: color-mix(in oklab, var(--st-warn) 38%, transparent);
  }
  .t-col--attn .t-col-hd {
    color: var(--st-warn);
    border-bottom-color: color-mix(in oklab, var(--st-warn) 30%, transparent);
  }
  .t-col-bd {
    padding: 8px;
    display: grid;
    gap: 8px;
    overflow-y: auto;
    align-content: start;
    flex: 1;
  }

  /* Narrow: stack the groups vertically with needs-you first and let the board
     scroll as one page - no horizontal panning. */
  @media (max-width: 640px) {
    .t-board {
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
      overflow-y: auto;
      padding: 10px;
    }
    .t-col {
      flex: none;
    }
    .t-col-bd {
      overflow: visible;
    }
    .t-col[data-col='needs-you'] {
      order: -1;
    }
  }
</style>
