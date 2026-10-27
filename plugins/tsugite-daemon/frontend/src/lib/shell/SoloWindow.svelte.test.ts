/// <reference types="@vitest/browser/context" />
import { page } from '@vitest/browser/context';
import { render } from 'vitest-browser-svelte';
import { afterEach, expect, test, vi } from 'vitest';
import SoloWindow from './SoloWindow.svelte';
import { sessions } from '$lib/stores/sessions.svelte';
import { artifacts } from '$lib/stores/artifacts.svelte';
import { TESTID } from '$lib/testids';
import { sessionRow } from '../../views/chats/__fixtures__/sessionRow';

const realGetInfo = sessions.getInfo.bind(sessions);

afterEach(async () => {
  await page.viewport(1440, 900);
  sessions.getInfo = realGetInfo;
  sessions.rows = [];
  artifacts.items = {};
  vi.restoreAllMocks();
});

function stubSessionIo() {
  sessions.getInfo = async () => ({ metadata: {}, contextLimit: null, cumulativeTokens: null });
  vi.spyOn(sessions, 'load').mockResolvedValue();
}

function openAgentArtifact() {
  artifacts.applySessionEvent({
    event_type: 'artifact_open',
    artifact_id: 'doc',
    session_id: 's1',
    content: '# hi',
    content_type: 'markdown',
    title: 'Doc',
    opened_by: 'agent',
  });
}

function expectNoShellParts(container: HTMLElement) {
  for (const id of [TESTID.navRail, TESTID.spaceBar, TESTID.topbar, TESTID.mux]) {
    expect(container.querySelector(`[data-testid="${id}"]`)).toBeNull();
  }
  expect(container.querySelector('.work-rail')).toBeNull();
  expect(container.querySelector(`[data-testid="${TESTID.chatRailToggle}"]`)).toBeNull();
}

test('renders one chat with no nav rail, space bar, context rail or tab strip', async () => {
  stubSessionIo();
  sessions.rows = [sessionRow('s1', { title: 'solo chat' })];
  const { container } = await render(SoloWindow, { params: { sessionId: 's1' } });
  await expect.element(page.getByTestId(TESTID.chatConversation)).toBeInTheDocument();
  await expect.element(page.getByTestId(TESTID.chatComposer)).toBeInTheDocument();
  expectNoShellParts(container);
});

test('an agent artifact docks beside the chat and no rails appear', async () => {
  stubSessionIo();
  sessions.rows = [sessionRow('s1')];
  openAgentArtifact();
  const { container } = await render(SoloWindow, { params: { sessionId: 's1' } });
  await expect.element(page.getByTestId(TESTID.soloArtifact)).toBeInTheDocument();
  await expect.element(page.getByTestId(TESTID.chatConversation)).toBeInTheDocument();
  expectNoShellParts(container);
});

test('at 700px and narrower an open artifact shows one pane at a time with a chat/artifact toggle', async () => {
  await page.viewport(700, 780);
  stubSessionIo();
  sessions.rows = [sessionRow('s1')];
  openAgentArtifact();
  await render(SoloWindow, { params: { sessionId: 's1' } });
  const chat = page.getByTestId(TESTID.chatConversation);
  const artifact = page.getByTestId(TESTID.soloArtifact);
  await expect.element(chat).toBeVisible();
  await expect.element(artifact).not.toBeVisible();
  await page.getByRole('button', { name: 'artifact' }).click();
  await expect.element(artifact).toBeVisible();
  await expect.element(chat).not.toBeVisible();
  await page.getByRole('button', { name: 'chat' }).click();
  await expect.element(chat).toBeVisible();
});

test('wider than 700px an open artifact sits beside the chat with no toggle', async () => {
  await page.viewport(701, 780);
  stubSessionIo();
  sessions.rows = [sessionRow('s1')];
  openAgentArtifact();
  const { container } = await render(SoloWindow, { params: { sessionId: 's1' } });
  await expect.element(page.getByTestId(TESTID.soloArtifact)).toBeVisible();
  await expect.element(page.getByTestId(TESTID.chatConversation)).toBeVisible();
  await expect.element(container.querySelector<HTMLElement>('.t-seg')!).not.toBeVisible();
});

test('a superseded sessionId says so and links to the successor', async () => {
  stubSessionIo();
  sessions.rows = [sessionRow('old', { superseded_by: 'new' }), sessionRow('new')];
  await render(SoloWindow, { params: { sessionId: 'old' } });
  await expect.element(page.getByText('compacted into a successor')).toBeInTheDocument();
  await expect
    .element(page.getByRole('link', { name: 'Open the successor' }))
    .toHaveAttribute('href', '#solo?sessionId=new');
});

test('an unknown sessionId shows not-found after the fetch fails', async () => {
  stubSessionIo();
  vi.spyOn(sessions, 'getDetail').mockRejectedValue(
    Object.assign(new Error('not found'), { status: 404 }),
  );
  await render(SoloWindow, { params: { sessionId: 'nope' } });
  await expect.element(page.getByText('exists.')).toBeInTheDocument();
  await expect.element(page.getByText('nope')).toBeInTheDocument();
});

test('a sessionId missing from the list but present on the server resolves', async () => {
  stubSessionIo();
  vi.spyOn(sessions, 'getDetail').mockResolvedValue({ id: 'fresh' });
  await render(SoloWindow, { params: { sessionId: 'fresh' } });
  await expect.element(page.getByTestId(TESTID.chatConversation)).toBeInTheDocument();
});

test('an unknown alias shows an empty state and picks up a later claim', async () => {
  stubSessionIo();
  await render(SoloWindow, { params: { alias: 'assistant' } });
  await expect.element(page.getByTestId(TESTID.soloNote)).toHaveTextContent('assistant');
  sessions.rows = [sessionRow('s9', { alias: 'assistant' })];
  await expect.element(page.getByTestId(TESTID.chatConversation)).toBeInTheDocument();
});

test('a fetch failure other than 404 shows a retry, not not-found', async () => {
  stubSessionIo();
  const getDetail = vi
    .spyOn(sessions, 'getDetail')
    .mockRejectedValueOnce(Object.assign(new Error('boom'), { status: 500 }))
    .mockResolvedValue({ id: 'flaky' });
  await render(SoloWindow, { params: { sessionId: 'flaky' } });
  await expect.element(page.getByText("Couldn't load session")).toBeInTheDocument();
  await expect.element(page.getByText('exists.')).not.toBeInTheDocument();
  await page.getByRole('button', { name: 'Retry' }).click();
  await expect.element(page.getByTestId(TESTID.chatConversation)).toBeInTheDocument();
  expect(getDetail).toHaveBeenCalledTimes(2);
});

test('a fetched session outside the list reports its successor', async () => {
  stubSessionIo();
  vi.spyOn(sessions, 'getDetail').mockResolvedValue({ id: 'old', superseded_by: 'new' });
  await render(SoloWindow, { params: { sessionId: 'old' } });
  await expect.element(page.getByText('compacted into a successor')).toBeInTheDocument();
  await expect
    .element(page.getByRole('link', { name: 'Open the successor' }))
    .toHaveAttribute('href', '#solo?sessionId=new');
});

test('a failed list load shows a retry instead of loading forever', async () => {
  stubSessionIo();
  let down = true;
  vi.spyOn(sessions, 'load').mockImplementation(async () => {
    sessions.error = down ? 'down' : null;
  });
  sessions.rows = [sessionRow('s1')];
  await render(SoloWindow, { params: { sessionId: 's1' } });
  await expect.element(page.getByText("Couldn't load the session list.")).toBeInTheDocument();
  down = false;
  await page.getByRole('button', { name: 'Retry' }).click();
  await expect.element(page.getByTestId(TESTID.chatConversation)).toBeInTheDocument();
});
