// The host bridge and DOM helpers both forge pages share.

import { age } from './filters.js';

const BASE = '/api/plugins/forge';
let token = '';

export async function api(path, method = 'GET', body) {
  const headers = { Authorization: 'Bearer ' + token };
  if (body !== undefined) headers['Content-Type'] = 'application/json';
  const resp = await fetch(BASE + path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `${path} returned ${resp.status}`);
  return data;
}

export function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function protocol(url) {
  try {
    return new URL(url).protocol;
  } catch {
    return '';
  }
}

/** A pull request title that links out, as plain text for any URL that is not http or https. */
export function prLink(title, url) {
  if (!['http:', 'https:'].includes(protocol(url))) return el('span', title);
  const link = el('a', title);
  link.href = url;
  link.target = '_blank';
  link.rel = 'noopener';
  return link;
}

/** A repo section: its heading led by its forge's icon, then `items`. `forge` is the forge's /status entry, if known. */
export function repoSection(text, forge, items) {
  const heading = el('h2');
  if (forge) {
    const icon = el('span', undefined, 'forge-icon');
    icon.dataset.kind = forge.kind;
    icon.setAttribute('role', 'img');
    icon.setAttribute('aria-label', forge.name);
    icon.title = forge.name;
    heading.append(icon);
  }
  heading.append(text);
  const list = el('ul');
  list.append(...items);
  const section = el('section');
  section.append(heading, list);
  return section;
}

export function prMeta(pr, now) {
  const spans = [
    el('span', `#${pr.number}`),
    el('span', pr.author),
    el('span', age(pr.created_at, now)),
    el('span', `CI ${pr.ci_state}`, `ci-${pr.ci_state}`),
  ];
  if (pr.unresolved_threads === null) spans.push(el('span', 'threads ?'));
  else if (pr.unresolved_threads > 0) spans.push(el('span', `${pr.unresolved_threads} unresolved`, 'unresolved'));
  if (pr.draft) spans.push(el('span', 'draft', 'badge'));
  for (const label of pr.labels) spans.push(el('span', label, 'badge'));
  return spans;
}

function applyTheme(theme) {
  for (const [name, value] of Object.entries(theme.tokens)) {
    document.documentElement.style.setProperty(name, value);
  }
}

/** Answer the host's `tsugite:init` with ready and `title`, then call `onInit(msg)`. */
export function connect(title, onInit) {
  // The host sees no pointer events inside the frame. Report them so it focuses this pane.
  const claimFocus = () => parent.postMessage({ type: 'tsugite:focus' }, location.origin);
  addEventListener('pointerdown', claimFocus);
  addEventListener('focusin', claimFocus);

  addEventListener('message', (event) => {
    const msg = event.data;
    if (event.origin !== location.origin || !msg || typeof msg !== 'object') return;
    if (msg.type === 'tsugite:init') {
      applyTheme(msg.theme);
      token = msg.token;
      parent.postMessage({ type: 'tsugite:ready' }, location.origin);
      parent.postMessage({ type: 'tsugite:title', title }, location.origin);
      onInit(msg);
    } else if (msg.type === 'tsugite:theme') {
      applyTheme(msg.theme);
    }
  });
}
