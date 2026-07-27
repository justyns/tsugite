/**
 * Frontmatter parser for the agent builder. There is no client YAML library and
 * no server endpoint that returns a parsed AgentConfig, so the builder parses the
 * `.md` source itself to drive the structured Form tab. The Markdown tab remains
 * the editable source of truth (PUT /api/agent-files/content); this parser is a
 * read-only reflection, so it favours degrading gracefully (an unrecognised value
 * becomes a plain string) over throwing.
 *
 * It covers the YAML subset agent frontmatter actually uses: scalars (with quote
 * stripping + int/bool/null coercion), block sequences of scalars, block
 * sequences of maps (prefetch), one-level nested maps (sandbox/network), and
 * block scalars (`|` / `>`, e.g. instructions). Comments inside a block scalar
 * are preserved; structural comments are stripped.
 */

// `splitFrontmatter` (fence detection: BOM/CRLF tolerance, unterminated-fence
// handling) is shared with the files wiki's markdown renderer - see
// `../files/frontmatter` for why it lives there. Re-exported here so existing
// importers of `./agentFrontmatter` are unaffected.
import { splitFrontmatter } from '../files/frontmatter';
export { splitFrontmatter };

export type YamlValue = string | number | boolean | null | YamlValue[] | { [k: string]: YamlValue };

export interface ParsedAgentFile {
  /** Best-effort parsed frontmatter map (empty when there is no frontmatter). */
  frontmatter: Record<string, YamlValue>;
  /** Markdown body after the closing `---` (the prompt template). */
  body: string;
  /** Whether a leading `---` fenced frontmatter block was present. */
  hasFrontmatter: boolean;
}

function indentOf(line: string): number {
  let n = 0;
  while (n < line.length && line[n] === ' ') n++;
  return n;
}

/** A structural line is blank if it is empty or only a comment. */
function isSkippable(line: string): boolean {
  const t = line.trim();
  return t === '' || t.startsWith('#');
}

/** Strip an unquoted trailing `# comment`. Quoted scalars keep their `#`. */
function stripTrailingComment(value: string): string {
  const hashAt = value.indexOf(' #');
  return hashAt >= 0 ? value.slice(0, hashAt) : value;
}

/** Coerce a scalar token into a typed value (quotes, bool, null, number). */
export function coerceScalar(rawInput: string): YamlValue {
  const raw = rawInput.trim();
  if (raw === '') return '';
  const first = raw[0];
  if (first === '"' || first === "'") {
    const closer = raw.lastIndexOf(first);
    if (closer > 0) return raw.slice(1, closer);
  }
  const bare = stripTrailingComment(raw).trim();
  if (bare === '~' || bare === 'null' || bare === 'Null' || bare === 'NULL') return null;
  if (bare === 'true' || bare === 'True' || bare === 'TRUE') return true;
  if (bare === 'false' || bare === 'False' || bare === 'FALSE') return false;
  if (bare === '{}') return {};
  if (bare === '[]') return [];
  // Inline flow list of scalars, e.g. [a, b, c].
  if (bare.startsWith('[') && bare.endsWith(']')) {
    const inner = bare.slice(1, -1).trim();
    return inner === '' ? [] : inner.split(',').map((s) => coerceScalar(s));
  }
  if (/^-?\d+$/.test(bare)) return parseInt(bare, 10);
  if (/^-?\d*\.\d+$/.test(bare)) return parseFloat(bare);
  return bare;
}

interface Cursor {
  lines: string[];
  i: number;
}

/** Parse a block scalar (`|` or `>`) body: every line indented deeper than the
 *  key. Folding style is preserved literally enough for display. */
function parseBlockScalar(cur: Cursor, parentIndent: number, fold: boolean): string {
  const collected: string[] = [];
  let blockIndent = -1;
  while (cur.i < cur.lines.length) {
    const line = cur.lines[cur.i]!;
    if (line.trim() === '') {
      collected.push('');
      cur.i++;
      continue;
    }
    const ind = indentOf(line);
    if (ind <= parentIndent) break;
    if (blockIndent === -1) blockIndent = ind;
    collected.push(line.slice(blockIndent));
    cur.i++;
  }
  while (collected.length && collected[collected.length - 1] === '') collected.pop();
  return fold ? collected.join(' ').replace(/\s+/g, ' ').trim() : collected.join('\n');
}

/** Parse a mapping whose entries all sit at exactly `indent`. */
function parseMap(cur: Cursor, indent: number): Record<string, YamlValue> {
  const obj: Record<string, YamlValue> = {};
  while (cur.i < cur.lines.length) {
    const line = cur.lines[cur.i]!;
    if (isSkippable(line)) {
      cur.i++;
      continue;
    }
    const ind = indentOf(line);
    if (ind < indent) break;
    if (ind > indent) break; // defensive: unexpected deeper line, let caller handle
    const content = line.slice(indent);
    const colon = findKeyColon(content);
    if (colon === -1) break; // not a mapping line (e.g. a stray sequence)
    const key = content.slice(0, colon).trim();
    const after = content.slice(colon + 1).trim();
    cur.i++;
    obj[key] = parseValueAfterKey(cur, indent, after);
  }
  return obj;
}

/** Locate the `:` that separates a mapping key from its value, ignoring colons
 *  inside a quoted key. Returns -1 when the line is not `key: ...`. */
function findKeyColon(content: string): number {
  const quote = content[0];
  if (quote === '"' || quote === "'") {
    const close = content.indexOf(quote, 1);
    if (close > 0 && content[close + 1] === ':') return close + 1;
  }
  const colon = content.indexOf(':');
  if (colon === -1) return -1;
  // Must be `key:` or `key: value` (colon followed by space or EOL).
  const next = content[colon + 1];
  if (next === undefined || next === ' ') return colon;
  return -1;
}

function parseValueAfterKey(cur: Cursor, indent: number, after: string): YamlValue {
  if (after === '|' || after === '|-' || after === '|+')
    return parseBlockScalar(cur, indent, false);
  if (after === '>' || after === '>-' || after === '>+') return parseBlockScalar(cur, indent, true);
  if (after !== '') return coerceScalar(after);
  // Nothing after the colon: a nested block (map or sequence) or an empty value.
  const child = nextStructuralIndent(cur);
  if (child === null || child <= indent) return null;
  const first = cur.lines[cur.i]!.slice(child);
  if (first.startsWith('- ') || first === '-') return parseSeq(cur, child);
  return parseMap(cur, child);
}

/** Parse a block sequence whose `-` items all sit at exactly `indent`. */
function parseSeq(cur: Cursor, indent: number): YamlValue[] {
  const arr: YamlValue[] = [];
  while (cur.i < cur.lines.length) {
    const line = cur.lines[cur.i]!;
    if (isSkippable(line)) {
      cur.i++;
      continue;
    }
    const ind = indentOf(line);
    if (ind < indent) break;
    if (ind > indent) break;
    const content = line.slice(indent);
    if (content[0] !== '-') break;
    const rest = content.slice(1).replace(/^ /, '');
    cur.i++;
    // `- key: value` starts an inline map item; the dash occupies 2 columns.
    if (rest !== '' && findKeyColon(rest) !== -1) {
      const itemIndent = indent + 2;
      const obj: Record<string, YamlValue> = {};
      const colon = findKeyColon(rest);
      const key = rest.slice(0, colon).trim();
      const after = rest.slice(colon + 1).trim();
      obj[key] = parseValueAfterKey(cur, itemIndent, after);
      // Remaining keys of the same map item sit at itemIndent.
      Object.assign(obj, parseMap(cur, itemIndent));
      arr.push(obj);
    } else if (rest === '') {
      const child = nextStructuralIndent(cur);
      if (child !== null && child > indent) {
        const f = cur.lines[cur.i]!.slice(child);
        arr.push(f.startsWith('- ') || f === '-' ? parseSeq(cur, child) : parseMap(cur, child));
      } else {
        arr.push(null);
      }
    } else {
      arr.push(coerceScalar(rest));
    }
  }
  return arr;
}

function nextStructuralIndent(cur: Cursor): number | null {
  for (let j = cur.i; j < cur.lines.length; j++) {
    if (!isSkippable(cur.lines[j]!)) {
      cur.i = j;
      return indentOf(cur.lines[j]!);
    }
  }
  cur.i = cur.lines.length;
  return null;
}

/** Parse the YAML subset used by agent frontmatter into a plain object. */
export function parseYamlSubset(yaml: string): Record<string, YamlValue> {
  const lines = yaml.replace(/\r\n/g, '\n').split('\n');
  const cur: Cursor = { lines, i: 0 };
  return parseMap(cur, 0);
}

export function parseAgentFile(src: string): ParsedAgentFile {
  const { fm, body, hasFrontmatter } = splitFrontmatter(src);
  const frontmatter = hasFrontmatter ? parseYamlSubset(fm) : {};
  return { frontmatter, body, hasFrontmatter };
}

// ---- domain summary: parsed frontmatter -> typed AgentConfig display fields ----

export interface ToolEntry {
  /** Raw token, e.g. `read_file` or `@terminal`. */
  name: string;
  /** True for an `@namespace` group token. */
  namespace: boolean;
}

export interface PrefetchEntry {
  tool?: string;
  assign?: string;
}

export interface AgentSummary {
  name?: string;
  description?: string;
  model?: string;
  extends?: string;
  effort?: string;
  maxTurns?: number;
  visibility?: string;
  spawnable?: boolean;
  strictTools?: boolean;
  disableHistory?: boolean;
  autoContext?: boolean | null;
  runIf?: string;
  tools: ToolEntry[];
  /** Plain-string attachment paths (string form). */
  attachments: string[];
  /** Count of dict-form AttachmentSpec entries (assign/mode/index). */
  attachmentSpecs: number;
  autoLoadSkills: string[];
  autoLoadAgents: string[];
  autoLoadAgentList: boolean;
  skillPaths: string[];
  allowedSecrets: string[];
  prefetch: PrefetchEntry[];
  sandbox?: Record<string, YamlValue>;
  network?: Record<string, YamlValue>;
  instructions?: string;
  /** Frontmatter keys present but not surfaced as a dedicated field. */
  extraKeys: string[];
}

const SURFACED_KEYS = new Set([
  'name',
  'description',
  'model',
  'extends',
  'reasoning_effort',
  'max_turns',
  'visibility',
  'spawnable',
  'strict_tools',
  'disable_history',
  'auto_context',
  'run_if',
  'tools',
  'attachments',
  'auto_load_skills',
  'auto_load_agents',
  'auto_load_agent_list',
  'skill_paths',
  'allowed_secrets',
  'prefetch',
  'sandbox',
  'network',
  'instructions',
]);

function asString(v: YamlValue | undefined): string | undefined {
  if (v == null) return undefined;
  if (typeof v === 'string') return v;
  if (typeof v === 'number' || typeof v === 'boolean') return String(v);
  return undefined;
}

function asStringList(v: YamlValue | undefined): string[] {
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

function asBool(v: YamlValue | undefined): boolean | undefined {
  return typeof v === 'boolean' ? v : undefined;
}

function asRecord(v: YamlValue | undefined): Record<string, YamlValue> | undefined {
  return v && typeof v === 'object' && !Array.isArray(v)
    ? (v as Record<string, YamlValue>)
    : undefined;
}

/** Reduce parsed frontmatter to the typed fields the Form tab renders. */
export function summarizeAgent(fm: Record<string, YamlValue>): AgentSummary {
  const attachmentsRaw = Array.isArray(fm.attachments) ? fm.attachments : [];
  const attachments = attachmentsRaw.filter((x): x is string => typeof x === 'string');
  const attachmentSpecs = attachmentsRaw.length - attachments.length;

  const prefetchRaw = Array.isArray(fm.prefetch) ? fm.prefetch : [];
  const prefetch: PrefetchEntry[] = prefetchRaw.map((p) => {
    const rec = asRecord(p);
    return { tool: asString(rec?.tool), assign: asString(rec?.assign) };
  });

  const maxTurnsN = typeof fm.max_turns === 'number' ? fm.max_turns : undefined;

  return {
    name: asString(fm.name),
    description: asString(fm.description),
    model: asString(fm.model),
    extends: asString(fm.extends),
    effort: asString(fm.reasoning_effort),
    maxTurns: maxTurnsN,
    visibility: asString(fm.visibility),
    spawnable: asBool(fm.spawnable),
    strictTools: asBool(fm.strict_tools),
    disableHistory: asBool(fm.disable_history),
    autoContext: fm.auto_context === null ? null : asBool(fm.auto_context),
    runIf: asString(fm.run_if),
    tools: asStringList(fm.tools).map((name) => ({ name, namespace: name.startsWith('@') })),
    attachments,
    attachmentSpecs,
    autoLoadSkills: asStringList(fm.auto_load_skills),
    autoLoadAgents: asStringList(fm.auto_load_agents),
    autoLoadAgentList: asBool(fm.auto_load_agent_list) ?? false,
    skillPaths: asStringList(fm.skill_paths),
    allowedSecrets: asStringList(fm.allowed_secrets),
    prefetch,
    sandbox: asRecord(fm.sandbox),
    network: asRecord(fm.network),
    instructions: asString(fm.instructions),
    extraKeys: Object.keys(fm).filter((k) => !SURFACED_KEYS.has(k)),
  };
}
