// Mirrors session_store.py's ALIAS_PATTERN, which the daemon applies with fullmatch.
const ALIAS_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;

export function isValidAlias(alias: string): boolean {
  return ALIAS_PATTERN.test(alias);
}

/** Leading dashes go before the truncation so the length budget is spent on content. */
export function suggestAlias(title: string): string {
  const slug = title
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+/, '')
    .slice(0, 64)
    .replace(/-+$/, '');
  return isValidAlias(slug) ? slug : '';
}
