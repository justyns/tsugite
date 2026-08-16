/** Lowest unused `Space N`, from 2 - the seeded first space is `Main`. A fresh
 *  space has no content to name it after. */
export function nextSpaceName(existing: string[]): string {
  const taken = new Set(existing.map((n) => n.trim()));
  for (let n = 2; ; n += 1) {
    const name = `Space ${n}`;
    if (!taken.has(name)) return name;
  }
}
