const BASE_TITLE = 'Tsugite';

export function pageTitle(needsYouCount: number): string {
  return needsYouCount > 0 ? `(${needsYouCount}) ${BASE_TITLE}` : BASE_TITLE;
}
