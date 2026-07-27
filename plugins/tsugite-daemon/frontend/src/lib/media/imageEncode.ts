/**
 * Client-side photo re-encode applied before upload. Downscales the longest edge
 * to `maxEdge` and re-encodes as JPEG at `quality`. This trims phone-photo
 * bandwidth, keeps images under the API's per-image cap, and normalizes formats
 * the canvas can't emit (HEIC) into JPEG. SVG/GIF and non-images pass through
 * untouched (rasterizing a vector or flattening an animation would destroy it).
 * The daemon's upload-size backstop still applies to whatever we hand it.
 */

export interface ImageEncodeConfig {
  maxEdge: number;
  quality: number;
}

export const DEFAULT_IMAGE_CONFIG: ImageEncodeConfig = { maxEdge: 1568, quality: 0.85 };

// Raster image types the canvas can safely re-encode. Vector (svg) and animated
// (gif) images are deliberately excluded so they pass through as-is.
const REENCODABLE = /^image\/(jpeg|png|webp|bmp|tiff|heic|heif|avif)$/i;

export async function reencodeImage(file: File, config: ImageEncodeConfig): Promise<File> {
  if (!REENCODABLE.test(file.type)) return file;

  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    // Undecodable in this browser (e.g. HEIC on a non-Safari engine): leave the
    // original so the daemon's size backstop / path-hint fallback can handle it.
    return file;
  }

  const longest = Math.max(bitmap.width, bitmap.height);
  const scale = longest > config.maxEdge ? config.maxEdge / longest : 1;
  const w = Math.round(bitmap.width * scale);
  const h = Math.round(bitmap.height * scale);

  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    bitmap.close();
    return file;
  }
  // Flatten any alpha onto white — JPEG has no transparency, and unfilled pixels
  // would otherwise turn black.
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(bitmap, 0, 0, w, h);
  bitmap.close();

  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob(resolve, 'image/jpeg', config.quality),
  );
  if (!blob) return file;

  const name = file.name.replace(/\.[^./\\]+$/, '') + '.jpg';
  return new File([blob], name, { type: 'image/jpeg' });
}
