<script lang="ts">
  import { ICONS, type IconName } from './icons';

  let {
    name,
    filled = false,
    size,
    color,
    class: cls = '',
  }: {
    name: IconName;
    filled?: boolean;
    /** Pixel size (width+height). Omit to inherit CSS `.ic` sizing (13px base, or a `.parent .ic` rule). */
    size?: number;
    /** One-off `currentColor` tint. Omit to inherit. */
    color?: string;
    class?: string;
  } = $props();

  const icon = $derived(ICONS[name]);
  const style = $derived(
    [size != null ? `width:${size}px;height:${size}px` : '', color ? `color:${color}` : '']
      .filter(Boolean)
      .join(';'),
  );
</script>

<svg
  class={['ic', filled && 'ic--f', cls]}
  viewBox={icon.viewBox}
  style={style || undefined}
  aria-hidden="true"
>
  {@html icon.body}
</svg>
