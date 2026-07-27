import { describe, expect, test } from 'vitest';
import { formatElapsed } from './format';

describe('formatElapsed', () => {
  test('renders zero as 00:00', () => {
    expect(formatElapsed(0)).toBe('00:00');
  });

  test('pads single-digit seconds', () => {
    expect(formatElapsed(7)).toBe('00:07');
  });

  test('rolls seconds into minutes', () => {
    expect(formatElapsed(65)).toBe('01:05');
  });

  test('floors fractional seconds rather than rounding', () => {
    expect(formatElapsed(7.9)).toBe('00:07');
  });

  test('does not clamp minutes to two digits - a run can go past 99 minutes', () => {
    expect(formatElapsed(3661)).toBe('61:01');
  });

  test('clamps a negative elapsed (clock skew) to 00:00 instead of going negative', () => {
    expect(formatElapsed(-5)).toBe('00:00');
  });
});
