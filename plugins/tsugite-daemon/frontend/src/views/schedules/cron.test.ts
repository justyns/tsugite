import { describe, it, expect } from 'vitest';
import { describeCron } from './cron';

describe('describeCron', () => {
  it('describes the reference cron expressions', () => {
    expect(describeCron('*/15 * * * *')).toBe('every 15 min');
    expect(describeCron('0 * * * *')).toBe('hourly');
    expect(describeCron('0 3 * * *')).toBe('daily 03:00');
    expect(describeCron('0 9 * * 1-5')).toBe('weekdays 09:00');
    expect(describeCron('0 4 * * 0')).toBe('sundays 04:00');
    expect(describeCron('30 7 * * *')).toBe('daily 07:30');
  });

  it('handles @macros case-insensitively', () => {
    expect(describeCron('@monthly')).toBe('monthly');
    expect(describeCron('@hourly')).toBe('hourly');
    expect(describeCron('@daily')).toBe('daily 00:00');
    expect(describeCron('@midnight')).toBe('daily 00:00');
    expect(describeCron('@weekly')).toBe('weekly');
    expect(describeCron('@yearly')).toBe('yearly');
    expect(describeCron('@ANNUALLY')).toBe('yearly');
    expect(describeCron('@reboot')).toBe('on daemon start');
  });

  it('covers step and interval shapes', () => {
    expect(describeCron('* * * * *')).toBe('every minute');
    expect(describeCron('*/1 * * * *')).toBe('every minute');
    expect(describeCron('*/5 * * * *')).toBe('every 5 min');
    expect(describeCron('0 */2 * * *')).toBe('every 2 hours');
    expect(describeCron('15 * * * *')).toBe('hourly at :15');
  });

  it('names single weekdays and weekend/weekday sets', () => {
    expect(describeCron('0 9 * * 1')).toBe('mondays 09:00');
    expect(describeCron('0 9 * * 7')).toBe('sundays 09:00'); // 7 aliases sunday
    expect(describeCron('0 9 * * 6,0')).toBe('weekends 09:00');
  });

  it('describes day-of-month schedules', () => {
    expect(describeCron('0 0 1 * *')).toBe('1st of the month 00:00');
    expect(describeCron('30 6 15 * *')).toBe('15th of the month 06:30');
    expect(describeCron('0 0 22 * *')).toBe('22nd of the month 00:00');
  });

  it('returns null for unrecognized / malformed input so the caller shows the raw expr', () => {
    expect(describeCron('')).toBeNull();
    expect(describeCron(null)).toBeNull();
    expect(describeCron('   ')).toBeNull();
    expect(describeCron('0 3 * *')).toBeNull(); // only 4 fields
    expect(describeCron('bogus')).toBeNull();
    expect(describeCron('@never')).toBeNull();
    expect(describeCron('0 9 1 * 1')).toBeNull(); // dom AND dow set - ambiguous
    expect(describeCron('0 9 * 6 *')).toBeNull(); // specific month, not modeled
    expect(describeCron('99 99 * * *')).toBeNull(); // out-of-range time
  });
});
