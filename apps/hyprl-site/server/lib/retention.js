'use strict';

// Calendar months, clamped to the last day of the destination month (UTC).
function expiresAt(createdAt) {
  const date = new Date(createdAt);
  if (typeof createdAt !== 'string' || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?Z$/.test(createdAt) ||
      !Number.isFinite(date.getTime()) || date.toISOString().replace('.000Z', 'Z') !== createdAt.replace('.000Z', 'Z')) {
    throw new Error('Invalid subscriber timestamp');
  }
  const target = new Date(date);
  target.setUTCDate(1);
  target.setUTCFullYear(date.getUTCFullYear() + 2);
  const lastDay = new Date(Date.UTC(target.getUTCFullYear(), target.getUTCMonth() + 1, 0)).getUTCDate();
  target.setUTCDate(Math.min(date.getUTCDate(), lastDay));
  return target.toISOString();
}

function startRetentionJob(store, { now = Date.now, logger = console, schedule = setInterval, cancel = clearInterval } = {}) {
  // Fail startup if the store cannot be purged; never quietly retain overdue records.
  store.purgeExpiredSync(now());
  const run = () => store.purgeExpired(now());
  const timer = schedule(() => {
    run().catch(() => logger.error('[retention] purge failed; operator action required'));
  }, 60 * 60 * 1000);
  timer.unref?.();
  return { run, close: () => cancel(timer) };
}
module.exports = { expiresAt, startRetentionJob };
