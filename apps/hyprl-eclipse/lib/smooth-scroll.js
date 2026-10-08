/**
 * One damping stage for fine-pointer wheel input. The scene reads this rendered page position directly.
 * Touch, keyboard, focus, browser navigation and reduced motion retain immediate native behavior.
 */
export function createSmoothScroll({ damping = 12 } = {}) {
  const media = matchMedia('(prefers-reduced-motion: reduce)'), fine = matchMedia('(pointer: fine)');
  const root = document.documentElement, originalBehavior = root.style.scrollBehavior;
  let current = scrollY, target = scrollY, written = scrollY, previous = 0, tween = null, external = false, frame = 0;
  const enabled = () => !media.matches && fine.matches;
  const max = () => Math.max(0, root.scrollHeight - innerHeight);
  const clamp = y => Math.min(Math.max(y, 0), max());
  const blocked = el => document.querySelector('dialog[open]') || el?.closest?.('dialog, textarea, input, select, [contenteditable], [data-native-scroll]');
  // STANDARDS.md on-screen movement: cubic-bezier(0.77, 0, 0.175, 1).
  function ease(t) {
    if (t <= 0 || t >= 1) return t;
    let lo = 0, hi = 1;
    for (let i = 0; i < 18; i++) {
      const u = (lo + hi) / 2, v = 1 - u, x = 3 * v * v * u * .77 + 3 * v * u * u * .175 + u * u * u;
      if (x < t) lo = u; else hi = u;
    }
    const u = (lo + hi) / 2; return u * u * (3 - 2 * u);
  }
  function reset() { current = target = written = scrollY; tween = null; previous = 0; }
  function native() { if (Math.abs(scrollY - written) > 1) reset(); }
  function write(y) { written = y; window.scrollTo({ top: y, behavior: 'instant' }); }
  function wheel(e) {
    if (!enabled() || e.ctrlKey || e.defaultPrevented || blocked(e.target) || Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
    e.preventDefault(); native();
    if (tween) { target = current; tween = null; }
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? innerHeight : 1;
    target = clamp(target + e.deltaY * unit); wake();
  }
  function key(e) {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || blocked(e.target)) return;
    if (e.key === ' ' && e.target.closest?.('button, a, summary, [role="tab"]')) return;
    const step = { ArrowDown: 110, ArrowUp: -110, PageDown: innerHeight * .88, PageUp: -innerHeight * .88, ' ': innerHeight * .88 * (e.shiftKey ? -1 : 1) }[e.key];
    if (step === undefined && e.key !== 'Home' && e.key !== 'End') return;
    e.preventDefault(); reset();
    current = target = clamp(e.key === 'Home' ? 0 : e.key === 'End' ? max() : current + step); write(current);
  }
  function anchor(e) {
    const link = e.target.closest?.('a[href^="#"]');
    if (!link || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || blocked(e.target)) return;
    let id; try { id = decodeURIComponent(link.getAttribute('href').slice(1)); } catch { return; }
    const el = id ? document.getElementById(id) : null; if (id && !el) return;
    e.preventDefault();
    const offset = parseFloat(getComputedStyle(root).scrollPaddingTop) || 0;
    scrollTo(el ? el.getBoundingClientRect().top + scrollY - offset : 0, { immediate: e.detail === 0 || link.classList.contains('skip-link') });
    if (el) { if (!el.hasAttribute('tabindex')) el.tabIndex = -1; el.focus({ preventScroll: true }); }
    history.pushState(null, '', id ? `#${id}` : location.pathname + location.search);
  }
  function scrollTo(y, { immediate = false } = {}) {
    reset(); const to = clamp(y), distance = Math.abs(to - current); target = to;
    if (immediate || !enabled() || distance < 1) { current = to; write(to); return; }
    // Occasional narrative travel; wheel or keyboard retargets it from the current pose.
    tween = { from: current, to, start: null, duration: Math.min(900, Math.max(420, distance * .16)) }; wake();
  }
  function update(now) {
    native();
    if (!enabled()) { reset(); return false; }
    const dt = previous ? Math.max(0, Math.min((now - previous) / 1000, .2)) : 1 / 60; previous = now;
    if (tween) {
      tween.start ??= now;
      const t = Math.min((now - tween.start) / tween.duration, 1);
      current = tween.from + (tween.to - tween.from) * ease(t); if (t >= 1) tween = null;
    } else {
      current += (target - current) * (1 - Math.exp(-dt * damping)); if (Math.abs(target - current) < .25) current = target;
    }
    const y = Math.round(current);
    if (y !== scrollY) write(y);
    return tween !== null || Math.abs(current - target) > .01;
  }
  function loop(now) { frame = update(now) ? requestAnimationFrame(loop) : 0; if (!frame) previous = 0; }
  function wake() { if (!external && !frame) frame = requestAnimationFrame(loop); }
  function resize() { current = clamp(current); target = clamp(target); if (tween) tween.to = target; }
  function preference() { reset(); root.style.scrollBehavior = 'auto'; }
  preference(); media.addEventListener('change', preference); fine.addEventListener('change', preference);
  addEventListener('wheel', wheel, { passive: false }); addEventListener('keydown', key); document.addEventListener('click', anchor);
  addEventListener('scroll', native, { passive: true }); addEventListener('resize', resize);
  return {
    update, scrollTo,
    drive() { external = true; cancelAnimationFrame(frame); frame = 0; },
    destroy() {
      cancelAnimationFrame(frame); removeEventListener('wheel', wheel); removeEventListener('keydown', key);
      document.removeEventListener('click', anchor); removeEventListener('scroll', native); removeEventListener('resize', resize);
      media.removeEventListener('change', preference); fine.removeEventListener('change', preference); root.style.scrollBehavior = originalBehavior;
    },
  };
}
