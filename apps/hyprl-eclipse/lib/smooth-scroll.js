/**
 * Inertial page scroll for wheel and keyboard input (touch keeps its native momentum).
 * The page still scrolls natively — this only eases where it goes — so sticky sections, anchors,
 * find-in-page and the scrollbar keep working; any scroll it did not cause resynchronises it.
 *
 * update(now) advances one frame; the scene calls it before rendering so WebGL and the DOM move in
 * the same frame. Without a driver (no WebGL), it runs its own animation loop.
 */
export function createSmoothScroll({ damping = 7.5 } = {}) {
  if (matchMedia('(prefers-reduced-motion: reduce)').matches || !matchMedia('(pointer: fine)').matches) return null;
  const root = document.documentElement;
  root.style.scrollBehavior = 'auto';  // every frame sets the position itself
  let current = scrollY, target = scrollY, written = scrollY, previous = 0, tween = null, external = false, frame = 0;
  const max = () => Math.max(0, root.scrollHeight - innerHeight);
  const clamp = y => Math.min(Math.max(y, 0), max());
  const blocked = el => document.querySelector('dialog[open]') || el?.closest?.('dialog, textarea, input, select, [contenteditable], [data-native-scroll]');
  const ease = t => (t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

  function wheel(e) {
    if (e.ctrlKey || e.defaultPrevented || blocked(e.target)) return;
    e.preventDefault(); native();
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? innerHeight : 1;
    if (tween) { target = current; tween = null; }
    target = clamp(target + e.deltaY * unit);
    wake();
  }
  function key(e) {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || blocked(e.target)) return;
    if (e.key === ' ' && e.target.closest?.('button, a, summary, [role="tab"]')) return;
    const page = innerHeight * .88;
    const step = { ArrowDown: 110, ArrowUp: -110, PageDown: page, PageUp: -page, ' ': e.shiftKey ? -page : page }[e.key];
    if (e.key === 'Home' || e.key === 'End') { e.preventDefault(); scrollTo(e.key === 'Home' ? 0 : max()); return; }
    if (step === undefined) return;
    e.preventDefault(); native();
    if (tween) { target = current; tween = null; }
    target = clamp(target + step);
    wake();
  }
  function anchor(e) {
    const link = e.target.closest?.('a[href^="#"]');
    if (!link || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || link.classList.contains('skip-link')) return;
    const id = decodeURIComponent(link.getAttribute('href').slice(1));
    const el = id ? document.getElementById(id) : null;
    if (id && !el) return;
    e.preventDefault();
    const offset = parseFloat(getComputedStyle(root).scrollPaddingTop) || 0;
    scrollTo(el ? el.getBoundingClientRect().top + scrollY - offset : 0);
    history.pushState(null, '', id ? `#${id}` : location.pathname + location.search);
  }
  /** Eased travel to y, its duration growing with the distance. */
  function scrollTo(y) {
    native();
    const to = clamp(y), distance = Math.abs(to - current);
    tween = { from: current, to, start: performance.now(), duration: Math.min(1700, Math.max(700, distance * .45)) };
    target = to;
    wake();
  }
  function native() {
    // A scroll this module did not write (scrollbar, find-in-page, focus, touch): follow it. Input handlers
    // call it first too, since that scroll's event may only fire after their own input.
    if (Math.abs(scrollY - written) > 2) { current = target = written = scrollY; tween = null; }
  }

  function update(now) {
    const dt = previous ? Math.min((now - previous) / 1000, .05) : 1 / 60; previous = now;
    if (tween) {
      const t = Math.min((now - tween.start) / tween.duration, 1);
      current = tween.from + (tween.to - tween.from) * ease(t);
      if (t >= 1) tween = null;
    } else {
      current += (target - current) * (1 - Math.exp(-dt * damping));
      if (Math.abs(target - current) < .3) current = target;
    }
    const y = Math.round(current * 2) / 2;
    if (Math.abs(y - scrollY) >= .5) { written = y; window.scrollTo(0, y); }
    return tween !== null || current !== target;
  }
  function loop(now) { frame = update(now) ? requestAnimationFrame(loop) : 0; if (!frame) previous = 0; }
  function wake() { if (!external && !frame) frame = requestAnimationFrame(loop); }
  function resize() { target = clamp(target); }

  addEventListener('wheel', wheel, { passive: false });
  addEventListener('keydown', key);
  document.addEventListener('click', anchor);
  addEventListener('scroll', native, { passive: true });
  addEventListener('resize', resize);
  return {
    update,
    scrollTo,
    /** The caller drives update() from its own frame loop from now on. */
    drive() { external = true; if (frame) { cancelAnimationFrame(frame); frame = 0; } },
    destroy() {
      cancelAnimationFrame(frame); removeEventListener('wheel', wheel); removeEventListener('keydown', key);
      document.removeEventListener('click', anchor); removeEventListener('scroll', native); removeEventListener('resize', resize);
      root.style.scrollBehavior = '';
    },
  };
}
