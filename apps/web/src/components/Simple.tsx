/**
 * Small building blocks of the Beginner pages: the one-sentence page intro, a list that shows five rows
 * and a "Voir plus" button, in-page tabs and a side drawer for details.
 */
import { useEffect, useId, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { createPortal } from 'react-dom';
import { NavLink, useSearchParams } from 'react-router-dom';
import { carrySelection } from '../lib/cockpit';

const INTRO_KEY = 'hyprl.intro.dismissed';

function dismissed(): string[] {
  try {
    const raw = window.localStorage.getItem(INTRO_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === 'string') : [];
  } catch {
    return [];
  }
}

/** The question a page answers, in one sentence. Closing it is remembered on this browser only. */
export function PageIntro({ id, question, children }: { id: string; question: string; children?: ReactNode }) {
  const [hidden, setHidden] = useState(() => dismissed().includes(id));
  if (hidden) return null;
  const close = () => {
    try {
      window.localStorage.setItem(INTRO_KEY, JSON.stringify([...dismissed(), id]));
    } catch {
      // A private window may refuse storage; the intro then just closes for this visit.
    }
    setHidden(true);
  };
  return (
    <aside className="page-intro" aria-label="À quoi sert cette page">
      <p><strong>{question}</strong>{children && <> {children}</>}</p>
      <button className="intro-close" onClick={close} aria-label="Masquer cette explication">×</button>
    </aside>
  );
}

/** Five rows, then "Voir plus" adds five more. */
export function ShowMore<T>({ items, render, step = 5, empty, label }: {
  items: T[]; render: (item: T, index: number) => ReactNode; step?: number; empty?: ReactNode; label: string;
}) {
  const [shown, setShown] = useState(step);
  if (items.length === 0) return <>{empty ?? null}</>;
  return (
    <>
      <ul className="simple-list" aria-label={label}>{items.slice(0, shown).map((item, i) => render(item, i))}</ul>
      {items.length > shown && (
        <button className="more" onClick={() => setShown((value) => value + step)}>
          Voir plus ({items.length - shown})
        </button>
      )}
    </>
  );
}

export interface Tab { id: string; label: string; content: () => ReactNode }

/** In-page tabs: only the selected panel is rendered, so a page stays one screen tall. */
export function Tabs({ tabs, label, initial }: { tabs: Tab[]; label: string; initial?: string }) {
  const [current, setCurrent] = useState(initial && tabs.some((t) => t.id === initial) ? initial : tabs[0]?.id);
  const base = useId();
  const active = tabs.find((t) => t.id === current) ?? tabs[0];
  const move = (offset: number) => {
    const index = tabs.findIndex((t) => t.id === active?.id);
    const next = tabs[(index + offset + tabs.length) % tabs.length];
    if (next) {
      setCurrent(next.id);
      document.getElementById(`${base}-${next.id}`)?.focus();
    }
  };
  if (!active) return null;
  return (
    <div className="tabs">
      <div role="tablist" aria-label={label} className="tablist"
        onKeyDown={(e) => { if (e.key === 'ArrowRight') move(1); if (e.key === 'ArrowLeft') move(-1); }}>
        {tabs.map((t) => (
          <button key={t.id} id={`${base}-${t.id}`} role="tab" aria-selected={t.id === active.id}
            aria-controls={`${base}-${t.id}-panel`} tabIndex={t.id === active.id ? 0 : -1}
            className="tab-button" onClick={() => setCurrent(t.id)}>{t.label}</button>
        ))}
      </div>
      <div role="tabpanel" id={`${base}-${active.id}-panel`} aria-labelledby={`${base}-${active.id}`} className="tabpanel">
        {active.content()}
      </div>
    </div>
  );
}

/** A side drawer: Escape or the close button dismisses it and focus returns to what opened it. */
export function Drawer({ title, onClose, children }: { title: ReactNode; onClose: () => void; children: ReactNode }) {
  const panel = useRef<HTMLDivElement>(null);
  const titleId = useId();
  useEffect(() => {
    const opener = document.activeElement as HTMLElement | null;
    panel.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('keydown', onKey);
      opener?.focus?.();
    };
  }, [onClose]);
  return createPortal(
    <div className="drawer-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="drawer" role="dialog" aria-modal="true" aria-labelledby={titleId} tabIndex={-1} ref={panel}>
        <header className="drawer-head">
          <h2 id={titleId}>{title}</h2>
          <button className="intro-close" onClick={onClose} aria-label="Fermer le détail">×</button>
        </header>
        <div className="drawer-body">{children}</div>
      </div>
    </div>,
    document.body,
  );
}

/** Tabs that are pages: Actualités (feed, official sources) and IA (decisions, does news help). */
export function PageTabs({ label, links }: { label: string; links: Array<{ to: string; label: string }> }) {
  const [params] = useSearchParams();
  const carry = carrySelection(params);
  return (
    <nav className="page-tabs" aria-label={label}>
      {links.map((link) => <NavLink key={link.to} to={{ pathname: link.to, search: carry }} end>{link.label}</NavLink>)}
    </nav>
  );
}

export const NEWS_TABS = [{ to: '/radar', label: 'Fil d’actualité' }, { to: '/events', label: 'Sources officielles (Fed, SEC)' }];
export const AI_TABS = [{ to: '/trader', label: 'Décisions des IA' }, { to: '/news-value', label: 'L’actualité aide ?' }];
