import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { MarketOverlayChart } from '../components/MarketOverlayChart';
import { TraderLifecycle } from '../components/TraderLifecycle';
import { analystLabel as radarLabel, rebasedCurves } from '../lib/radar';
import { analystLabel as traderLabel } from '../lib/trader';
import { PUBLIC_METHOD } from '../lib/publicAgentNames';
import { analysisSnapshot } from './phase2Fixtures';
import { paperSnapshot } from './radarFixtures';

describe('public agent names', () => {
  it('renders both analyst and reviewer roles without changing input identities', () => {
    const identities = ['analyst_claude', 'analyst_gpt', 'reviewer_claude', 'reviewer_gpt'];
    expect(identities.map(radarLabel)).toEqual(['HypR-A1', 'HypR-E1', 'Reviewer (HypR-A1)', 'Reviewer (HypR-E1)']);
    expect(identities.map(traderLabel)).toEqual(['HypR-A1 analyst', 'HypR-E1 analyst', 'Reviewed HypR-A1', 'Reviewed HypR-E1']);
    expect(identities).toEqual(['analyst_claude', 'analyst_gpt', 'reviewer_claude', 'reviewer_gpt']);
    expect(PUBLIC_METHOD).toBe('agents propriétaires HyprL construits sur des modèles de langage tiers');
  });

  it('renders legacy chart labels and accessible descriptions with public names', () => {
    const before = JSON.stringify(analysisSnapshot);
    const { container } = render(<MarketOverlayChart asset="AAAUSD" prices={analysisSnapshot.prices['AAA-USD'] ?? []}
      overlays={analysisSnapshot.overlays} selected={null} onSelect={() => {}} />);
    expect(container.textContent).toContain('HypR-A1');
    for (const element of container.querySelectorAll('[aria-label], title, figcaption')) {
      expect(`${element.getAttribute('aria-label') ?? ''} ${element.textContent}`).not.toMatch(/Claude|Opus|GPT|Codex|analyst_claude|analyst_gpt/);
    }
    expect(JSON.stringify(analysisSnapshot)).toBe(before);
  });

  it('keeps retirement and account identities while showing public names', () => {
    const roles = { analyst_gpt: { state: 'RETIRED' as const, effective_from: '2026-10-20', operator_decision_hash: 'synthetic' } };
    render(<TraderLifecycle roles={roles} />);
    expect(screen.getByLabelText('Retired trader roles')).toHaveTextContent('HypR-E1 analyst: RETIRED');
    expect(screen.getByLabelText('Retired trader roles').textContent).not.toMatch(/Claude|Opus|GPT|Codex/);
    const before = JSON.stringify(paperSnapshot);
    const book = rebasedCurves(paperSnapshot).find((s) => s.key === 'claude_book');
    expect(book?.label).toBe('HypR-A1 book');
    expect(JSON.stringify(paperSnapshot)).toBe(before);
  });
});
