import type { DecisionChain } from '../api/analysisTypes';
import { blankDecision } from '../lib/decisions';
import { host, instant, signedNumber, signedPct } from '../lib/radar';

/** The same six-step reading path in every decision view, including missing fields. */
export function DecisionCard({ decision, title }: { decision?: DecisionChain | null; title?: string }) {
  const d = decision ?? blankDecision();
  const result = d.result;
  return (
    <section className="decision-chain" aria-label={title ?? 'Chaîne de décision'}>
      {title && <h4 className="decision-title">{title}</h4>}
      <ol>
        <li>
          <h5>Fait sourcé <span className="verification" data-level={d.verification}>{d.verification}</span></h5>
          <p>{d.fact ?? 'Aucun fait sourcé consigné.'}</p>
          <p className="metric-sub">{d.verification === 'OFFICIEL'
            ? 'Source primaire identifiable citée. Le cockpit ne revérifie pas le contenu.'
            : d.verification === 'FIL' ? 'Présent dans le flux; aucune confirmation primaire identifiée.' : 'Aucune source identifiable.'}</p>
          {d.sources.length > 0 && <ul className="decision-sources">{d.sources.map((s, i) => (
            <li key={`${s.url}-${i}`}>
              {s.url ? <a href={s.url} target="_blank" rel="noreferrer noopener">{s.publisher ?? host(s.url)}</a> : s.publisher ?? 'Source inconnue'}
              {' '}<span className="verification" data-level={s.verification}>{s.verification}</span>
              <span className="metric-sub"> · publication {instant(s.published_at)}</span>
            </li>
          ))}</ul>}
        </li>
        <li><h5>Attentes du marché</h5><p>{d.expectations ?? 'Consensus daté non consigné.'}</p>
          <p className="metric-sub">Date du consensus : {d.expectation_at ? instant(d.expectation_at) : 'non consignée'}</p>
          <p>Déjà pricé : {d.priced_in ?? 'non mesuré / non consigné'}</p></li>
        <li><h5>Scénario</h5><p>{d.scenario ?? 'Scénario non consigné.'}</p></li>
        <li><h5>Condition d’entrée</h5><p>{d.entry_condition ?? 'Condition d’entrée non consignée.'}</p></li>
        <li><h5>Invalidation</h5><p>{d.invalidation ?? 'Ce qui ferait abandonner le modèle n’est pas consigné.'}</p></li>
        <li><h5>Résultat après coûts</h5>{result ? <>
          <p>{result.net_return !== null && <>Rendement net / unité : {signedPct(result.net_return)}. </>}
            {result.pnl_after_costs !== null && <>P&amp;L net : {signedNumber(result.pnl_after_costs)} USD. </>}
            {result.r_after_costs !== null && <>R net : {signedNumber(result.r_after_costs)}. </>}</p>
          <p className="metric-sub">{instant(result.at)} · {result.basis === 'modelled_roundtrip_cost' ? 'Coûts aller-retour modélisés; résultat de recherche.' : 'Résultat explicite du journal.'}</p>
        </> : <p>En attente : résultat réalisé après coûts non connu.</p>}</li>
      </ol>
    </section>
  );
}
