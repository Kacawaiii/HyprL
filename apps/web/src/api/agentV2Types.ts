export interface ShadowLayer {
  closed: number; wins: number; pnl: number; expectancy_r: number | null; cluster_days: number; evidence: 'UNPROVEN';
}
export interface AgentV2Summary {
  schema: 'agent-v2-summary-v1'; mode: 'SYNTHETIC' | 'SHADOW_FORWARD'; asof: string;
  state: 'NO_QUOTES' | 'STALE' | 'SHADOW'; spec_hash: string; preregistration_hash: string; code_sha: string;
  ledger_hash: string; events: number; pending: number; broker_orders: 0;
  last_mark: { at: string; equity: number; cash: number; heat: number | null; governor: string } | null;
  positions: { symbol: string; layer: string; playbook: string; qty: number; entry: number; stop: number; initial_risk: number; cluster: string; filled_at: string }[];
  layers: { core: ShadowLayer; news: ShadowLayer };
  supervisor_vetoes: { count: number; codes: Record<string, number>; counterfactual: 'UNMEASURED' };
  risk_vetoes: Record<string, number>;
  last_blocked: { at: string; action: string; reason: string } | null;
  promotion: 'OPERATOR_ONLY_AFTER_REPLAY';
}
