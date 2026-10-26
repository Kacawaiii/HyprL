import names from '../../../../configs/public_agent_names.json';

export const PUBLIC_AGENT_NAMES = names;
export const PUBLIC_METHOD = 'agents propriétaires HyprL construits sur des modèles de langage tiers';

/** Display only: records, model ids, selectors and keys retain their original values. */
export function publicAgentLabel(identity: string): string {
  const families: Record<string, keyof typeof names> = {
    opus: 'opus', claude: 'opus', analyst_claude: 'opus', reviewer_claude: 'opus',
    gpt: 'gpt', codex: 'gpt', analyst_gpt: 'gpt', reviewer_gpt: 'gpt',
  };
  const family = families[identity.toLowerCase()];
  return family ? names[family] : identity;
}

export const BOOK_LABEL = `${names.opus} book`;

/** Legacy snapshot labels are converted while rendering; the snapshot is never changed. */
export function publicBookLabel(label: string): string {
  return label.replace(/Claude book/g, BOOK_LABEL);
}
