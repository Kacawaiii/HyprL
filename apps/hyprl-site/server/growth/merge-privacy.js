'use strict';
// Preserve operator-controlled publisher/provider facts; add only growth sections.
const fs = require('node:fs');
function mergePrivacy(live, proposed) {
  const headings = ['Lettre hebdomadaire gratuite', 'Parrainage Discord'];
  const sections = headings.map(heading => {
    const marker = '<h2>' + heading + '</h2>';
    const start = proposed.indexOf(marker), end = proposed.indexOf('<h2>', start + marker.length);
    if (start < 0 || end < 0) throw Error('Missing growth privacy section');
    return proposed.slice(start, end);
  });
  for (const heading of headings) {
    const marker = '<h2>' + heading + '</h2>';
    const start = live.indexOf(marker);
    if (start >= 0) {
      const end = live.indexOf('<h2>', start + marker.length);
      if (end < 0) throw Error('Unbounded existing growth privacy section');
      live = live.slice(0, start) + live.slice(end);
    }
  }
  const anchor = '<h2>Espace membre et connexion Discord</h2>';
  if (live.split(anchor).length !== 2) throw Error('Live privacy structure needs operator review');
  return live.replace(anchor, sections.join('') + anchor);
}
if (require.main === module) {
  const [live, staged] = process.argv.slice(2);
  if (!live || !staged || live === staged) throw Error('Distinct live and staged privacy files required');
  const value = mergePrivacy(fs.readFileSync(live, 'utf8'), fs.readFileSync(staged, 'utf8'));
  fs.writeFileSync(staged, value);
  console.log('Growth privacy sections merged; live publisher/provider facts retained.');
}
module.exports = { mergePrivacy };
