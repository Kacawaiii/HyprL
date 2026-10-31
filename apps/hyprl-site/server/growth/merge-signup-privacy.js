'use strict';
// Update only signup behavior, preserving live publisher, processor and member facts.
const fs = require('node:fs');
const HEADINGS = ['Données, finalité et base légale', 'Lettre hebdomadaire gratuite', 'Durée de conservation et effacement', 'Sécurité et données techniques'];
function section(html, heading) {
  const marker = '<h2>' + heading + '</h2>', start = html.indexOf(marker);
  if (start < 0 || html.split(marker).length !== 2) throw Error('Signup privacy structure needs operator review');
  let end = html.indexOf('<h2', start + marker.length);
  if (end < 0) end = html.indexOf('\n      </div>', start);
  if (end < 0) throw Error('Unbounded signup privacy section');
  return { start, end, text: html.slice(start, end) };
}
function mergeSignupPrivacy(live, proposed) {
  for (const heading of HEADINGS) {
    const old = section(live, heading), next = section(proposed, heading);
    live = live.slice(0, old.start) + next.text + live.slice(old.end);
  }
  return live;
}
if (require.main === module) {
  const [live, staged] = process.argv.slice(2);
  if (!live || !staged || live === staged) throw Error('Distinct live and staged privacy files required');
  fs.writeFileSync(staged, mergeSignupPrivacy(fs.readFileSync(live, 'utf8'), fs.readFileSync(staged, 'utf8')));
  console.log('Signup privacy behavior updated; live publisher/provider/member facts retained.');
}
module.exports = { mergeSignupPrivacy };
