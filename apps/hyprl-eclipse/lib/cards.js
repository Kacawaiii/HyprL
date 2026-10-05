import * as THREE from 'three';
import { CSS3DObject } from '../vendor/CSS3DRenderer.js';

/**
 * Hero glass cards: WebGL glass panels added to `world` and their readable HTML overlays (CSS3D) added to
 * `labels`. Positions are in the hero camera's space (perspective 43°, at z = 14 looking at the origin).
 */
export function createHeroCards(world, labels) {
  function roundedShape(w, h, rr) { const s = new THREE.Shape(), x = -w / 2, y = -h / 2; s.moveTo(x + rr, y); s.lineTo(x + w - rr, y); s.quadraticCurveTo(x + w, y, x + w, y + rr); s.lineTo(x + w, y + h - rr); s.quadraticCurveTo(x + w, y + h, x + w - rr, y + h); s.lineTo(x + rr, y + h); s.quadraticCurveTo(x, y + h, x, y + h - rr); s.lineTo(x, y + rr); s.quadraticCurveTo(x, y, x + rr, y); return s; }
  const chart = `<svg class="card-chart" viewBox="0 0 210 90" fill="none"><path d="M0 20H210M0 45H210M0 70H210" stroke="#aabbcf" stroke-opacity=".1"/><path d="M0 78L13 72L23 76L35 58L45 62L58 41L70 48L86 31L100 39L116 20L130 29L145 14L160 25L177 10L190 18L210 5" stroke="#b6c9e2" stroke-width="1.3"/><path d="M0 85L16 80L34 82L48 74L67 78L89 67L113 70L138 59L158 63L185 51L210 48" stroke="#d1bd9c" stroke-opacity=".45"/></svg>`;
  const specs = [
    { name: 'LeftPanel', x: -5.1, y: -3.15, z: .6, ry: .25, rz: .075, width: 490, height: 282, html: `<div class="card-heading"><span><i class="card-symbol">⌁</i> Marchés</span><span class="card-tag">WATCHLIST</span></div><div class="card-subtitle">CRYPTO · FOREX · OR</div><div class="card-title">En perspective.</div><div class="card-meta">Votre univers, en un regard</div>${chart}<div class="card-footer"><span>VUE MULTI-ACTIFS<b>Une même interface</b></span><span>ESPACE<b>Personnalisé</b></span></div>` },
    { name: 'RightPanel', x: 5.1, y: -3.15, z: .6, ry: -.25, rz: -.075, width: 490, height: 282, html: `<div class="card-heading"><span><i class="card-symbol">◎</i> Gestion du risque</span><span class="card-tag">RISK VIEW</span></div><div class="card-subtitle">EXPOSITION · ALLOCATION</div><div class="card-title">Garder le contrôle.</div><div class="card-meta">Le détail fait la différence</div>${chart}<div class="card-footer"><span>PRIORITÉ<b>Votre exposition</b></span><span>APPROCHE<b>Structurée</b></span></div>` },
    { name: 'CenterPanel', x: 0, y: -2.55, z: 2, ry: 0, rz: 0, width: 530, height: 320, html: `<div class="card-heading"><span><i class="card-symbol">H</i> HYPRL Workspace</span><span class="card-tag">APERÇU</span></div><div class="card-subtitle">STRATEGY OVERVIEW</div><div class="card-title">La vue d'ensemble.</div><div class="card-meta">Des informations qui font sens</div>${chart}<div class="card-footer"><span>MARCHÉS<b>Crypto · Forex · Or</b></span><span>STRATÉGIES<b>Votre espace</b></span><span>FOCUS<b>Clarté & risque</b></span></div><div class="card-glare"></div>` }
  ];
  const panels = [];
  for (const spec of specs) {
    const group = new THREE.Group(); group.name = spec.name;
    const geometry = new THREE.ExtrudeGeometry(roundedShape(spec.width * .01, spec.height * .01, .22), { depth: .075, bevelEnabled: true, bevelSegments: 3, steps: 1, bevelSize: .025, bevelThickness: .02, curveSegments: 12 });
    const glass = new THREE.MeshPhysicalMaterial({ color: 0x1c2230, metalness: .45, roughness: .22, transparent: true, opacity: .7, clearcoat: 1, clearcoatRoughness: .1, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geometry, glass); mesh.name = `${spec.name}_Glass`; group.add(mesh);
    group.add(new THREE.LineSegments(new THREE.EdgesGeometry(geometry, 22), new THREE.LineBasicMaterial({ color: 0x93a9c6, transparent: true, opacity: .16 })));
    world.add(group);
    const element = document.createElement('div'); element.className = `scene-card ${spec.name === 'CenterPanel' ? 'center' : ''}`; element.innerHTML = spec.html;
    const label = new CSS3DObject(element); label.name = `${spec.name}_ReadableOverlay`; labels.add(label);
    panels.push({ spec, group, label });
  }

  function position(mobile) {
    const scale = mobile ? .0073 : .0085;
    for (const { spec, group, label } of panels) {
      group.scale.setScalar(scale / .01); group.position.set(spec.x * (mobile ? .69 : 1), spec.y + (mobile ? .06 : 0), spec.z); group.rotation.set(-.05, spec.ry, spec.rz);
      label.scale.setScalar(scale); label.position.copy(group.position); label.rotation.copy(group.rotation); label.position.z += .1;
      group.visible = label.visible = !mobile || spec.name === 'CenterPanel';
    }
  }
  function update(time, width) {
    for (const { spec, group, label } of panels) {
      group.position.y = spec.y + (width < 600 ? .06 : 0) + Math.sin(time * .38 + spec.x) * .07; group.rotation.z = spec.rz + Math.sin(time * .23 + spec.x) * .008;
      label.position.copy(group.position); label.position.z += .1; label.rotation.copy(group.rotation);
    }
  }
  function exportTo(g) {
    for (const { group } of panels) { const c = new THREE.Group(); c.name = group.name; c.applyMatrix4(group.matrixWorld); c.add(new THREE.Mesh(group.children[0].geometry, new THREE.MeshPhysicalMaterial({ color: 0x1c2230, metalness: .45, roughness: .22, transparent: true, opacity: .7, clearcoat: 1 }))); g.add(c); }
  }
  return { position, update, exportTo };
}
