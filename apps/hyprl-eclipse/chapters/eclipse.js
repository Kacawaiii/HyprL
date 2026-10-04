import * as THREE from 'three';
import { CSS3DObject } from '../vendor/CSS3DRenderer.js';
import { NOISE, VERT_UV, glow, rng, rockGeometry, rockMaterial, starField, halfSize, portable, portableInstances } from '../lib/kit.js';

/**
 * Chapitre 01 — Éclipse (références : éclipse + anneau + astéroïdes, couronne en filaments,
 * faisceau et cartes de verre). The hero copy sits inside the black disc; the corona frames it.
 */
export function createEclipseChapter({ isMobile }) {
  const scene = new THREE.Scene(); scene.name = 'Chapitre_01_Eclipse';
  const camera = new THREE.PerspectiveCamera(43, 1, .1, 140); camera.position.set(0, 0, 14);
  const labelCamera = camera.clone();
  const labels = new THREE.Scene();
  const accent = new THREE.Color('#dac09a'), rimWarm = new THREE.Color('#e9dccb'), streakTint = new THREE.Color('#c9ccd4');
  const world = new THREE.Group(); world.name = 'HYPRL_Eclipse'; scene.add(world);

  scene.add(new THREE.HemisphereLight(0xc5d6f4, 0x08090b, 1.3));
  const key = new THREE.PointLight(0xeddbc1, 55, 40, 2); key.position.set(0, 2.5, 6); scene.add(key);
  const fill = new THREE.PointLight(0x8caed8, 22, 25, 2); fill.position.set(-6, -1, 5); scene.add(fill);

  // Deep-space nebula: blue-grey filaments, warmed around the eclipse.
  const nebulaMaterial = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uColor: { value: accent.clone() } },
    vertexShader: VERT_UV, depthWrite: false,
    fragmentShader: NOISE + /* glsl */`varying vec2 vUv;uniform float uTime;uniform vec3 uColor;
      void main(){vec2 p=(vUv-.5)*vec2(3.2,2.);float n=fbm(p*1.3+vec2(uTime*.006,0.));float n2=fbm(p*3.4+n*2.4-vec2(0.,uTime*.004));
        float r=length(p*vec2(.8,1.));
        vec3 col=vec3(.0015,.002,.0035)+vec3(.012,.016,.028)*pow(n2,2.4)*smoothstep(1.9,.2,r);
        col+=uColor*exp(-r*2.6)*.035;
        gl_FragColor=vec4(col,1.);}`
  });
  const nebula = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), nebulaMaterial); nebula.name = 'Nebula'; nebula.renderOrder = -10; world.add(nebula);

  const stars = starField({ count: isMobile() ? 180 : 420, spread: [70, 46], depth: [-14, -34], seed: 27 }); world.add(stars);

  // ── Eclipse group ───────────────────────────────────────────────
  const eclipse = new THREE.Group(); eclipse.name = 'Eclipse'; world.add(eclipse);

  const coronaMaterial = glow(NOISE + /* glsl */`varying vec2 vUv;uniform float uTime,uScale,uSide;uniform vec3 uColor,uRim;
    void main(){vec2 p=(vUv-.5)*uScale;float r=length(p);float a=atan(p.y,p.x);float d=max(r-1.,0.);
      vec2 cs=vec2(cos(a),sin(a));
      float w1=fbm(cs*2.2+vec2(d*1.4-uTime*.035,d*.6));
      float w2=fbm(cs*5.5+vec2(w1*1.8,0.)+vec2(d*2.6-uTime*.05,-d*1.3));
      float fil=pow(w2,4.2)*3.4+pow(w1,5.)*1.2;
      float fall=exp(-d*2.4),glowR=exp(-d*6.),hot=exp(-d*26.);
      float side=.55+.45*cos(a-uSide);
      vec3 col=uColor*(fil*fall*.85+glowR*.22)*(.45+.75*side)+uRim*hot*(1.4+1.6*side)+vec3(1.)*hot*hot*.9*side;
      col*=smoothstep(.985,1.004,r)*smoothstep(uScale*.5,uScale*.36,r);
      gl_FragColor=vec4(col,1.);}`,
    { uTime: { value: 0 }, uScale: { value: 5 }, uSide: { value: .75 }, uColor: { value: new THREE.Color('#c4c7cd') }, uRim: { value: rimWarm.clone() } });
  const corona = new THREE.Mesh(new THREE.PlaneGeometry(5, 5), coronaMaterial); corona.name = 'Corona'; corona.position.z = -.6; eclipse.add(corona);

  const sphereMaterial = new THREE.ShaderMaterial({
    uniforms: { uRim: { value: rimWarm.clone() }, uColor: { value: accent.clone() }, uSide: { value: .75 } },
    vertexShader: /* glsl */`varying vec3 vN;varying vec3 vV;varying vec3 vP;void main(){vec4 mv=modelViewMatrix*vec4(position,1.);vN=normalize(normalMatrix*normal);vV=normalize(-mv.xyz);vP=position;gl_Position=projectionMatrix*mv;}`,
    fragmentShader: NOISE + /* glsl */`varying vec3 vN;varying vec3 vV;varying vec3 vP;uniform vec3 uRim,uColor;uniform float uSide;
      void main(){vec3 n=normalize(vN);float facing=clamp(dot(n,normalize(vV)),0.,1.);float rim=pow(1.-facing,6.);
        float side=.5+.5*dot(normalize(n.xy+1e-4),vec2(cos(uSide),sin(uSide)));
        float surf=fbm(vP.xy*2.6+vP.z*1.7)*.5+fbm(vP.yz*6.)*.5;
        vec3 body=vec3(.0016,.0019,.0028)+vec3(.006,.008,.013)*surf*pow(1.-facing,1.5);
        gl_FragColor=vec4(body+mix(uColor,uRim,.55)*rim*(.25+1.4*side*side),1.);}`
  });
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 64), sphereMaterial); sphere.name = 'EclipseBody'; eclipse.add(sphere);

  // Orbital ring (tilted, occluded behind the body by depth).
  const ringPlane = new THREE.Group(); ringPlane.name = 'OrbitalPlane'; ringPlane.rotation.set(1.3, 0, -.12); eclipse.add(ringPlane);
  const ringMaterial = glow(NOISE + /* glsl */`varying vec2 vUv;varying vec3 vP;uniform vec3 uColor;uniform float uTime,uWidth;
    void main(){float a=atan(vP.y,vP.x);float front=.35+.65*smoothstep(.6,-1.,sin(a));
      float spark=.75+.5*vnoise(vec2(a*40.+uTime*.3,0.));
      float x=abs(length(vP.xy)-1.)/uWidth;float line=exp(-x*x*3.)+exp(-x*.6)*.12;
      gl_FragColor=vec4(uColor*line*front*spark*1.25,1.);}`,
    { uColor: { value: new THREE.Color('#d9dfe9') }, uTime: { value: 0 }, uWidth: { value: .004 } },
    { side: THREE.DoubleSide, vertexShader: /* glsl */`varying vec2 vUv;varying vec3 vP;void main(){vUv=uv;vP=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}` });
  const ring = new THREE.Mesh(new THREE.RingGeometry(.95, 1.05, 360, 1), ringMaterial); ring.name = 'OrbitalRing'; ringPlane.add(ring);

  // Belt of asteroids following the ring.
  const rockMat = rockMaterial({ rim: accent.clone(), fill: new THREE.Color('#8caed8') });
  const rockGeos = [rockGeometry(1, 2), rockGeometry(7, 2), rockGeometry(13, 1)];
  const r = rng(91), beltCount = isMobile() ? 48 : 96, belt = [], beltMeshes = [];
  for (let k = 0; k < 3; k++) { const m = new THREE.InstancedMesh(rockGeos[k], rockMat, Math.ceil(beltCount / 3)); m.name = `AsteroidBelt_${k}`; m.frustumCulled = false; ringPlane.add(m); beltMeshes.push(m); }
  for (let i = 0; i < beltCount; i++) {
    const k = i % 3, a = r() * Math.PI * 2, spread = (r() - .5);
    belt.push({ mesh: beltMeshes[k], index: Math.floor(i / 3), a, f: 1 + spread * .16 + (r() - .5) * .04, h: (r() - .5) * .05, s: .005 + Math.pow(r(), 3.5) * .022, axis: new THREE.Vector3(r() - .5, r() - .5, r() - .5).normalize(), spin: (r() - .5) * .8, speed: .008 + r() * .01, phase: r() * 6 });
  }
  // Foreground debris placed in screen space (sx, sy ∈ [-1, 1]) so it never covers the copy.
  const fgSpecs = [
    { sx: -1.02, sy: .34, z: 5.2, s: .12, big: true }, { sx: -.66, sy: -.1, z: 3, s: .045 }, { sx: -.78, sy: .7, z: 1.5, s: .035 }, { sx: -.95, sy: -.55, z: 4, s: .06 },
    { sx: .86, sy: .55, z: 4.2, s: .07 }, { sx: .7, sy: .2, z: 2.5, s: .03 }, { sx: .93, sy: -.2, z: 1, s: .045 }, { sx: .62, sy: .78, z: 0, s: .025 },
    { sx: -.5, sy: .86, z: -1, s: .02 }, { sx: .45, sy: .9, z: 2, s: .02 }, { sx: -.35, sy: -.92, z: 6, s: .03 }, { sx: .3, sy: -.95, z: 6.5, s: .035 }
  ];
  const fg = new THREE.Group(); fg.name = 'ForegroundDebris'; world.add(fg);
  const fgRocks = fgSpecs.map((spec, i) => {
    const m = new THREE.Mesh(spec.big ? rockGeometry(41, 3) : rockGeos[i % 3], rockMat); m.name = spec.big ? 'MonolithRock' : `Debris_${i}`;
    m.rotation.set(r() * 6, r() * 6, r() * 6); fg.add(m);
    return { m, spec, axis: new THREE.Vector3(r() - .5, r() - .5, r() - .5).normalize(), spin: (r() - .5) * .25 };
  });

  // Diamond-ring flare + anamorphic streak (follows the pointer around the limb).
  const flareMaterial = glow(/* glsl */`varying vec2 vUv;uniform vec3 uColor;uniform float uPulse;
    void main(){vec2 p=(vUv-.5)*2.;float d=length(p);float core=exp(-d*d*260.)*3.2;float halo=exp(-d*7.)*.55;
      float sx=exp(-abs(p.y)*90.)*exp(-abs(p.x)*1.8);float sy=exp(-abs(p.x)*110.)*exp(-abs(p.y)*3.2);
      vec2 q=mat2(.7071,-.7071,.7071,.7071)*p;float sd=(exp(-abs(q.y)*150.)*exp(-abs(q.x)*7.)+exp(-abs(q.x)*150.)*exp(-abs(q.y)*7.))*.45;
      vec3 col=uColor*(halo+sx*.9+sy*.6+sd)*uPulse+vec3(1.,.98,.95)*core;
      gl_FragColor=vec4(col*smoothstep(1.,.75,d),1.);}`, { uColor: { value: accent.clone() }, uPulse: { value: 1 } }, { depthTest: false });
  const flare = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), flareMaterial); flare.name = 'DiamondFlare'; flare.renderOrder = 5; eclipse.add(flare);
  const streakMaterial = glow(/* glsl */`varying vec2 vUv;uniform vec3 uColor;void main(){vec2 p=(vUv-.5)*2.;float s=exp(-abs(p.y)*30.)*exp(-abs(p.x)*1.9);gl_FragColor=vec4(uColor*s*.38,1.);}`,
    { uColor: { value: streakTint.clone() } }, { depthTest: false });
  const streak = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), streakMaterial); streak.name = 'AnamorphicStreak'; streak.renderOrder = 4; eclipse.add(streak);

  // Vertical light beam into the eclipse (landing reference).
  const beamMaterial = glow(/* glsl */`varying vec2 vUv;uniform vec3 uColor;uniform float uTime;
    void main(){float x=abs(vUv.x-.5);float core=exp(-x*260.);float inner=exp(-x*46.)*.22;float outer=exp(-x*9.)*.03;
      float fade=smoothstep(0.,.18,vUv.y)*(1.-smoothstep(.7,1.,vUv.y));float pulse=.92+.08*sin(uTime*.45);
      gl_FragColor=vec4(uColor*(core*1.4+inner+outer)*fade*pulse,1.);}`, { uColor: { value: accent.clone() }, uTime: { value: 0 } });
  const beam = new THREE.Mesh(new THREE.PlaneGeometry(3.6, 1), beamMaterial); beam.name = 'LightBeam'; beam.renderOrder = 2; world.add(beam);

  // ── Glass panels (WebGL) + readable HTML overlays (CSS3D) ──────
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

  // ── Layout (in hero pixels, so the copy always sits inside the disc) ──
  const layout = { w: 1, h: 1, center: new THREE.Vector2(), R: 1, flareAngle: .75 };
  function pxToWorld(px, py, z) { const hs = halfSize(camera, z); return new THREE.Vector2((px / layout.w - .5) * 2 * hs.w, (.5 - py / layout.h) * 2 * hs.h); }
  function resize(w, heroH) {
    layout.w = w; layout.h = heroH; const mobile = w < 600;
    for (const c of [camera, labelCamera]) { c.aspect = w / heroH; c.updateProjectionMatrix(); }
    const zE = -6, cy = mobile ? Math.min(268, heroH * .34) : Math.min(305, heroH * .31);
    const Rpx = mobile ? Math.min(w * .47, 205) : Math.max(220, Math.min(w * .205, 300));
    const c = pxToWorld(w / 2, cy, zE), unit = halfSize(camera, zE).h * 2 / heroH;
    layout.center.copy(c); layout.R = Rpx * unit;
    eclipse.position.set(c.x, c.y, zE); eclipse.scale.setScalar(layout.R);
    const ringPx = mobile ? w * .62 : Math.min(w * .44, Rpx * 2.55);
    ring.scale.setScalar(ringPx / Rpx); beltMeshes.forEach(m => m.scale.setScalar(ringPx / Rpx));
    ringMaterial.uniforms.uWidth.value = .0045 * Rpx / ringPx * 2.4;
    corona.scale.setScalar(1); coronaMaterial.uniforms.uScale.value = 5;
    const hsN = halfSize(camera, -30); nebula.position.set(c.x * 2.2, c.y * 2.2, -30); nebula.scale.set(hsN.w * 2.6, hsN.h * 2.6, 1);
    // Beam from above the frame down to the limb.
    const top = halfSize(camera, -6.3).h; beam.position.set(c.x, (c.y + layout.R + top + 2) / 2 + .2, -6.3); beam.scale.set(mobile ? .7 : 1, top + 2 - c.y - layout.R, 1);
    streak.scale.set(mobile ? 9 : 16, .7, 1);
    for (const f of fgRocks) {
      const hs = halfSize(camera, f.spec.z), sx = mobile ? Math.sign(f.spec.sx) * Math.max(Math.abs(f.spec.sx), .8) : f.spec.sx;
      f.base = new THREE.Vector3(sx * hs.w, f.spec.sy * hs.h, f.spec.z); f.m.position.copy(f.base); f.m.scale.setScalar(f.spec.s * hs.h * 2 * (mobile ? .8 : 1));
    }
    positionPanels(mobile);
  }
  function positionPanels(mobile) {
    const scale = mobile ? .0073 : .0085;
    for (const { spec, group, label } of panels) {
      group.scale.setScalar(scale / .01); group.position.set(spec.x * (mobile ? .69 : 1), spec.y + (mobile ? .06 : 0), spec.z); group.rotation.set(-.05, spec.ry, spec.rz);
      label.scale.setScalar(scale); label.position.copy(group.position); label.rotation.copy(group.rotation); label.position.z += .1;
      group.visible = label.visible = !mobile || spec.name === 'CenterPanel';
    }
  }

  const tmpQ = new THREE.Quaternion(), tmpM = new THREE.Matrix4(), tmpV = new THREE.Vector3(), tmpS = new THREE.Vector3();
  function update({ time, pointer, motion, scroll, viewH, pr }) {
    // Camera: pointer parallax; scroll through the hero by view offset (exact sync with the HTML cards).
    for (const c of [camera, labelCamera]) { c.position.set(pointer.x * .55 * motion, -pointer.y * .32 * motion, 14); c.lookAt(0, 0, 0); }
    camera.setViewOffset(layout.w, layout.h, 0, scroll, layout.w, viewH);
    eclipse.position.y = layout.center.y - scroll * .0045;
    stars.material.uniforms.uTime.value = time; stars.material.uniforms.uPR.value = pr;
    nebulaMaterial.uniforms.uTime.value = time; coronaMaterial.uniforms.uTime.value = time; beamMaterial.uniforms.uTime.value = time; ringMaterial.uniforms.uTime.value = time;
    // The diamond follows the pointer around the limb.
    const len = Math.hypot(pointer.x, pointer.y), target = len > .12 && motion ? Math.atan2(-pointer.y, pointer.x) : .75 + Math.sin(time * .07) * .2;
    let d = target - layout.flareAngle; d = Math.atan2(Math.sin(d), Math.cos(d)); layout.flareAngle += d * .04;
    const fa = layout.flareAngle; coronaMaterial.uniforms.uSide.value = fa; sphereMaterial.uniforms.uSide.value = fa;
    flare.position.set(Math.cos(fa) * 1.01, Math.sin(fa) * 1.01, .3); flare.scale.setScalar(1.25); streak.position.copy(flare.position); streak.position.z = .25;
    flareMaterial.uniforms.uPulse.value = .9 + .1 * Math.sin(time * 1.3);
    // Belt orbit + spin.
    for (const b of belt) {
      const a = b.a + time * b.speed; tmpV.set(Math.cos(a) * b.f, Math.sin(a) * b.f, b.h + Math.sin(time * .2 + b.phase) * .006);
      tmpQ.setFromAxisAngle(b.axis, time * b.spin + b.phase); tmpS.setScalar(b.s); b.mesh.setMatrixAt(b.index, tmpM.compose(tmpV, tmpQ, tmpS));
    }
    beltMeshes.forEach(m => { m.instanceMatrix.needsUpdate = true; });
    eclipse.updateMatrixWorld(); rockMat.uniforms.uLight.value.setFromMatrixPosition(eclipse.matrixWorld);
    for (const f of fgRocks) { if (!f.base) continue; f.m.rotateOnAxis(f.axis, f.spin * .016 * motion); f.m.position.set(f.base.x - pointer.x * (f.spec.z + 6) * .05 * motion, f.base.y + pointer.y * (f.spec.z + 6) * .03 * motion + Math.sin(time * .3 + f.spec.sx * 4) * .05, f.base.z); }
    for (const { spec, group, label } of panels) {
      group.position.y = spec.y + (layout.w < 600 ? .06 : 0) + Math.sin(time * .38 + spec.x) * .07; group.rotation.z = spec.rz + Math.sin(time * .23 + spec.x) * .008;
      label.position.copy(group.position); label.position.z += .1; label.rotation.copy(group.rotation);
    }
  }
  function setPalette(color, name) {
    accent.copy(color); const rim = name === 'ice' ? new THREE.Color('#d4e6f5') : new THREE.Color('#e9dccb');
    for (const m of [beamMaterial, flareMaterial, sphereMaterial, nebulaMaterial]) m.uniforms.uColor.value.copy(color);
    coronaMaterial.uniforms.uRim.value.copy(rim); sphereMaterial.uniforms.uRim.value.copy(rim);
    coronaMaterial.uniforms.uColor.value.set(name === 'ice' ? '#bccbdb' : '#c4c7cd');
    rockMat.uniforms.uRim.value.copy(color); key.color.copy(color);
  }
  function exportGroup() {
    world.updateMatrixWorld(true); const g = new THREE.Group(); g.name = 'Chapitre_01_Eclipse';
    g.add(portable(sphere, { color: 0x05060a, roughness: .8, metalness: .2 }));
    g.add(portable(ring, { color: 0xd9dfe9, emissive: 0xd9dfe9, emissiveIntensity: .6, side: THREE.DoubleSide }));
    beltMeshes.forEach((m, i) => g.add(portableInstances(m, { color: 0x2a2c31, roughness: .95, flatShading: true }, `AsteroidBelt_${i}`)));
    fgRocks.forEach(f => g.add(portable(f.m, { color: 0x2a2c31, roughness: .95, flatShading: true })));
    for (const { group } of panels) { const c = new THREE.Group(); c.name = group.name; c.applyMatrix4(group.matrixWorld); c.add(new THREE.Mesh(group.children[0].geometry, new THREE.MeshPhysicalMaterial({ color: 0x1c2230, metalness: .45, roughness: .22, transparent: true, opacity: .7, clearcoat: 1 }))); g.add(c); }
    return g;
  }
  return { name: 'eclipse', scene, camera, labels, labelCamera, resize, update, setPalette, exportGroup, post: { ca: .0012, bloom: .8, exposure: 1.05, sat: .3 } };
}
