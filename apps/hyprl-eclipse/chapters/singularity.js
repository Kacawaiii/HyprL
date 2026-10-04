import * as THREE from 'three';
import { NOISE, VERT_UV, glow, rng, starField, portable } from '../lib/kit.js';

/**
 * Chapitre 04 — Singularité (référence : trou noir doré, disque d'accrétion, vaisseau, vitesse).
 * The ship follows the pointer; warp streaks rush past.
 */
export function createSingularityChapter({ isMobile }) {
  const scene = new THREE.Scene(); scene.name = 'Chapitre_04_Singularite';
  const camera = new THREE.PerspectiveCamera(40, 1, .1, 200);
  const gold = new THREE.Color('#d8d2c6'), hot = new THREE.Color('#ffffff');
  const hole = new THREE.Group(); hole.name = 'BlackHole'; scene.add(hole);
  const state = { aspect: 1, shipPos: new THREE.Vector3(), shipTarget: new THREE.Vector3(), roll: 0 };

  scene.add(starField({ count: isMobile() ? 160 : 380, spread: [80, 50], depth: [-30, -60], seed: 77, color: '#e9d9bf' }));

  const discFrag = NOISE + /* glsl */`varying vec3 vP;uniform float uTime,uIn,uOut,uGain,uDop;uniform vec3 uGold,uHot;
    void main(){float r=length(vP.xy);float a=atan(vP.y,vP.x);float rn=clamp((r-uIn)/(uOut-uIn),0.,1.);
      float w=uTime*(1.1/(r*r*.25+.35));vec2 cs=vec2(cos(a-w),sin(a-w));
      float bands=fbm(vec2(r*7.,0.)+cs*1.4);float fine=fbm(vec2(r*24.,1.7)+cs*3.2);
      float dens=pow(bands,1.6)*1.5+pow(fine,2.)*.9;
      float inner=smoothstep(0.,.04,rn)*exp(-rn*3.2);
      float dop=1.+.75*cos(a-uDop);
      vec3 col=mix(uGold*vec3(.55,.32,.12),uGold,smoothstep(.8,.15,rn));col=mix(col,uHot,smoothstep(.25,0.,rn));
      col*=dens*inner*dop*uGain*smoothstep(1.,.75,rn);
      gl_FragColor=vec4(col,1.);}`;
  const discVert = /* glsl */`varying vec3 vP;void main(){vP=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`;
  const discUniforms = () => ({ uTime: { value: 0 }, uIn: { value: 1.75 }, uOut: { value: 8.5 }, uGain: { value: 1.55 }, uDop: { value: 3.14 }, uGold: { value: gold }, uHot: { value: hot } });
  const discMaterial = glow(discFrag, discUniforms(), { vertexShader: discVert, side: THREE.DoubleSide });
  const disc = new THREE.Mesh(new THREE.RingGeometry(1.75, 8.5, 256, 24), discMaterial); disc.name = 'AccretionDisc'; disc.rotation.set(-1.42, 0, .1); hole.add(disc);
  // Lensed image of the far side, wrapped over and under the horizon.
  const lensMaterial = glow(discFrag, { ...discUniforms(), uIn: { value: 1.62 }, uOut: { value: 5.2 }, uGain: { value: .85 }, uDop: { value: 1.57 } }, { vertexShader: discVert, side: THREE.DoubleSide });
  const lens = new THREE.Mesh(new THREE.RingGeometry(1.62, 5.2, 256, 16), lensMaterial); lens.name = 'LensedHalo'; lens.position.z = -.4; lens.rotation.z = .1; hole.add(lens);
  const horizon = new THREE.Mesh(new THREE.SphereGeometry(1.55, 96, 64), new THREE.MeshBasicMaterial({ color: 0x000000 })); horizon.name = 'EventHorizon'; hole.add(horizon);
  const photonMaterial = glow(/* glsl */`varying vec2 vUv;uniform vec3 uHot;void main(){float d=length(vUv-.5)*2.;float r=exp(-((d-.62)*60.)*((d-.62)*60.))*2.2+exp(-((d-.62)*9.)*((d-.62)*9.))*.25;gl_FragColor=vec4(uHot*r,1.);}`, { uHot: { value: hot } });
  const photon = new THREE.Mesh(new THREE.PlaneGeometry(5, 5), photonMaterial); photon.name = 'PhotonRing'; photon.position.z = .05; photon.renderOrder = 2; hole.add(photon);

  // Golden dust clouds.
  const cloudMaterial = (seed) => glow(NOISE + /* glsl */`varying vec2 vUv;uniform float uTime,uSeed;uniform vec3 uGold;
    void main(){vec2 p=vUv*2.-1.;float n=fbm(vUv*3.+vec2(uSeed,uTime*.01));float m=fbm(vUv*7.+n*2.);float e=smoothstep(1.,.2,length(p));
      gl_FragColor=vec4(uGold*pow(m,3.)*e*.5+vec3(.1)*pow(n,4.)*e,1.);}`, { uTime: { value: 0 }, uSeed: { value: seed }, uGold: { value: gold } });
  const clouds = [[-9, -1.5, -6, 12], [-6, 2.5, -14, 14], [8, -3, -8, 10], [2, -4.5, -3, 9]].map(([x, y, z, s], i) => { const m = new THREE.Mesh(new THREE.PlaneGeometry(s, s * .6), cloudMaterial(i * 4.1)); m.name = `GoldDust_${i}`; m.position.set(x, y, z); scene.add(m); return m; });

  // Warp streaks.
  const r = rng(808), streakN = isMobile() ? 70 : 150, sp = new Float32Array(streakN * 6), sc = new Float32Array(streakN * 6), streaks = [];
  for (let i = 0; i < streakN; i++) { const a = r() * Math.PI * 2, rad = 3 + r() * 13; streaks.push({ x: Math.cos(a) * rad, y: Math.sin(a) * rad * .7, z: -60 + r() * 75, v: 14 + r() * 26, len: .6 + r() * 2.2, b: .25 + r() * .75 }); }
  const streakGeo = new THREE.BufferGeometry(); streakGeo.setAttribute('position', new THREE.BufferAttribute(sp, 3)); streakGeo.setAttribute('color', new THREE.BufferAttribute(sc, 3));
  const streakLines = new THREE.LineSegments(streakGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); streakLines.name = 'WarpStreaks'; streakLines.frustumCulled = false; scene.add(streakLines);

  // The ship: sleek arrow hull, canopy, fins, twin engines.
  const ship = new THREE.Group(); ship.name = 'Vessel'; scene.add(ship);
  const hullShape = new THREE.Shape([[2.3, 0], [.7, .2], [-.3, 1.05], [-.85, 1.05], [-.55, .32], [-1.15, .3], [-1.15, -.3], [-.55, -.32], [-.85, -1.05], [-.3, -1.05], [.7, -.2]].map(([x, y]) => new THREE.Vector2(x, y)));
  const hullGeo = new THREE.ExtrudeGeometry(hullShape, { depth: .12, bevelEnabled: true, bevelThickness: .06, bevelSize: .05, bevelSegments: 2 }); hullGeo.center(); hullGeo.rotateX(-Math.PI / 2);
  const metal = new THREE.MeshStandardMaterial({ color: 0x4a4e57, metalness: .55, roughness: .32 });
  const hull = new THREE.Mesh(hullGeo, metal); hull.name = 'Hull'; ship.add(hull);
  const canopy = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), new THREE.MeshStandardMaterial({ color: 0x0b0d12, metalness: .9, roughness: .1 })); canopy.name = 'Canopy'; canopy.scale.set(.55, .13, .17); canopy.position.set(.55, .12, 0); ship.add(canopy);
  const finShape = new THREE.Shape([new THREE.Vector2(0, 0), new THREE.Vector2(.55, 0), new THREE.Vector2(.05, .42)]);
  for (const s of [-1, 1]) { const fin = new THREE.Mesh(new THREE.ExtrudeGeometry(finShape, { depth: .03, bevelEnabled: false }), metal); fin.name = `Fin_${s > 0 ? 'R' : 'L'}`; fin.position.set(-1.05, .05, s * .25); fin.rotation.x = s * .25; ship.add(fin); }
  const engineGeo = new THREE.CylinderGeometry(.11, .14, .5, 20); engineGeo.rotateZ(Math.PI / 2);
  const exhaustMaterial = glow(/* glsl */`varying vec2 vUv;uniform vec3 uHot,uGold;uniform float uPulse;void main(){vec2 p=vUv-vec2(1.,.5);float core=exp(-abs(p.y)*40.)*exp(p.x*3.2);float glowE=exp(-length(p*vec2(1.,3.))*6.);
    gl_FragColor=vec4((uHot*core*1.6+uGold*glowE)*uPulse,1.);}`, { uHot: { value: hot }, uGold: { value: gold }, uPulse: { value: 1 } }, { side: THREE.DoubleSide });
  for (const s of [-1, 1]) {
    const e = new THREE.Mesh(engineGeo, metal); e.name = `Engine_${s > 0 ? 'R' : 'L'}`; e.position.set(-1.1, 0, s * .2); ship.add(e);
    const trail = new THREE.Mesh(new THREE.PlaneGeometry(3.4, .5), exhaustMaterial); trail.name = `Exhaust_${s > 0 ? 'R' : 'L'}`; trail.position.set(-1.35 - 1.7, 0, s * .2); ship.add(trail);
    const trail2 = trail.clone(); trail2.rotation.x = Math.PI / 2; ship.add(trail2);
  }
  ship.add(new THREE.PointLight(0xe8e2d8, 4, 4, 2).translateX(-1.6));
  const sunKey = new THREE.PointLight(0xe9e2d6, 160, 30, 2); sunKey.position.set(0, .5, 1); scene.add(sunKey);
  scene.add(new THREE.HemisphereLight(0xd8d4cc, 0x0a0806, .7));

  function resize(w, h) { camera.aspect = w / h; camera.updateProjectionMatrix(); state.aspect = w / h; const mobile = w < 600; hole.scale.setScalar(mobile ? .62 : 1); ship.scale.setScalar(mobile ? .75 : 1.25); }
  const fwd = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0), mtx = new THREE.Matrix4(), qq = new THREE.Quaternion(), qr = new THREE.Quaternion();
  let prevT = 0;
  function update({ time, pointer, motion, local, dt }) {
    const lp = local ?? .4;
    camera.position.set(pointer.x * .7 * motion, .9 - pointer.y * .4 * motion, 17 + lp * 9); camera.lookAt(0, 3.4 - lp * 11 + (state.aspect < .8 ? 3.2 : 0), 0);
    hole.position.y = 0; for (const m of [discMaterial, lensMaterial]) m.uniforms.uTime.value = time;
    clouds.forEach((c, i) => { c.material.uniforms.uTime.value = time; c.lookAt(camera.position); c.position.x += Math.sin(time * .05 + i) * .002; });
    // Warp streaks toward the camera.
    for (let i = 0; i < streakN; i++) {
      const s = streaks[i]; s.z += s.v * (dt || 0) * motion; if (s.z > 16) s.z -= 76;
      const k = i * 6, L = s.len * (1 + s.v * .05); sp.set([s.x, s.y, s.z, s.x * 1.02, s.y * 1.02, s.z - L], k);
      const fadeIn = THREE.MathUtils.smoothstep(s.z, -60, -40) * s.b; sc.set([fadeIn * (.5 + gold.r * .5), fadeIn * (.35 + gold.g * .5), fadeIn * (.2 + gold.b * .5), 0, 0, 0], k);
    }
    streakGeo.attributes.position.needsUpdate = true; streakGeo.attributes.color.needsUpdate = true;
    // Ship: cruising loop, steered by the pointer.
    const u = ((time * .045) % 1 + 1) % 1, mobile = state.aspect < .8;
    const path = (t) => new THREE.Vector3(THREE.MathUtils.lerp(mobile ? -3.4 : -9, mobile ? 3.4 : 7, t), -3.2 + Math.sin(t * Math.PI) * 1.4, THREE.MathUtils.lerp(9.5, 1, t));
    state.shipTarget.copy(path(u)).add(new THREE.Vector3(pointer.x * 1.6 * motion, -pointer.y * 1.4 * motion, 0));
    const prev = state.shipPos.clone(); state.shipPos.lerp(state.shipTarget, time - prevT > .5 ? 1 : .08); prevT = time;
    ship.position.copy(state.shipPos);
    fwd.copy(path(Math.min(u + .02, 1))).sub(path(u)).normalize();
    mtx.lookAt(fwd, new THREE.Vector3(), up); qq.setFromRotationMatrix(mtx); qq.multiply(qr.setFromAxisAngle(up, -Math.PI / 2));
    const lateral = (state.shipPos.y - prev.y) * 18; state.roll += (THREE.MathUtils.clamp(-pointer.x * .5 * motion + lateral, -.6, .6) - state.roll) * .06;
    ship.quaternion.copy(qq).multiply(qr.setFromAxisAngle(new THREE.Vector3(1, 0, 0), state.roll));
    const fade = THREE.MathUtils.smoothstep(u, 0, .06) * (1 - THREE.MathUtils.smoothstep(u, .9, 1)); ship.visible = fade > .01;
    exhaustMaterial.uniforms.uPulse.value = (.85 + .15 * Math.sin(time * 30)) * fade;
  }
  function setPalette(color, name) { gold.set(name === 'ice' ? '#c6d6e8' : '#d8d2c6'); hot.set(name === 'ice' ? '#f4f8ff' : '#ffffff'); sunKey.color.set(name === 'ice' ? '#d9e6f2' : '#e9e2d6'); }
  function exportGroup() {
    const g = new THREE.Group(); g.name = 'Chapitre_04_Singularite'; scene.updateMatrixWorld(true);
    g.add(portable(horizon, { color: 0x000000, roughness: 1 }));
    g.add(portable(disc, { color: 0xd8d2c6, emissive: 0xd8d2c6, emissiveIntensity: 1, side: THREE.DoubleSide, transparent: true, opacity: .8 }));
    const s = new THREE.Group(); s.name = 'Vessel'; for (const c of ship.children) if (c.isMesh && c.material.isMeshStandardMaterial) s.add(portable(c, { color: 0x4a4e57, metalness: .55, roughness: .32 })); g.add(s);
    return g;
  }
  return { name: 'singularity', scene, camera, resize, update, setPalette, exportGroup, post: { ca: .002, bloom: 1, exposure: 1.05, sat: .25 } };
}
