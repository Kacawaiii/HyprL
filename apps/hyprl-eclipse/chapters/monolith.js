import * as THREE from 'three';
import { BUMP, NOISE, VERT_SCREEN, noise2, rng, portable } from '../lib/kit.js';
import { createHeroCards } from '../lib/cards.js';

/**
 * Chapitre 02 — Monolithe (référence : monolithe lointain sur une plaine sombre, arête enneigée à droite avec un pic
 * aigu, collines basses dans la brume à gauche, soleil étoilé rasant à gauche, anneau planétaire géant).
 * Composition of the reference: very low horizon, a small distant monolith for scale, heavy aerial perspective.
 * Monochrome: the only colour is a faint tint of the active ambiance on the sun.
 * First chapter: it carries the hero's readable glass cards (CSS3D labels, their own camera).
 */
const PITCH = THREE.MathUtils.degToRad(12);   // camera looks up: the horizon sits low in the frame

export function createMonolithChapter({ isMobile }) {
  const scene = new THREE.Scene(); scene.name = 'Chapitre_02_Monolithe';
  const camera = new THREE.PerspectiveCamera(34, 1, .5, 4000);
  const sun = new THREE.Color('#f2e6d2');
  const sunDir = new THREE.Vector3(-.4, .11, -.91).normalize();
  // The land is lit from the left as in the reference (side light), the haze scatters toward the drawn sun.
  const landLight = new THREE.Vector3(-.9, .14, -.12).normalize();
  const state = { sun: new THREE.Vector2(.07, .34), arcC: new THREE.Vector2(), arcR: 1, aspect: 1, horizon: .15 };

  // Sky (screen space): dark zenith, bright haze band at the horizon, star-shaped low sun, giant ring arc.
  const skyMaterial = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uAspect: { value: 1 }, uSun: { value: state.sun }, uTint: { value: sun.clone() }, uArcC: { value: state.arcC }, uArcR: { value: 1 }, uArcEnd: { value: 0 }, uShift: { value: new THREE.Vector2() }, uHorizon: { value: .15 } },
    vertexShader: VERT_SCREEN, depthWrite: false, depthTest: false,
    fragmentShader: NOISE + /* glsl */`varying vec2 vUv;uniform float uTime,uAspect,uArcR,uArcEnd,uHorizon;uniform vec2 uSun,uArcC,uShift;uniform vec3 uTint;
      void main(){vec2 q=vUv+uShift;vec2 asp=vec2(uAspect,1.);float y=q.y,above=max(y-uHorizon,0.);
        vec3 top=vec3(.0055,.006,.0075),mid=vec3(.016,.017,.021),hor=vec3(.1,.106,.122);
        vec3 col=mix(hor,mid,smoothstep(0.,.3,above));col=mix(col,top,smoothstep(.22,.85,above));
        col+=vec3(.075,.079,.09)*exp(-above*17.)*(.5+.5*exp(-abs(q.x-.45)*2.6));
        vec2 d=(q-uSun)*asp;float r=length(d);
        col+=uTint*(exp(-r*3.2)*.07+exp(-r*7.)*.34+exp(-r*r*700.)*1.2);
        float sx=exp(-abs(d.y)*200.)*exp(-abs(d.x)*12.),sy=exp(-abs(d.x)*230.)*exp(-abs(d.y)*16.);
        vec2 dr=mat2(.7071,-.7071,.7071,.7071)*d;float sd=exp(-abs(dr.y)*420.)*exp(-abs(dr.x)*16.)+exp(-abs(dr.x)*420.)*exp(-abs(dr.y)*16.);
        vec2 de=mat2(.9239,-.3827,.3827,.9239)*d;float se=exp(-abs(de.y)*700.)*exp(-abs(de.x)*34.)+exp(-abs(de.x)*700.)*exp(-abs(de.y)*34.);
        col+=vec3(1.)*exp(-r*r*7000.)*5.+uTint*(sx*.75+sy*.55+sd*.14+se*.06);
        col+=uTint*exp(-above*7.)*exp(-abs(q.x-uSun.x)*2.6)*.14;
        // Ring: thin bright outer limb, soft lit band on its inner side; it fades into the haze at the horizon.
        vec2 pa=(q-.5)*asp;float s=length(pa-uArcC)-uArcR;
        float ring=exp(-(s*s)/(1.1e-5))*.85+(s<0.?exp(s/.007)*.14:exp(-s*80.)*.04);
        ring*=smoothstep(uHorizon+.004,uHorizon+.09,y)*smoothstep(uArcEnd+.035,uArcEnd-.01,pa.x)*(.72+.28*smoothstep(uHorizon,1.,y));
        col+=vec3(.8,.84,.9)*ring;
        float h=fbm(q*vec2(2.2,6.)+vec2(uTime*.006,0.));col+=vec3(.035,.037,.042)*h*h*smoothstep(.3,0.,above);
        gl_FragColor=vec4(col,1.);}`
  });
  const sky = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), skyMaterial); sky.name = 'Sky'; sky.frustumCulled = false; sky.renderOrder = -10; scene.add(sky);

  // Land: one shader for plain, ridge and hills. Smooth normals + screen-resolved relief, snow on gentle slopes,
  // light from the low sun, and aerial perspective toward a haze that brightens near the horizon and the sun.
  const HAZE = /* glsl */`uniform vec3 uSunDir,uTint;
    vec3 haze(vec3 v){vec3 c=mix(vec3(.028,.031,.038),vec3(.1,.106,.122),smoothstep(.14,-.01,v.y));
      vec2 a=normalize(v.xz+1e-5),b=normalize(uSunDir.xz);return c+uTint*pow(max(dot(a,b),0.),5.)*.16*smoothstep(.22,0.,v.y);}`;
  const landMaterial = (snowAmount, albedoLow, fogK) => new THREE.ShaderMaterial({
    uniforms: { uSunDir: { value: sunDir }, uLight: { value: landLight }, uTint: { value: sun.clone() }, uSnow: { value: snowAmount }, uLow: { value: albedoLow }, uFogK: { value: fogK } },
    vertexShader: /* glsl */`varying vec3 vW;varying vec3 vN;void main(){vec4 w=modelMatrix*vec4(position,1.);vW=w.xyz;vN=normalize(mat3(modelMatrix)*normal);gl_Position=projectionMatrix*viewMatrix*w;}`,
    fragmentShader: NOISE + BUMP + HAZE + /* glsl */`varying vec3 vW;varying vec3 vN;uniform vec3 uLight;uniform float uSnow,uLow,uFogK;
      void main(){vec3 g=normalize(vN);float px=length(fwidth(vW));
        float rel=fbm(vW.xz*.11)*.9+fbm(vW.xz*.45+3.7)*.3*smoothstep(2.5,.6,px);vec3 n=bumpN(g,vW,rel*smoothstep(6.,1.2,px));
        float ndl=dot(n,uLight);float lit=pow(clamp(ndl*1.35+.02,0.,1.),1.3);
        float snow=uSnow*smoothstep(.3,.62,g.y+(fbm(vW.xz*.25)-.5)*.35)*smoothstep(1.5,7.,vW.y);
        vec3 albedo=mix(vec3(uLow),vec3(.64,.66,.7),snow);
        vec3 col=albedo*(lit*1.15*uTint+.05)+vec3(.015)*max(-ndl,0.)*snow;
        vec3 v=vW-cameraPosition;float dist=length(v);vec3 hz=haze(v/dist);
        col=mix(col,hz,1.-exp(-dist*uFogK));
        col=mix(col,hz*1.05,exp(-max(vW.y,0.)*.22)*smoothstep(90.,420.,dist)*.35);
        gl_FragColor=vec4(col,1.);}`
  });
  const ss = THREE.MathUtils.smoothstep;
  const ridgedN = (x, z, oct = 5) => { let a = .5, f = 1, s = 0; for (let i = 0; i < oct; i++) { const n = 1 - Math.abs(noise2(x * f + 11.3, z * f - 4.1) * 2 - 1); s += n * n * a; a *= .5; f *= 2.03; } return s; };
  function patch(name, x0, x1, z0, z1, sx, sz, height, material) {
    const g = new THREE.PlaneGeometry(x1 - x0, z1 - z0, sx, sz); g.rotateX(-Math.PI / 2); g.translate((x0 + x1) / 2, 0, (z0 + z1) / 2);
    const p = g.attributes.position;
    for (let i = 0; i < p.count; i++) p.setY(i, height(p.getX(i), p.getZ(i)));
    g.computeVertexNormals();
    const mesh = new THREE.Mesh(g, material); mesh.name = name; scene.add(mesh); return mesh;
  }
  const res = isMobile() ? .5 : 1;

  // Snowy ridge on the right: a long gentle rise from the valley, a sharp peak, then a high shoulder off-frame.
  const crestZ = x => -240 + 22 * Math.sin(x * .021);
  const crestH = x => (26 * ss(x, 4, 70) + 40 * Math.exp(x < 78 ? (x - 78) / 22 : -(x - 78) / 8) + 8 * ss(x, 86, 120)) * (1 - .18 * ss(x, 130, 250));   // long rise, sharp tip, steep right
  const ridge = patch('SnowRidge', 0, 280, -320, -150, Math.round(460 * res), Math.round(220 * res), (x, z) => {
    const H = crestH(x), u = (z - crestZ(x)) / (34 + 10 * noise2(x * .03, 7)), shape = Math.pow(Math.max(0, 1 - Math.abs(u)), 1.3);
    const gullies = ridgedN(x * .16, z * .05) * .55 + ridgedN(x * .5, z * .5, 3) * .25;   // erosion streaks down the face
    return H * shape * (.68 + .5 * gullies) + ridgedN(x * .07, z * .07, 3) * 3 * shape - 2;
  }, landMaterial(1, .05, .0036));

  // Low rounded hills on the left, two layers lost in the backlit haze.
  const hills = (x0, x1, z, H, width, seed) => (x, zz) => {
    const left = ss(-x, 14, 150), u = (zz - z - 18 * noise2(x * .012, seed)) / width;
    return (H * left * (.35 + .9 * noise2(x * .016, seed + 3) * noise2(x * .041, seed + 5)) + 3 * ridgedN(x * .05, zz * .05, 3)) * Math.max(0, 1 - u * u) - 1.5;
  };
  const hillsNear = patch('HillsNear', -340, 0, -360, -220, Math.round(260 * res), Math.round(90 * res), hills(-340, 0, -290, 22, 70, 2), landMaterial(.1, .03, .0058));
  const hillsFar = patch('HillsFar', -520, -40, -620, -420, Math.round(220 * res), Math.round(70 * res), hills(-520, -40, -520, 26, 95, 9), landMaterial(.1, .04, .0042));
  const rangeFar = patch('RangeFar', 60, 520, -640, -430, Math.round(200 * res), Math.round(70 * res), (x, z) => (30 * ss(x, 70, 160) + 10 * ridgedN(x * .04, z * .04, 4)) * Math.max(0, 1 - ((z + 535) / 95) ** 2) - 2, landMaterial(.8, .05, .003));

  // The plain: dark, nearly flat, slightly rolling.
  const plain = patch('Plain', -900, 900, -900, 60, Math.round(220 * res), Math.round(160 * res), (x, z) => (noise2(x * .02, z * .02) - .5) * 1.2 + (noise2(x * .1, z * .1) - .5) * .25, landMaterial(0, .009, .003));

  // The monolith: small and far, for scale. Slate, nearly black, a faint sheen on the lit edge.
  const monolithMaterial = new THREE.ShaderMaterial({
    uniforms: { uSunDir: { value: sunDir }, uTint: { value: sun.clone() } },
    vertexShader: /* glsl */`varying vec3 vW;varying vec3 vN;void main(){vec4 w=modelMatrix*vec4(position,1.);vW=w.xyz;vN=normalize(mat3(modelMatrix)*normal);gl_Position=projectionMatrix*viewMatrix*w;}`,
    fragmentShader: HAZE + /* glsl */`varying vec3 vW;varying vec3 vN;
      void main(){vec3 n=normalize(vN);vec3 v=vW-cameraPosition;float dist=length(v);vec3 e=-v/dist;
        float lit=max(dot(n,uSunDir),0.);float spec=pow(max(dot(reflect(-uSunDir,n),e),0.),24.);
        vec3 col=vec3(.012,.013,.016)+uTint*(lit*.05+spec*.12);
        col=mix(col,haze(v/dist),1.-exp(-dist*.0036));col+=vec3(.05)*exp(-max(vW.y,0.)*1.4);
        gl_FragColor=vec4(col,1.);}`
  });
  const monolith = new THREE.Mesh(new THREE.BoxGeometry(1.8, 4.9, .55), monolithMaterial); monolith.name = 'Monolith'; monolith.position.set(0, 2.4, -62); scene.add(monolith);

  // Ground mist around the monolith's base and along the valley.
  const mistMaterial = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uSeed: { value: 0 } },
    vertexShader: /* glsl */`varying vec2 vUv;varying vec3 vW;void main(){vUv=uv;vec4 w=modelMatrix*vec4(position,1.);vW=w.xyz;gl_Position=projectionMatrix*viewMatrix*w;}`,
    fragmentShader: NOISE + /* glsl */`varying vec2 vUv;varying vec3 vW;uniform float uTime,uSeed;
      void main(){float n=fbm(vW.xz*.035+vec2(uTime*.012+uSeed,uSeed));float edge=smoothstep(0.,.3,vUv.x)*smoothstep(1.,.7,vUv.x)*smoothstep(0.,.35,vUv.y)*smoothstep(1.,.55,vUv.y);
        gl_FragColor=vec4(vec3(.1,.106,.12),pow(n,2.)*.26*edge);}`,
    transparent: true, depthWrite: false
  });
  const mists = [];
  for (let i = 0; i < 4; i++) { const m = new THREE.Mesh(new THREE.PlaneGeometry(260, 120), mistMaterial.clone()); m.material.uniforms.uSeed.value = i * 3.7; m.name = `Mist_${i}`; m.rotation.x = -Math.PI / 2; m.position.set(0, .5 + i * .9, -70 - i * 30); scene.add(m); mists.push(m); }

  // A little dust in the light.
  const r = rng(5), dust = [];
  for (let i = 0; i < (isMobile() ? 80 : 180); i++) dust.push((r() - .5) * 40, r() * 8, -r() * 40);
  const dustGeo = new THREE.BufferGeometry(); dustGeo.setAttribute('position', new THREE.Float32BufferAttribute(dust, 3));
  const dustPts = new THREE.Points(dustGeo, new THREE.PointsMaterial({ color: 0x9a9da4, size: .025, transparent: true, opacity: .22, depthWrite: false, blending: THREE.AdditiveBlending })); dustPts.name = 'Dust'; scene.add(dustPts);

  const lands = [ridge, hillsNear, hillsFar, rangeFar, plain];
  // Hero cards: HTML overlays only (CSS3D), laid out for their own camera; the WebGL panels stay out of this scene.
  const labels = new THREE.Scene(), labelCamera = new THREE.PerspectiveCamera(43, 1, .1, 140); labelCamera.position.set(0, 0, 14);
  const cards = createHeroCards(new THREE.Group(), labels);
  const tmp = new THREE.Vector3();
  function resize(w, h, heroH = h) {
    state.w = w; labelCamera.aspect = w / heroH; labelCamera.updateProjectionMatrix(); cards.position(w < 600);
    camera.aspect = w / h; camera.fov = w / h < 1 ? 58 : 34; camera.updateProjectionMatrix(); state.aspect = w / h;
    skyMaterial.uniforms.uAspect.value = state.aspect;
    state.horizon = .5 - .5 * Math.tan(PITCH) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    skyMaterial.uniforms.uHorizon.value = state.horizon;
    // Ring: leaves the frame at the top-left, comes down to the horizon just left of the monolith.
    const A = new THREE.Vector2((.06 - .5) * state.aspect, .52), B = new THREE.Vector2((.47 - .5) * state.aspect, state.horizon - .5 + .07);
    const cx = .1 * state.aspect + .02, cy = ((cx - A.x) ** 2 - (cx - B.x) ** 2 + A.y ** 2 - B.y ** 2) / (2 * (A.y - B.y));
    state.arcC.set(cx, cy); state.arcR = Math.hypot(B.x - cx, B.y - cy); skyMaterial.uniforms.uArcR.value = state.arcR; skyMaterial.uniforms.uArcEnd.value = B.x;
  }
  function update({ time, pointer, motion, local }) {
    const dolly = local ?? .5, z = 16 - dolly * 7;
    camera.position.set(pointer.x * .5 * motion, 1.7 - pointer.y * .15 * motion, z);
    camera.lookAt(camera.position.x * .2, camera.position.y + Math.tan(PITCH) * 100, z - 100);
    camera.updateMatrixWorld();
    // The sun slides a little along the horizon with the pointer; the land is lit from where it is drawn.
    state.sun.set((state.aspect < 1 ? .14 : .07) + (pointer.x * .5 + .5) * .05 * motion, state.horizon + .2);
    sunDir.copy(tmp.set(state.sun.x * 2 - 1, state.sun.y * 2 - 1, .5).unproject(camera).sub(camera.position)).normalize();
    landLight.set(sunDir.x * 2.2, .14, -.12).normalize();
    skyMaterial.uniforms.uShift.value.set(-pointer.x * .004 * motion, pointer.y * .003 * motion);
    skyMaterial.uniforms.uTime.value = time;
    mists.forEach((m, i) => { m.material.uniforms.uTime.value = time; m.position.x = Math.sin(time * .02 + i) * 6; });
    dustPts.rotation.y = Math.sin(time * .02) * .05; dustPts.position.y = Math.sin(time * .15) * .1;
    labelCamera.position.set(pointer.x * .55 * motion, -pointer.y * .32 * motion, 14); labelCamera.lookAt(0, 0, 0); cards.update(time, state.w ?? 1440);
  }
  function setPalette(color, name) {
    sun.set(name === 'ice' ? '#dbeaff' : '#f2e6d2');
    for (const m of [skyMaterial, monolithMaterial, ...lands.map(l => l.material)]) m.uniforms.uTint.value.copy(sun);
  }
  function exportGroup() {
    const g = new THREE.Group(); g.name = 'Chapitre_02_Monolithe';
    g.add(portable(monolith, { color: 0x050506, roughness: .25, metalness: .6 }));
    for (const land of lands) g.add(portable(land, { color: land === plain ? 0x1a1b1e : 0x8a8c90, roughness: 1 }));
    return g;
  }
  return { name: 'monolith', scene, camera, labels, labelCamera, resize, update, setPalette, exportGroup, post: { ca: .0008, bloom: .45, exposure: 1, sat: .22 } };
}
