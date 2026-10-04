import * as THREE from 'three';
import { NOISE, VERT_SCREEN, noise2, rng, portable } from '../lib/kit.js';

/**
 * Chapitre 02 — Monolithe (référence : monolithe, brume, anneau planétaire, soleil rasant).
 * Monochrome: the only colour is a faint tint of the active ambiance on the sun.
 */
export function createMonolithChapter({ isMobile }) {
  const scene = new THREE.Scene(); scene.name = 'Chapitre_02_Monolithe';
  const camera = new THREE.PerspectiveCamera(34, 1, .1, 400);
  const haze = new THREE.Color('#5d6066');
  const sun = new THREE.Color('#f2e6d2');
  const state = { sun: new THREE.Vector2(.08, .3), arcC: new THREE.Vector2(), arcR: 1, aspect: 1 };

  // Sky: gradient, giant ring arc, low sun with 4-point rays, drifting haze. Screen-space.
  const skyMaterial = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uAspect: { value: 1 }, uSun: { value: state.sun }, uTint: { value: sun.clone() }, uArcC: { value: state.arcC }, uArcR: { value: 1 }, uShift: { value: new THREE.Vector2() }, uHorizon: { value: .33 } },
    vertexShader: VERT_SCREEN, depthWrite: false, depthTest: false,
    fragmentShader: NOISE + /* glsl */`varying vec2 vUv;uniform float uTime,uAspect,uArcR,uHorizon;uniform vec2 uSun,uArcC,uShift;uniform vec3 uTint;
      void main(){vec2 q=vUv+uShift;vec2 asp=vec2(uAspect,1.);float y=q.y;
        vec3 top=vec3(.0035,.0038,.0046),mid=vec3(.028,.03,.034),hor=vec3(.2,.205,.215);
        vec3 col=mix(hor,mid,smoothstep(uHorizon-.02,uHorizon+.32,y));col=mix(col,top,smoothstep(uHorizon+.18,1.05,y));
        vec2 d=(q-uSun)*asp;float r=length(d);
        col+=uTint*(exp(-r*7.)*.7+exp(-r*2.2)*.2);
        float sx=exp(-abs(d.y)*260.)*exp(-abs(d.x)*3.2),sy=exp(-abs(d.x)*300.)*exp(-abs(d.y)*9.);
        vec2 dr=mat2(.7071,-.7071,.7071,.7071)*d;float sd=exp(-abs(dr.y)*400.)*exp(-abs(dr.x)*22.)+exp(-abs(dr.x)*400.)*exp(-abs(dr.y)*22.);
        col+=vec3(1.)*exp(-r*r*5200.)*4.+uTint*(sx*.9+sy*.55+sd*.5);
        vec2 pa=(q-.5)*asp;float dc=abs(length(pa-uArcC)-uArcR);
        float ang=atan(pa.y-uArcC.y,pa.x-uArcC.x);
        float arc=(exp(-dc*dc*4.5e5)*1.1+exp(-dc*140.)*.09)*smoothstep(uHorizon-.03,uHorizon+.12,y)*(.55+.45*smoothstep(-2.6,-1.6,ang));
        col+=vec3(.86,.88,.92)*arc;
        float h=fbm(q*vec2(2.4,7.)+vec2(uTime*.008,0.));col+=vec3(.06)*h*h*smoothstep(uHorizon+.25,uHorizon-.05,y);
        gl_FragColor=vec4(col,1.);}`
  });
  const sky = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), skyMaterial); sky.name = 'Sky'; sky.frustumCulled = false; sky.renderOrder = -10; scene.add(sky);

  // Terrain: valley in the centre, ranges left and right, one sharp peak (ref.).
  const seg = isMobile() ? [140, 90] : [260, 170];
  const terrainGeo = new THREE.PlaneGeometry(220, 170, seg[0], seg[1]); terrainGeo.rotateX(-Math.PI / 2);
  const p = terrainGeo.attributes.position;
  const ridged = (x, z) => { let a = .55, f = .045, s = 0; for (let i = 0; i < 6; i++) { const n = 1 - Math.abs(noise2(x * f + 11.3, z * f - 4.1) * 2 - 1); s += n * n * a; a *= .5; f *= 2.07; } return s; };
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i), z = p.getZ(i) - 45;
    const far = THREE.MathUtils.smoothstep(-z, 10, 34);
    const valley = THREE.MathUtils.smoothstep(Math.abs(x + 1.5) - Math.max(0, -z - 20) * .1, 5, 20);
    let h = ridged(x, z) * valley * far * (x > 0 ? 13 : 5.5);
    h += 15 * Math.exp(-((x - 21) ** 2 + (z + 46) ** 2) / 70) * (.75 + .5 * ridged(x * 2, z * 2));
    h += (noise2(x * .3, z * .3) - .5) * .35;
    p.setXYZ(i, x, h, z);
  }
  terrainGeo.computeVertexNormals();
  const landMaterial = new THREE.ShaderMaterial({
    uniforms: { uSunDir: { value: new THREE.Vector3(-.85, .22, -.45).normalize() }, uHaze: { value: haze }, uTint: { value: sun.clone() } },
    vertexShader: /* glsl */`varying vec3 vW;void main(){vec4 w=modelMatrix*vec4(position,1.);vW=w.xyz;gl_Position=projectionMatrix*viewMatrix*w;}`,
    fragmentShader: NOISE + /* glsl */`varying vec3 vW;uniform vec3 uSunDir,uHaze,uTint;
      void main(){vec3 n=normalize(cross(dFdx(vW),dFdy(vW)));float lit=max(dot(n,uSunDir),0.);
        float steep=1.-n.y;float snow=smoothstep(.2,.55,steep)*smoothstep(1.,6.,vW.y);
        float grain=fbm(vW.xz*1.6)*.5+.5;
        vec3 rock=vec3(.012,.0125,.014)*grain;
        vec3 col=rock+vec3(.42,.43,.45)*pow(lit,1.6)*snow*grain*uTint+vec3(.03)*lit;
        float dist=length(vW-cameraPosition);float fog=1.-exp(-pow(dist*.012,1.6));
        float low=exp(-max(vW.y,0.)*.35);
        float near=smoothstep(60.,8.,dist);col=mix(col,uHaze*.5,clamp(fog*(.35+.65*low)*(1.-near*.85),0.,1.));col*=1.-near*.55*low;
        gl_FragColor=vec4(col,1.);}`
  });
  const terrain = new THREE.Mesh(terrainGeo, landMaterial); terrain.name = 'Terrain'; terrain.position.z = 35; scene.add(terrain);

  // The monolith.
  const monolithMaterial = new THREE.ShaderMaterial({
    uniforms: { uSunDir: { value: landMaterial.uniforms.uSunDir.value }, uHaze: { value: haze }, uTint: { value: sun.clone() } },
    vertexShader: /* glsl */`varying vec3 vW;varying vec3 vN;void main(){vec4 w=modelMatrix*vec4(position,1.);vW=w.xyz;vN=normalize(mat3(modelMatrix)*normal);gl_Position=projectionMatrix*viewMatrix*w;}`,
    fragmentShader: /* glsl */`varying vec3 vW;varying vec3 vN;uniform vec3 uSunDir,uHaze,uTint;
      void main(){vec3 n=normalize(vN);vec3 v=normalize(cameraPosition-vW);float lit=max(dot(n,uSunDir),0.);
        float spec=pow(max(dot(reflect(-uSunDir,n),v),0.),40.);
        vec3 col=vec3(.004,.0042,.005)+uTint*(lit*.03+spec*.25);
        float dist=length(vW-cameraPosition);col=mix(col,uHaze*.55,clamp(1.-exp(-pow(dist*.012,1.6)),0.,1.)*.85);
        col+=uHaze*.25*smoothstep(1.6,0.,vW.y);
        gl_FragColor=vec4(col,1.);}`
  });
  const monolith = new THREE.Mesh(new THREE.BoxGeometry(1.25, 4.6, .32), monolithMaterial); monolith.name = 'Monolith'; monolith.position.set(0, 2.25, -14); scene.add(monolith);

  // Ground mist: drifting translucent sheets.
  const mistMaterial = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uHaze: { value: haze }, uSeed: { value: 0 } },
    vertexShader: /* glsl */`varying vec2 vUv;varying vec3 vW;void main(){vUv=uv;vec4 w=modelMatrix*vec4(position,1.);vW=w.xyz;gl_Position=projectionMatrix*viewMatrix*w;}`,
    fragmentShader: NOISE + /* glsl */`varying vec2 vUv;varying vec3 vW;uniform float uTime,uSeed;uniform vec3 uHaze;
      void main(){float n=fbm(vW.xz*.09+vec2(uTime*.02+uSeed,uSeed));float edge=smoothstep(0.,.25,vUv.x)*smoothstep(1.,.75,vUv.x)*smoothstep(0.,.3,vUv.y)*smoothstep(1.,.6,vUv.y);
        gl_FragColor=vec4(uHaze*.9,pow(n,2.2)*.55*edge);}`,
    transparent: true, depthWrite: false
  });
  const mists = [];
  for (let i = 0; i < 4; i++) { const m = new THREE.Mesh(new THREE.PlaneGeometry(90, 50), mistMaterial.clone()); m.material.uniforms.uSeed.value = i * 3.7; m.name = `Mist_${i}`; m.rotation.x = -Math.PI / 2; m.position.set(0, .35 + i * .45, -20 - i * 4); scene.add(m); mists.push(m); }

  // Floating dust in the light.
  const r = rng(5), dust = [];
  for (let i = 0; i < (isMobile() ? 120 : 260); i++) dust.push((r() - .5) * 30, r() * 7, -r() * 26);
  const dustGeo = new THREE.BufferGeometry(); dustGeo.setAttribute('position', new THREE.Float32BufferAttribute(dust, 3));
  const dustPts = new THREE.Points(dustGeo, new THREE.PointsMaterial({ color: 0x9a9da4, size: .03, transparent: true, opacity: .28, depthWrite: false, blending: THREE.AdditiveBlending })); dustPts.name = 'Dust'; scene.add(dustPts);

  function resize(w, h) {
    camera.aspect = w / h; camera.fov = w / h < 1 ? 58 : 34; camera.updateProjectionMatrix(); state.aspect = w / h;
    skyMaterial.uniforms.uAspect.value = state.aspect;
    // Arc from the top-left corner down to the horizon at the centre.
    const A = new THREE.Vector2(-.47 * state.aspect, .52), B = new THREE.Vector2(.02, -.17);
    const cx = Math.max(.3, state.aspect * .32), cy = ((cx - A.x) ** 2 - cx ** 2 + A.y ** 2 - B.y ** 2 + 2 * cx * B.x - B.x ** 2) / (2 * (A.y - B.y));
    state.arcC.set(cx, cy); state.arcR = Math.hypot(B.x - cx, B.y - cy); skyMaterial.uniforms.uArcR.value = state.arcR;
  }
  function update({ time, pointer, motion, local }) {
    const dolly = local ?? .5;
    camera.position.set(pointer.x * .6 * motion, 1.25 - pointer.y * .25 * motion + dolly * .25, 19 - dolly * 5);
    camera.lookAt(0, 5.6 + dolly * .2, -20);
    // The sun slides along the horizon with the pointer.
    // True horizon on screen (pitch of the camera), so the sky glow sits behind the far terrain.
    const pitch = Math.atan2(camera.position.y - (5.6 + dolly * .2), camera.position.z + 20);
    const horizon = .5 + .5 * Math.tan(pitch) / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    skyMaterial.uniforms.uHorizon.value = horizon;
    state.sun.set(.075 + (pointer.x * .5 + .5) * .1 * motion + (motion ? 0 : .05), horizon + .1);
    skyMaterial.uniforms.uShift.value.set(-pointer.x * .006 * motion, pointer.y * .004 * motion);
    const sx = (state.sun.x - .5) * state.aspect; landMaterial.uniforms.uSunDir.value.set(sx * 1.4, .2, -.7).normalize();
    skyMaterial.uniforms.uTime.value = time;
    mists.forEach((m, i) => { m.material.uniforms.uTime.value = time; m.position.x = Math.sin(time * .03 + i) * 3; });
    dustPts.rotation.y = Math.sin(time * .02) * .05; dustPts.position.y = Math.sin(time * .15) * .1;
  }
  function setPalette(color, name) { sun.set(name === 'ice' ? '#dbeaff' : '#f2e6d2'); for (const m of [skyMaterial, landMaterial, monolithMaterial]) m.uniforms.uTint.value.copy(sun); }
  function exportGroup() {
    const g = new THREE.Group(); g.name = 'Chapitre_02_Monolithe';
    g.add(portable(monolith, { color: 0x050506, roughness: .25, metalness: .6 }));
    g.add(portable(terrain, { color: 0x3a3c40, roughness: 1, flatShading: true }));
    return g;
  }
  return { name: 'monolith', scene, camera, resize, update, setPalette, exportGroup, post: { ca: .0008, bloom: .55, exposure: 1.15, sat: .35 } };
}
