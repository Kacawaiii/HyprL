import * as THREE from 'three';
import { NOISE, VERT_SCREEN, VERT_WORLD, glow, rng, portable, portableInstances } from '../lib/kit.js';

/**
 * Chapitre 03 — Prisme (référence : orbe qui éclate, éclats de verre irisés, lignes de lumière,
 * aberration chromatique). Scroll breaks the orb further; the pointer turns the cloud.
 */
export function createPrismChapter({ isMobile }) {
  const scene = new THREE.Scene(); scene.name = 'Chapitre_03_Prisme';
  const camera = new THREE.PerspectiveCamera(45, 1, .1, 200); camera.position.set(0, 0, 12);
  const violet = new THREE.Color('#9aa0aa'), orbPos = new THREE.Vector3(1.2, 1.3, -3);
  const state = { aspect: 1, explode: .5 };

  const bgMaterial = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uAspect: { value: 1 }, uOrb: { value: new THREE.Vector2(.6, .62) }, uViolet: { value: violet } },
    vertexShader: VERT_SCREEN, depthWrite: false, depthTest: false,
    fragmentShader: NOISE + /* glsl */`varying vec2 vUv;uniform float uTime,uAspect;uniform vec2 uOrb;uniform vec3 uViolet;
      void main(){vec2 asp=vec2(uAspect,1.);float r=length((vUv-uOrb)*asp);
        vec3 col=mix(vec3(.11,.113,.12),vec3(.02,.021,.024),smoothstep(.16,.62,vUv.y));col=mix(col,vec3(.0035,.0038,.0046),smoothstep(.55,1.05,vUv.y));
        col+=uViolet*(exp(-r*2.6)*.05+exp(-r*9.)*.07);
        float n=fbm(vUv*vec2(2.4,7.)*asp+vec2(uTime*.008,0.));col+=vec3(.06)*n*n*smoothstep(.55,.15,vUv.y);
        gl_FragColor=vec4(col,1.);}`
  });
  const bg = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), bgMaterial); bg.name = 'Backdrop'; bg.frustumCulled = false; bg.renderOrder = -10; scene.add(bg);

  // Reflective plank floor (bluish, fading into the dark).
  const floorMaterial = new THREE.ShaderMaterial({
    uniforms: { uOrb: { value: orbPos }, uViolet: { value: violet }, uTime: { value: 0 } },
    vertexShader: VERT_WORLD, transparent: true, depthWrite: false,
    fragmentShader: NOISE + /* glsl */`varying vec3 vW;uniform vec3 uOrb,uViolet;uniform float uTime;
      void main(){float plank=smoothstep(.0,.04,abs(fract(vW.x*.55)-.5)*2.-.02);float seam=1.-plank;
        float grain=fbm(vec2(vW.x*2.,vW.z*.25))*.5+.5;
        vec3 v=normalize(cameraPosition-vW);float fres=pow(clamp(1.-v.y,0.,1.),3.);
        float refl=exp(-abs(vW.x-uOrb.x)*.35)*exp(-abs(vW.z-uOrb.z)*.06);
        vec3 col=vec3(.012,.0125,.014)*grain*(.5+fres*.6)+uViolet*refl*.12*(.7+.3*grain)+vec3(.8)*seam*.03*refl;
        float fade=smoothstep(-40.,-6.,vW.z)*smoothstep(14.,2.,vW.z);
        gl_FragColor=vec4(col,fade);}`
  });
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 60), floorMaterial); floor.name = 'PlankFloor'; floor.rotation.x = -Math.PI / 2; floor.position.set(0, -4.4, -10); scene.add(floor);

  // The orb, cracked with a 3D Voronoi pattern.
  const orbMaterial = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uViolet: { value: violet }, uCrack: { value: .5 } },
    vertexShader: /* glsl */`varying vec3 vP;varying vec3 vN;varying vec3 vV;void main(){vP=position;vec4 mv=modelViewMatrix*vec4(position,1.);vN=normalize(normalMatrix*normal);vV=normalize(-mv.xyz);gl_Position=projectionMatrix*mv;}`,
    fragmentShader: /* glsl */`varying vec3 vP;varying vec3 vN;varying vec3 vV;uniform float uTime,uCrack;uniform vec3 uViolet;
      vec3 h3(vec3 p){p=vec3(dot(p,vec3(127.1,311.7,74.7)),dot(p,vec3(269.5,183.3,246.1)),dot(p,vec3(113.5,271.9,124.6)));return fract(sin(p)*43758.5453);}
      void main(){vec3 p=vP*2.4;vec3 i=floor(p),f=fract(p);float d1=8.,d2=8.;
        for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++)for(int z=-1;z<=1;z++){vec3 g=vec3(x,y,z);vec3 o=h3(i+g);float d=length(g+o-f);if(d<d1){d2=d1;d1=d;}else if(d<d2)d2=d;}
        float crack=1.-smoothstep(0.,.07+uCrack*.05,d2-d1);
        float fres=pow(clamp(1.-dot(normalize(vN),normalize(vV)),0.,1.),2.5);
        vec3 col=vec3(.006)+uViolet*fres*.9+vec3(.85,.8,1.)*fres*fres*.6+mix(uViolet,vec3(1.),.5)*crack*(.6+uCrack*2.2);
        gl_FragColor=vec4(col,1.);}`
  });
  const orb = new THREE.Mesh(new THREE.SphereGeometry(1.25, 96, 64), orbMaterial); orb.name = 'Orb'; orb.position.copy(orbPos); scene.add(orb);
  const haloMaterial = glow(/* glsl */`varying vec2 vUv;uniform vec3 uViolet;uniform float uPower;void main(){float d=length(vUv-.5)*2.;float edgeF=smoothstep(1.,.75,d);float ring=exp(-((d-.52)*14.)*((d-.52)*14.))*.6+exp(-d*3.)*.35;gl_FragColor=vec4(mix(uViolet,vec3(1.),.35)*ring*uPower*edgeF,1.);}`,
    { uViolet: { value: violet }, uPower: { value: 1 } }, { depthTest: false });
  const halo = new THREE.Mesh(new THREE.PlaneGeometry(5, 5), haloMaterial); halo.name = 'OrbHalo'; halo.position.copy(orbPos); halo.renderOrder = 3; scene.add(halo);

  // Shards: iridescent glass slabs on explosion trajectories.
  const count = isMobile() ? 22 : 44, r = rng(303);
  const shardGeo = new THREE.BoxGeometry(1, 1, 1);
  const seeds = new Float32Array(count); for (let i = 0; i < count; i++) seeds[i] = r();
  shardGeo.setAttribute('aSeed', new THREE.InstancedBufferAttribute(seeds, 1));
  const shardMaterial = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uViolet: { value: violet } },
    vertexShader: /* glsl */`attribute float aSeed;varying vec3 vW;varying vec3 vN;varying vec3 vL;varying float vSeed;
      void main(){mat4 m=modelMatrix*instanceMatrix;vec4 w=m*vec4(position,1.);vW=w.xyz;vN=normalize(mat3(m)*normal);vL=position;vSeed=aSeed;gl_Position=projectionMatrix*viewMatrix*w;}`,
    fragmentShader: /* glsl */`varying vec3 vW;varying vec3 vN;varying vec3 vL;varying float vSeed;uniform float uTime;uniform vec3 uViolet;
      void main(){vec3 n=normalize(vN);vec3 v=normalize(cameraPosition-vW);float c=clamp(abs(dot(n,v)),0.,1.);float f=pow(1.-c,2.);
        vec3 dd=clamp(.5-abs(vL),0.,.5);float mid=dd.x+dd.y+dd.z-max(dd.x,max(dd.y,dd.z))-min(dd.x,min(dd.y,dd.z));float edge=exp(-mid*2.*34.);
        float h=c*2.2+vW.y*.12+uTime*.04+vSeed;vec3 film=.5+.5*cos(6.2831*(h+vec3(0.,.33,.67)));
        vec3 col=uViolet*(.03+f*.22)+film*(f*.06+edge*.03)+vec3(.9,.88,1.)*edge*(.22+.25*f)+vec3(1.)*pow(max(dot(reflect(-v,n),normalize(vec3(.3,.8,.5))),0.),30.)*.35;
        gl_FragColor=vec4(col,1.);}`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide
  });
  const shards = new THREE.InstancedMesh(shardGeo, shardMaterial, count); shards.name = 'GlassShards'; shards.frustumCulled = false;
  const cloud = new THREE.Group(); cloud.name = 'ShardCloud'; cloud.position.copy(orbPos); scene.add(cloud); cloud.add(shards);
  const shardData = [];
  for (let i = 0; i < count; i++) {
    const dir = new THREE.Vector3(r() - .5, (r() - .5) * .75, (r() - .35) * .7).normalize(), long = r() < .55;
    shardData.push({ dir, d0: 1.5 + r() * 1.8, speed: 1 + r() * 3.2, axis: new THREE.Vector3(r() - .5, r() - .5, r() - .5).normalize(), spin: (r() - .5) * .5, phase: r() * 6,
      scale: long ? new THREE.Vector3(.05 + r() * .09, .7 + r() * 1.6, .015 + r() * .02) : new THREE.Vector3(.22 + r() * .55, .16 + r() * .45, .015 + r() * .015) });
  }

  // Light lines crossing the scene.
  const lineCount = isMobile() ? 5 : 8, lp = [], lc = [];
  for (let i = 0; i < lineCount; i++) {
    const a = r() * Math.PI, o = new THREE.Vector3((r() - .5) * 6, (r() - .5) * 4, -2 - r() * 6), d = new THREE.Vector3(Math.cos(a), Math.sin(a), (r() - .5) * .6).multiplyScalar(14 + r() * 10);
    lp.push(o.x - d.x, o.y - d.y, o.z - d.z, o.x + d.x, o.y + d.y, o.z + d.z); const b = .08 + r() * .22; lc.push(b, b, b, b, b, b);
  }
  const lineGeo = new THREE.BufferGeometry(); lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(lp, 3)); lineGeo.setAttribute('color', new THREE.Float32BufferAttribute(lc, 3));
  const lines = new THREE.LineSegments(lineGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })); lines.name = 'LightLines'; scene.add(lines);

  // Debris dust flung from the orb.
  const dn = isMobile() ? 220 : 520, dpos = new Float32Array(dn * 3), ddir = [];
  for (let i = 0; i < dn; i++) ddir.push({ v: new THREE.Vector3(r() - .5, (r() - .5) * .8, (r() - .4) * .8).normalize(), d: 1.3 + r() * 3.5, s: .4 + r() * 2 });
  const debrisGeo = new THREE.BufferGeometry(); debrisGeo.setAttribute('position', new THREE.BufferAttribute(dpos, 3));
  const debris = new THREE.Points(debrisGeo, new THREE.PointsMaterial({ color: 0xc8cacf, size: .035, transparent: true, opacity: .8, blending: THREE.AdditiveBlending, depthWrite: false })); debris.name = 'Debris'; debris.frustumCulled = false; cloud.add(debris);

  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), v3 = new THREE.Vector3(), tmp = new THREE.Vector3();
  function resize(w, h) {
    camera.aspect = w / h; camera.updateProjectionMatrix(); state.aspect = w / h; bgMaterial.uniforms.uAspect.value = state.aspect;
    const mobile = w < 600; orbPos.set(mobile ? .4 : 3.5, mobile ? 2.6 : 2.4, -5); orb.position.copy(orbPos); halo.position.copy(orbPos); cloud.position.copy(orbPos);
    orb.scale.setScalar(mobile ? .7 : 1); halo.scale.setScalar(mobile ? .7 : 1);
    tmp.copy(orbPos).project(camera); bgMaterial.uniforms.uOrb.value.set(tmp.x * .5 + .5, tmp.y * .5 + .5);
  }
  function update({ time, pointer, motion, local }) {
    const lp01 = local ?? .5; state.explode = .15 + lp01 * .75 + Math.sin(time * .25) * .04 * motion;
    camera.position.set(pointer.x * .5 * motion, -pointer.y * .35 * motion, 12 - lp01 * 1.5); camera.lookAt(0, .3, -3);
    cloud.rotation.set(pointer.y * .25 * motion + Math.sin(time * .1) * .05, pointer.x * .45 * motion + time * .03, 0);
    orb.rotation.y = time * .05; orbMaterial.uniforms.uCrack.value = state.explode; orbMaterial.uniforms.uTime.value = time;
    shardMaterial.uniforms.uTime.value = time; bgMaterial.uniforms.uTime.value = time;
    haloMaterial.uniforms.uPower.value = .7 + state.explode * .6;
    for (let i = 0; i < count; i++) {
      const s = shardData[i], dist = s.d0 + state.explode * s.speed + Math.sin(time * .3 + s.phase) * .12;
      v3.copy(s.dir).multiplyScalar(dist); q.setFromAxisAngle(s.axis, s.phase + time * s.spin + state.explode * s.spin * 4);
      shards.setMatrixAt(i, m4.compose(v3, q, s.scale));
    }
    shards.instanceMatrix.needsUpdate = true;
    for (let i = 0; i < dn; i++) { const d = ddir[i], k = d.d + state.explode * d.s; dpos[i * 3] = d.v.x * k; dpos[i * 3 + 1] = d.v.y * k + Math.sin(time * .4 + i) * .03; dpos[i * 3 + 2] = d.v.z * k; }
    debrisGeo.attributes.position.needsUpdate = true;
    lines.rotation.z = Math.sin(time * .05) * .04;
  }
  function setPalette() {}
  function exportGroup() {
    const g = new THREE.Group(); g.name = 'Chapitre_03_Prisme';
    g.add(portable(orb, { color: 0x1a1b1e, emissive: 0x9aa0aa, emissiveIntensity: .25, roughness: .15, metalness: .1 }));
    g.add(portableInstances(shards, { color: 0xc8ccd4, transparent: true, opacity: .45, roughness: .05, metalness: 0 }, 'GlassShards', new THREE.BoxGeometry(1, 1, 1)));
    return g;
  }
  return { name: 'prism', scene, camera, resize, update, setPalette, exportGroup, post: { ca: .0035, bloom: .8, exposure: 1.1, sat: .2 } };
}
