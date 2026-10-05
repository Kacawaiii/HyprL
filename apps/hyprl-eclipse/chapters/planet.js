import * as THREE from 'three';
import { NOISE, VERT_SCREEN, glow, starField, portable } from '../lib/kit.js';

/**
 * Chapitre 02 — Planète. Same art direction as the Monolith: monochrome, cold greys, one low star on the left.
 * The ringed giant seen in the Monolith's sky: a thin lit crescent on its night side, banded clouds, rings with
 * their Cassini gap, the planet's shadow across the rings and the rings' shadow on the planet. Scrolling pushes the
 * camera toward the limb; the pointer turns the view slightly.
 */
export function createPlanetChapter({ isMobile }) {
  const scene = new THREE.Scene(); scene.name = 'Chapitre_02_Planete';
  const camera = new THREE.PerspectiveCamera(34, 1, .1, 2000);
  const tint = new THREE.Color('#f2e6d2');
  const R = 10, ringIn = 13.2, ringOut = 25;
  const sunDir = new THREE.Vector3(-.86, .2, -.47).normalize();          // from the planet toward the star
  const ringNormal = new THREE.Vector3(.32, .93, .18).normalize();       // rings open ~22° to the camera
  const state = { sun: new THREE.Vector2(.06, .78), aspect: 1 };

  // Space: near-black, a faint cold haze, the star at the left edge with a soft cross.
  const skyMaterial = new THREE.ShaderMaterial({
    uniforms: { uAspect: { value: 1 }, uSun: { value: state.sun }, uTint: { value: tint.clone() }, uTime: { value: 0 } },
    vertexShader: VERT_SCREEN, depthWrite: false, depthTest: false,
    fragmentShader: NOISE + /* glsl */`varying vec2 vUv;uniform float uAspect,uTime;uniform vec2 uSun;uniform vec3 uTint;
      void main(){vec2 asp=vec2(uAspect,1.);vec3 col=mix(vec3(.004,.0045,.0055),vec3(.012,.013,.016),smoothstep(1.,0.,vUv.y));
        float n=fbm(vUv*asp*1.6+vec2(uTime*.004,0.));col+=vec3(.02,.021,.025)*n*n*smoothstep(.9,.1,length((vUv-vec2(.25,.6))*asp));
        vec2 d=(vUv-uSun)*asp;float r=length(d);
        col+=uTint*(exp(-r*3.)*.06+exp(-r*9.)*.22)+vec3(1.)*exp(-r*r*9000.)*4.;
        col+=uTint*(exp(-abs(d.y)*240.)*exp(-abs(d.x)*9.)*.5+exp(-abs(d.x)*260.)*exp(-abs(d.y)*14.)*.35);
        gl_FragColor=vec4(col,1.);}`
  });
  const sky = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), skyMaterial); sky.frustumCulled = false; sky.renderOrder = -10; sky.name = 'Space'; scene.add(sky);
  const stars = starField({ count: isMobile() ? 220 : 520, spread: [900, 520], depth: [-400, -700], seed: 61, color: '#c9ced8', size: [1.1, 2.6] });
  stars.material.uniforms.uFade.value = .5; scene.add(stars);

  // The planet: banded clouds along its own latitude (pole = ring normal), soft terminator, lit limb, ring shadow.
  const planetMaterial = new THREE.ShaderMaterial({
    uniforms: { uSun: { value: sunDir }, uPole: { value: ringNormal }, uTint: { value: tint.clone() }, uTime: { value: 0 }, uRing: { value: new THREE.Vector2(ringIn, ringOut) }, uR: { value: R } },
    vertexShader: /* glsl */`varying vec3 vP;varying vec3 vW;varying vec3 vN;void main(){vP=position;vec4 w=modelMatrix*vec4(position,1.);vW=w.xyz;vN=normalize(mat3(modelMatrix)*normal);gl_Position=projectionMatrix*viewMatrix*w;}`,
    fragmentShader: NOISE + /* glsl */`varying vec3 vP;varying vec3 vW;varying vec3 vN;uniform vec3 uSun,uPole,uTint;uniform float uTime,uR;uniform vec2 uRing;
      float ringDensity(float r){float t=(r-uRing.x)/(uRing.y-uRing.x);if(t<0.||t>1.)return 0.;
        float bands=.55+.45*fbm(vec2(t*38.,1.3));float gap=smoothstep(.015,.0,abs(t-.66))*.9;
        return clamp(bands*(1.-gap)*smoothstep(0.,.05,t)*smoothstep(1.,.9,t),0.,1.);}
      void main(){vec3 n=normalize(vN);vec3 v=normalize(cameraPosition-vW);
        float lat=dot(normalize(vP),uPole);vec3 east=normalize(cross(uPole,vec3(.1,0.,1.)));float lon=atan(dot(normalize(vP),cross(uPole,east)),dot(normalize(vP),east));
        float warp=fbm(vec2(lon*2.,lat*6.)+uTime*.006);float bands=fbm(vec2(lat*26.+warp*1.6,lon*.6));
        vec3 albedo=mix(vec3(.3,.31,.33),vec3(.62,.63,.66),bands)*(.85+.15*fbm(vec2(lon*9.,lat*40.)));
        float d=dot(n,uSun);float lit=smoothstep(-.06,.45,d)*(.35+.65*max(d,0.));
        // Shadow of the rings: walk toward the star and hit the ring plane.
        float t=-dot(vW,uPole)/dot(uSun,uPole);float sh=1.;if(t>0.){vec3 h=vW+uSun*t;sh=1.-ringDensity(length(h))*.85;}
        vec3 col=albedo*lit*sh*uTint*1.25;
        float rim=pow(1.-max(dot(n,v),0.),4.)*smoothstep(-.25,.35,d);col+=vec3(.75,.8,.9)*rim*.9;
        col+=vec3(.004,.0045,.006)*(1.-lit);
        gl_FragColor=vec4(col,1.);}`
  });
  const planet = new THREE.Mesh(new THREE.SphereGeometry(R, 160, 120), planetMaterial); planet.name = 'Planet'; scene.add(planet);

  // Atmosphere: a thin backlit halo around the limb, strongest on the star's side.
  const atmo = new THREE.Mesh(new THREE.SphereGeometry(R * 1.06, 96, 64), new THREE.ShaderMaterial({
    uniforms: { uSun: { value: sunDir }, uTint: { value: tint.clone() } },
    vertexShader: /* glsl */`varying vec3 vW;varying vec3 vN;void main(){vec4 w=modelMatrix*vec4(position,1.);vW=w.xyz;vN=normalize(mat3(modelMatrix)*normal);gl_Position=projectionMatrix*viewMatrix*w;}`,
    fragmentShader: /* glsl */`varying vec3 vW;varying vec3 vN;uniform vec3 uSun,uTint;
      void main(){vec3 n=normalize(vN);vec3 v=normalize(cameraPosition-vW);float f=pow(1.-abs(dot(n,v)),3.);
        float side=smoothstep(-.4,.5,dot(n,uSun));gl_FragColor=vec4((vec3(.7,.76,.88)*.55+uTint*.25)*f*side*.9,1.);}`,
    side: THREE.BackSide, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending
  }));
  atmo.name = 'Atmosphere'; scene.add(atmo);

  // Rings: banded, translucent; lit side brighter, forward-scattering when the star is behind; planet's shadow.
  const ringMaterial = new THREE.ShaderMaterial({
    uniforms: { uSun: { value: sunDir }, uPole: { value: ringNormal }, uTint: { value: tint.clone() }, uRing: { value: new THREE.Vector2(ringIn, ringOut) }, uR: { value: R } },
    vertexShader: /* glsl */`varying vec3 vW;void main(){vec4 w=modelMatrix*vec4(position,1.);vW=w.xyz;gl_Position=projectionMatrix*viewMatrix*w;}`,
    fragmentShader: NOISE + /* glsl */`varying vec3 vW;uniform vec3 uSun,uPole,uTint;uniform vec2 uRing;uniform float uR;
      void main(){float r=length(vW);float t=(r-uRing.x)/(uRing.y-uRing.x);
        float dens=(.55+.45*fbm(vec2(t*38.,1.3)))*(1.-smoothstep(.015,.0,abs(t-.66))*.9)*smoothstep(0.,.05,t)*smoothstep(1.,.9,t);
        dens*=.75+.25*fbm(vec2(t*160.,4.1));
        vec3 v=normalize(cameraPosition-vW);float sameSide=sign(dot(v,uPole))*sign(dot(uSun,uPole));
        float lit=sameSide>0.?abs(dot(uSun,uPole))*.9+.15:.12+pow(max(dot(-v,uSun),0.),6.)*.9;
        float b=dot(vW,uSun),c=dot(vW,vW)-uR*uR;float shadow=(b<0.&&b*b-c>0.)?.06:1.;
        vec3 col=mix(vec3(.38,.39,.42),vec3(.7,.71,.74),fbm(vec2(t*90.,8.)))*lit*shadow*uTint*1.2;
        gl_FragColor=vec4(col,clamp(dens,0.,1.)*.92);}`,
    transparent: true, depthWrite: false, side: THREE.DoubleSide
  });
  const ring = new THREE.Mesh(new THREE.RingGeometry(ringIn, ringOut, 360, 8), ringMaterial); ring.name = 'Rings';
  ring.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), ringNormal); scene.add(ring);

  const target = new THREE.Vector3(), offset = new THREE.Vector3();
  function resize(w, h) {
    camera.aspect = w / h; camera.fov = w / h < 1 ? 56 : 34; camera.updateProjectionMatrix(); state.aspect = w / h;
    skyMaterial.uniforms.uAspect.value = state.aspect; state.sun.set(state.aspect < 1 ? .1 : .06, .8);
  }
  function update({ time, pointer, motion, local }) {
    // Push in from the whole planet (lower right, room for the copy above) to its lit limb and the rings.
    const lp = local ?? .4, k = lp * lp * (3 - 2 * lp), dist = 92 - 58 * k;
    offset.set(.32 + pointer.x * .05 * motion, .1 - pointer.y * .04 * motion, 1).normalize().multiplyScalar(dist);
    camera.position.copy(offset);
    target.set(-R * (1.35 - .55 * k) * (state.aspect < 1 ? .4 : 1), R * (1.05 - .3 * k), 0);
    camera.lookAt(target);
    planet.rotation.y = time * .004;
    planetMaterial.uniforms.uTime.value = time; skyMaterial.uniforms.uTime.value = time;
    stars.material.uniforms.uTime.value = time;
  }
  function setPalette(color, name) {
    tint.set(name === 'ice' ? '#dbeaff' : '#f2e6d2');
    for (const m of [skyMaterial, planetMaterial, ringMaterial, atmo.material]) m.uniforms.uTint.value.copy(tint);
  }
  function exportGroup() {
    const g = new THREE.Group(); g.name = 'Chapitre_02_Planete';
    g.add(portable(planet, { color: 0x6a6c70, roughness: .9 }));
    g.add(portable(ring, { color: 0x9a9ca0, roughness: .8, transparent: true, opacity: .7, side: THREE.DoubleSide }));
    return g;
  }
  return { name: 'planet', scene, camera, resize, update, setPalette, exportGroup, post: { ca: .001, bloom: .6, exposure: 1.1, sat: .25 } };
}
