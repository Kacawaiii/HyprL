import * as THREE from 'three';
import { NOISE, VERT_SCREEN, glow, rng, portable } from '../lib/kit.js';

/**
 * Chapitre 03 — Singularité (référence : trou noir doré, disque vu par la tranche, vaisseau).
 * The accretion disc lies edge-on across the frame like a golden sea; above and below it the lensed far side of the
 * disc forms the ring, built from fine concentric filaments, wider and brighter on the approaching (left) side where a
 * plume of gold dust rises. Speed streaks run out of the vanishing point; the ship heads into it, steered by the
 * pointer. Backdrop in screen space (2.5D), ship and sparks in 3D.
 */
export function createSingularityChapter({ isMobile }) {
  const scene = new THREE.Scene(); scene.name = 'Chapitre_03_Singularite';
  const camera = new THREE.PerspectiveCamera(40, 1, .1, 400); camera.position.set(0, 0, 10);
  const gold = new THREE.Color('#ffb24a'), hot = new THREE.Color('#fff1d6');
  const state = { aspect: 1, center: new THREE.Vector2(.62, .47), horizon: .47, ship: new THREE.Vector3(), shipTarget: new THREE.Vector3(), roll: 0 };

  const backdrop = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uAspect: { value: 1 }, uC: { value: state.center }, uH: { value: .47 }, uRs: { value: .2 }, uGold: { value: gold }, uHot: { value: hot }, uFlow: { value: 0 } },
    vertexShader: VERT_SCREEN, depthWrite: false, depthTest: false,
    fragmentShader: NOISE + /* glsl */`varying vec2 vUv;uniform float uTime,uAspect,uH,uRs,uFlow;uniform vec2 uC;uniform vec3 uGold,uHot;
      // Lensed disc: concentric filaments around the shadow, a photon ring, a wide bright plume on the left.
      float ringI(vec2 p){float r=length(p),a=atan(p.y,p.x);if(r<uRs)return 0.;
        float left=smoothstep(-.1,1.,-cos(a));float plume=smoothstep(.2,1.,-cos(a))*smoothstep(-.9,.3,sin(a));
        float rr=r-uRs*(1.+.18*plume*sin(a*3.+uTime*.05));
        float w=.018+.075*left*left+.05*plume;
        float fil=pow(fbm(vec2(r*120.,a*1.6+uTime*.25)),1.6)*1.1+pow(fbm(vec2(r*340.,a*4.-uTime*.4)),2.)*.8;
        float body=exp(-max(rr,0.)/w)*smoothstep(-.003,.01,rr)*(.15+fil*1.3);
        float photon=exp(-pow((r-uRs*1.025)/.0028,2.))*1.5+exp(-pow((r-uRs*1.06)/.01,2.))*.25;
        return (body*(.18+.95*left)+photon*(.45+.6*left));}
      void main(){vec2 asp=vec2(uAspect,1.);vec2 p=(vUv-uC)*asp;float y=vUv.y-uH;
        vec3 col=vec3(.002,.0018,.0015);
        vec2 g=floor(vUv*asp*260.);float st=step(.9965,hash12(g))*(.5+.5*sin(uTime*2.+hash12(g+3.)*40.));col+=vec3(.9,.85,.75)*st*.6;
        float I=ringI(p);
        // the edge-on disc: a bright seam on the horizon, glowing toward the left and the centre
        float seam=exp(-abs(y)*160.)*(.3+.55*smoothstep(.9,-.2,(vUv.x-uC.x)*uAspect))+exp(-abs(y)*25.)*.06;
        // the golden sea below the horizon: perspective flow toward the viewer, clouds, reflected ring
        vec3 sea=vec3(0.);float cov=0.;
        if(y<0.){float z=.06/(-y+.002);vec2 w=vec2((vUv.x-uC.x)*uAspect*z,z+uTime*uFlow);
          float cl=fbm(w*vec2(.9,.22));float cl2=fbm(w*vec2(2.6,.6)+cl*1.4);cov=smoothstep(.3,.8,cl*.6+cl2*.6);
          float lightL=smoothstep(.75,-.35,(vUv.x-uC.x)*uAspect);
          sea=mix(uGold*vec3(.4,.26,.12),uGold,cl2)*cov*(.14+1.15*lightL)*smoothstep(-.7,-.02,y);
          sea+=uHot*pow(cl2,7.)*cov*2.2*(.3+lightL);
          vec2 m=vec2(p.x+(fbm(w*vec2(.5,3.))-.5)*.03,-p.y);I=max(I,ringI(m)*.5*(1.-cov*.5));
        }
        // plume of gold dust rising on the left, above the horizon
        vec2 q=p-vec2(-uRs*1.6,.05);float plumeMask=smoothstep(.75,.0,length(q*vec2(.55,1.)))*smoothstep(-.02,.12,y)*smoothstep(.0,-.6,p.x+.1);
        float dust=fbm(q*3.+vec2(uTime*.02,-uTime*.03))*fbm(q*7.-uTime*.02);
        vec3 plume=mix(uGold*vec3(.55,.36,.17),uHot,smoothstep(.28,.5,dust))*smoothstep(.16,.48,dust)*plumeMask*1.25;
        col+=mix(uGold,uHot,clamp(I*.6,0.,1.))*I*.95;
        col+=uGold*seam*1.1+uHot*exp(-abs(y)*500.)*.6*smoothstep(1.,.0,abs(vUv.x-uC.x)*2.);
        col=col*(1.-cov*.35)+sea+plume;
        // speed streaks out of the vanishing point (on the horizon, at the centre)
        vec2 d=(vUv-vec2(uC.x,uH))*asp;float rd=length(d),th=atan(d.y,d.x);float bin=floor(th*90.);
        float lane=step(.7,hash12(vec2(bin,7.)));float dash=smoothstep(.75,1.,fract(log(rd+.02)*3.-uTime*(1.2+uFlow*6.)+hash12(vec2(bin,1.))));
        float thin=smoothstep(.6,0.,abs(fract(th*90.)-.5)*2.*rd*60.);
        col+=mix(uGold,uHot,.6)*lane*dash*thin*smoothstep(.04,.35,rd)*.85;
        col*=1.-smoothstep(.55,1.1,length((vUv-.5)*asp))*.5;
        gl_FragColor=vec4(col,1.);}`
  });
  const sky = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), backdrop); sky.frustumCulled = false; sky.renderOrder = -10; sky.name = 'Singularity'; scene.add(sky);

  // Gold sparks drifting toward the camera.
  const r = rng(808), sparkN = isMobile() ? 220 : 600, sp = new Float32Array(sparkN * 3), sparks = [];
  for (let i = 0; i < sparkN; i++) sparks.push({ x: (r() - .5) * 26, y: (r() - .5) * 10 - 1.5, z: -60 + r() * 66, v: 4 + r() * 9 });
  const sparkGeo = new THREE.BufferGeometry(); sparkGeo.setAttribute('position', new THREE.BufferAttribute(sp, 3));
  const sparkPts = new THREE.Points(sparkGeo, new THREE.PointsMaterial({ color: 0xffc46a, size: .045, transparent: true, opacity: .85, blending: THREE.AdditiveBlending, depthWrite: false }));
  sparkPts.name = 'GoldSparks'; sparkPts.frustumCulled = false; scene.add(sparkPts);

  // The ship: sleek arrow hull, canopy, fins, twin engines with white-hot exhaust and long light trails.
  const ship = new THREE.Group(); ship.name = 'Vessel'; scene.add(ship);
  const hullShape = new THREE.Shape([[2.3, 0], [.7, .2], [-.3, 1.05], [-.85, 1.05], [-.55, .32], [-1.15, .3], [-1.15, -.3], [-.55, -.32], [-.85, -1.05], [-.3, -1.05], [.7, -.2]].map(([x, y]) => new THREE.Vector2(x, y)));
  const hullGeo = new THREE.ExtrudeGeometry(hullShape, { depth: .12, bevelEnabled: true, bevelThickness: .06, bevelSize: .05, bevelSegments: 2 }); hullGeo.center(); hullGeo.rotateX(-Math.PI / 2);
  const metal = new THREE.MeshStandardMaterial({ color: 0x3a3d44, metalness: .7, roughness: .3 });
  const hull = new THREE.Mesh(hullGeo, metal); hull.name = 'Hull'; ship.add(hull);
  const canopy = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 16), new THREE.MeshStandardMaterial({ color: 0x0b0d12, metalness: .9, roughness: .1 })); canopy.name = 'Canopy'; canopy.scale.set(.55, .13, .17); canopy.position.set(.55, .12, 0); ship.add(canopy);
  const finShape = new THREE.Shape([new THREE.Vector2(0, 0), new THREE.Vector2(.55, 0), new THREE.Vector2(.05, .42)]);
  for (const s of [-1, 1]) { const fin = new THREE.Mesh(new THREE.ExtrudeGeometry(finShape, { depth: .03, bevelEnabled: false }), metal); fin.name = `Fin_${s > 0 ? 'R' : 'L'}`; fin.position.set(-1.05, .05, s * .25); fin.rotation.x = s * .25; ship.add(fin); }
  const engineGeo = new THREE.CylinderGeometry(.11, .14, .5, 20); engineGeo.rotateZ(Math.PI / 2);
  const exhaustMaterial = glow(/* glsl */`varying vec2 vUv;uniform vec3 uHot,uGold;uniform float uPulse;void main(){vec2 p=vUv-vec2(1.,.5);float core=exp(-abs(p.y)*46.)*exp(p.x*1.4);float g=exp(-length(p*vec2(.7,3.))*4.);
    gl_FragColor=vec4((uHot*core*2.4+mix(uGold,uHot,.4)*g*.9)*uPulse,1.);}`, { uHot: { value: hot }, uGold: { value: gold }, uPulse: { value: 1 } }, { side: THREE.DoubleSide });
  for (const s of [-1, 1]) {
    const e = new THREE.Mesh(engineGeo, metal); e.name = `Engine_${s > 0 ? 'R' : 'L'}`; e.position.set(-1.1, 0, s * .2); ship.add(e);
    const trail = new THREE.Mesh(new THREE.PlaneGeometry(9, .55), exhaustMaterial); trail.name = `Exhaust_${s > 0 ? 'R' : 'L'}`; trail.position.set(-1.35 - 4.5, 0, s * .2); ship.add(trail);
    const trail2 = trail.clone(); trail2.rotation.x = Math.PI / 2; ship.add(trail2);
  }
  ship.add(new THREE.PointLight(0xffd9a0, 6, 6, 2).translateX(-1.6));
  const key = new THREE.DirectionalLight(0xffc06a, 3.2); key.position.set(-6, 1, -4); scene.add(key);
  scene.add(new THREE.HemisphereLight(0xffd9a0, 0x1a1006, .6));

  function resize(w, h) {
    camera.aspect = w / h; camera.fov = w / h < 1 ? 60 : 40; camera.updateProjectionMatrix(); state.aspect = w / h;
    backdrop.uniforms.uAspect.value = state.aspect; backdrop.uniforms.uH.value = state.horizon = .47;
    ship.scale.setScalar(state.aspect < 1 ? .5 : .62);
  }
  const fwd = new THREE.Vector3(), tmp = new THREE.Vector3(), xAxis = new THREE.Vector3(1, 0, 0);
  function update({ time, pointer, motion, local, dt }) {
    const lp = local ?? .4;
    // Scroll pushes in: the shadow grows, the sea flows faster.
    backdrop.uniforms.uRs.value = .33 + lp * .05; backdrop.uniforms.uFlow.value = .08 + lp * .1; backdrop.uniforms.uTime.value = time;
    state.center.set((state.aspect < 1 ? .55 : .62) - pointer.x * .01 * motion, .47 + pointer.y * .008 * motion);
    camera.position.set(pointer.x * .25 * motion, -pointer.y * .15 * motion, 10); camera.lookAt(0, 0, 0); camera.updateMatrixWorld();
    for (let i = 0; i < sparkN; i++) { const s = sparks[i]; s.z += s.v * (dt || 0) * motion; if (s.z > 8) s.z -= 68; sp[i * 3] = s.x; sp[i * 3 + 1] = s.y; sp[i * 3 + 2] = s.z; }
    sparkGeo.attributes.position.needsUpdate = true;
    // Ship: right of the centre, just above the sea, heading into the vanishing point; the pointer steers it.
    const mobile = state.aspect < 1;
    state.shipTarget.set((mobile ? .9 : 3.1) + pointer.x * 1.4 * motion + Math.sin(time * .4) * .15, -.05 - pointer.y * .7 * motion + Math.sin(time * .9) * .06, 1.5);
    const prevX = state.ship.x; state.ship.lerp(state.shipTarget, state.ship.lengthSq() ? .06 : 1); ship.position.copy(state.ship);
    tmp.set(state.center.x * 2 - 1, state.horizon * 2 - 1, .98).unproject(camera);
    fwd.copy(tmp).sub(ship.position).normalize();
    ship.quaternion.setFromUnitVectors(xAxis, fwd);
    state.roll += (THREE.MathUtils.clamp(-(state.ship.x - prevX) * 8 - pointer.x * .3 * motion, -.5, .5) - state.roll) * .08;
    ship.rotateX(state.roll + .25);
    exhaustMaterial.uniforms.uPulse.value = .85 + .15 * Math.sin(time * 30);
  }
  function setPalette(color, name) { gold.set(name === 'ice' ? '#8fc6ff' : '#ffb24a'); hot.set(name === 'ice' ? '#eef6ff' : '#fff1d6'); key.color.set(name === 'ice' ? '#a9d4ff' : '#ffc06a'); sparkPts.material.color.set(name === 'ice' ? '#a9d4ff' : '#ffc46a'); }
  function exportGroup() {
    const g = new THREE.Group(); g.name = 'Chapitre_03_Singularite';
    const s = new THREE.Group(); s.name = 'Vessel'; for (const c of ship.children) if (c.isMesh && c.material.isMeshStandardMaterial) s.add(portable(c, { color: 0x3a3d44, metalness: .7, roughness: .3 })); g.add(s);
    return g;
  }
  return { name: 'singularity', scene, camera, resize, update, setPalette, exportGroup, post: { ca: .003, bloom: 1, exposure: 1 } };
}
