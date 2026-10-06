import * as THREE from 'three';
import { NOISE, VERT_SCREEN, glow, rng, portable } from '../lib/kit.js';

/**
 * Chapitre 03 — Singularité (référence : trou noir doré, disque vu par la tranche, vaisseau).
 * The accretion disc lies edge-on across the frame like a mirror-like golden sea; above it the lensed far side of the
 * disc forms a huge ring that overflows the frame, built from fine flowing filaments, white-hot at the photon ring,
 * wider and brighter on the approaching (left) side where it curls outward into the sea like a wave. The sea mirrors
 * the ring, carries banks of gold dust; a plume of dust wraps the ring on the left. Stars bend around the shadow,
 * the air shimmers at the photon ring, speed streaks run out of the vanishing point and sparks rush at the camera.
 * The ship, steered by the pointer, flies into the scene with its reflection and engine light on the sea.
 * Backdrop in screen space (2.5D), ship, reflection and sparks in 3D.
 */
export function createSingularityChapter({ isMobile }) {
  const scene = new THREE.Scene(); scene.name = 'Chapitre_03_Singularite';
  const camera = new THREE.PerspectiveCamera(40, 1, .1, 400); camera.position.set(0, 0, 10);
  const gold = new THREE.Color('#ffc45a'), hot = new THREE.Color('#fff1d6');
  const state = { aspect: 1, center: new THREE.Vector2(.64, .47), horizon: .47, ship: new THREE.Vector3(), shipTarget: new THREE.Vector3(), roll: 0 };
  const shipUv = new THREE.Vector2(.7, .5);

  const backdrop = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uAspect: { value: 1 }, uC: { value: state.center }, uH: { value: .47 }, uRs: { value: .4 }, uGold: { value: gold }, uHot: { value: hot }, uFlow: { value: 0 }, uShip: { value: shipUv }, uMotion: { value: 1 } },
    vertexShader: VERT_SCREEN, depthWrite: false, depthTest: false,
    fragmentShader: NOISE + /* glsl */`varying vec2 vUv;uniform float uTime,uAspect,uH,uRs,uFlow,uMotion;uniform vec2 uC,uShip;uniform vec3 uGold,uHot;
      // Subpixel emitters embedded in the dust, jittered within cells rather than a visible dot grid.
      vec3 glitter(vec2 uv,float density,float light,float scale){vec2 q=uv*scale,g=floor(q);
        float h=hash12(g+7.3);vec2 jitter=vec2(hash12(g+3.7),hash12(g+9.1))*.7+.15;
        float spot=1.-smoothstep(.09,.30,length(fract(q)-jitter));
        float tw=.7+.3*sin(uTime*uMotion*2.8+h*70.);
        return mix(uGold*3.2,uHot*4.5,pow(h,8.))*step(.978,h)*spot*pow(density,.7)*light*tw;}
      // Overlapping lit billows: analytical sphere depth gives dark cores and backlit rims without ray marching.
      vec4 billows(vec2 p,vec2 anchor,vec2 span,float radius,float seed){
        if(any(greaterThan(abs(p-anchor),span*.5+vec2(radius*1.8))))return vec4(0.);
        vec2 warp=vec2(fbm3(p*18.+seed),fbm3(p*18.+seed+4.7))-.5;
        vec2 pp=p+warp*radius*.32;float detail=fbm(p*58.+warp*3.+seed);
        vec2 relief=(vec2(fbm3(p*18.+seed+vec2(.09,0.)),fbm3(p*18.+seed+vec2(0.,.09)))-(warp.x+.5))*3.5;
        vec4 cloud=vec4(0.);
        for(int i=0;i<12;i++){float fi=float(i);vec2 h=vec2(hash12(vec2(fi,seed)),hash12(vec2(seed,fi+4.)));
          vec2 c=anchor+(h-.5)*span;float size=radius*(.65+.6*hash12(h*31.));
          vec2 q=(pp-c)/(vec2(1.18,1.)*size);float d=dot(q,q);
          if(d<1.06){float z=sqrt(max(1.-d,0.));vec3 n=normalize(vec3(q+relief,z));
            float light=max(dot(n,normalize(vec3(.6,.7,.35))),0.);
            float rim=pow(1.-z,2.5)*pow(light,2.)*3.8;
            vec3 colour=uGold*(.012+.34*light*light)*(.25+1.7*pow(detail,1.3));
            colour+=mix(uGold*1.8,uHot,.5)*rim*(.2+1.2*detail);
            float alpha=smoothstep(1.06,.62,d)*.94;
            cloud.rgb=mix(cloud.rgb,colour,alpha);cloud.a=alpha+cloud.a*(1.-alpha);}}
        return cloud;}
      // Lensed far side of the disc (p from the centre, upper half): fine filaments flowing around the shadow,
      // a white-hot photon ring, thin on the receding right, wide on the approaching left where it curls into a wave.
      vec3 ring(vec2 p){float r=length(p),a=atan(p.y,p.x);
        float left=smoothstep(-.35,1.,-cos(a)),s=max(sin(a),0.);
        float wave=left*left*exp(-s*7.);
        float w=.06+.17*left*left+.28*wave;
        float inner=exp(-pow((r-uRs*.944)/.0013,2.))*(.4+1.2*left);
        float rr=(r-uRs)/w;if(rr<-.05)return uHot*inner;
        float t=uTime*uMotion;
        // filaments: concentric, streaked along the angle (motion blur), drifting around the ring
        float ang=a*(1.+.6*wave)+rr*wave*.9;
        float f1=vnoise(vec2(rr*42.,ang*2.4-t*.35)),f2=vnoise(vec2(rr*130.,ang*6.-t*.6)),f3=vnoise(vec2(rr*12.,ang*1.2-t*.15));
        float fil=pow(f1,4.)*2.2+pow(f2,6.)*2.2+f3*f3*.3;
        float env=smoothstep(-.04,.04,rr)*exp(-rr*(2.-1.*wave))*smoothstep(1.4,.5,rr);
        float lum=env*(.03+fil)*(.45+1.2*left+.9*wave);
        float photon=exp(-pow((r-uRs)/.0022,2.))*(.7+2.2*left)+exp(-pow((r-uRs*1.01)/.012,2.))*(.05+.35*left);
        vec3 c=mix(uGold*1.65,uHot*2.,clamp(exp(-rr*5.)*.25+pow(fil*.35,2.),0.,.8))*lum+mix(uGold,uHot,.85)*photon+uHot*inner;
        return c;}
      // star field with gravitational lensing: an image at p shows the source at p(1 - rE^2/r^2)
      vec3 stars(vec2 p){float r=length(p);if(r<uRs*1.01)return vec3(0.);vec2 b=p*(1.-uRs*uRs*1.3/(r*r));
        vec2 g=floor(b*300.),f=fract(b*300.)-.5;float h=hash12(g);float tw=.6+.4*sin(uTime*2.+h*60.);
        float s=step(.991,h)*smoothstep(.45,.05,length(f))*tw;return vec3(.95,.88,.75)*s*(.4+1.6*step(.9993,h));}
      void main(){vec2 asp=vec2(uAspect,1.);vec2 p=(vUv-uC)*asp;float y=vUv.y-uH;float t=uTime*uMotion;
        vec3 col=vec3(.0025,.002,.0016);
        float r=length(p),a=atan(p.y,p.x);
        float leftX=smoothstep(.15,-.75,p.x);
        if(y>=0.){
          // heat shimmer just outside the photon ring
          vec2 sh=(vec2(vnoise(p*40.+t*1.3),vnoise(p*40.-t*1.1+7.))-.5)*.0025*exp(-abs(r-uRs)*40.);
          col+=stars(p+sh);
          col+=ring(p+sh);
          // the plume: gold dust wrapping the ring on the left, lit by it, with glitter
          vec2 pp=p*3.2+vec2(t*.015,-t*.02);vec2 wq=vec2(fbm(pp*.8),fbm(pp*.8+vec2(4.2,1.3)));
          float d1=fbm(pp+wq*1.6),d2=fbm(pp*2.3+wq*2.2+d1);
          float band=smoothstep(uRs*1.1,uRs*1.35,r)*smoothstep(uRs*2.2,uRs*1.5,r)*smoothstep(.75,.97,-cos(a))*smoothstep(.03,.12,y);
          float dust=smoothstep(.45,.72,d1*.65+d2*.45)*band;
          float lit=(.6+.8*exp(-max(r-uRs*1.3,0.)*4.));
          float scale=uRs/.476;vec4 plumeCloud=billows(p,vec2(-.69,.25)*scale,vec2(.26,.25)*scale,.09*scale,31.);
          float mask=plumeCloud.a*smoothstep(.03,.14,y);
          col=mix(col,plumeCloud.rgb,mask);
          col+=uGold*dust*.035+glitter(vUv*asp,mask*smoothstep(.4,.64,d2),lit*1.8,640.);
        }else{
          // the sea: mirror of the ring, rippled, with banks of gold dust coming at the camera
          float d=-y;float z=.05/(d+.004);vec2 w=vec2((vUv.x-uC.x)*uAspect*z,z*.4+t*uFlow);
          float c1=fbm(w*7.),c2=fbm(w*19.+c1*2.4),c3=fbm(w*19.+c1*2.4+vec2(.08,-.05));
          float edgeLit=clamp((c2-c3)*9.+.5,0.,1.);
          float lightL=smoothstep(.35,-.7,p.x);
          float cov=smoothstep(.45,.6,c1*.65+c2*.45-.1+.2*lightL)*(.12+.88*lightL)*smoothstep(.01,.06,d);
          vec2 m=vec2(p.x+(fbm(w*vec2(.5,4.))-.5)*.018*smoothstep(0.,.2,d),-p.y);
          vec3 refl=ring(m)*.6+stars(m)*.5;
          col+=refl*(1.-cov*.85);
          vec3 cloud=mix(uGold*vec3(.06,.038,.018),uGold*vec3(.95,.75,.45),smoothstep(.45,.8,c2)*(.35+.65*edgeLit))*(.3+1.1*lightL)+uHot*pow(c2,4.)*edgeLit*3.5*lightL;
          col=mix(col,cloud,cov);
          // Near cloud banks occupy the left foreground; projected billows scale down in portrait.
          float scale=uRs/.476;vec4 bankCloud=billows(p,vec2(-.72,-.35)*scale,vec2(.65,.31)*scale,.145*scale,13.);
          float bank=bankCloud.a*smoothstep(.06,.18,d);col=mix(col,bankCloud.rgb,bank);cov=max(cov,bank);
          float clustered=fbm3(p*25.+13.);
          col+=glitter(vUv*asp,bank*smoothstep(.46,.68,clustered),.7+lightL*1.8,700.);
          col+=glitter(vUv*asp+2.1,cov*smoothstep(.35,.7,c2),.5+lightL,410.);
          col+=uGold*exp(-d*22.)*(.03+.15*lightL);
          // engine light and ship mirrored on the water, stretched vertically
          vec2 e=(vUv-vec2(uShip.x+.012,uH-(uShip.y-uH)-.01))*asp;col+=vec3(.45,.65,1.)*exp(-abs(e.x)*90.)*exp(-abs(e.y)*30.)*.35*smoothstep(0.,.02,d);
          // speed streaks out of the vanishing point
          vec2 v=(vUv-vec2(uC.x-.03,uH))*asp;float rd=length(v),th=atan(v.y,v.x);float bin=floor(th*110.);
          float lane=step(.55,hash12(vec2(bin,3.)));float dash=smoothstep(.55,1.,fract(log(rd+.02)*1.3-t*(1.6+uFlow*8.)+hash12(vec2(bin,9.))));
          float thin=smoothstep(.55,0.,abs(fract(th*110.)-.5)*2.*rd*110.);
          col+=mix(uGold,uHot,.75)*lane*dash*thin*smoothstep(.05,.45,rd)*1.1;
        }
        // the edge-on disc on the horizon: white-hot seam, brightest on the left where the wave lands
        float seam=exp(-abs(y)*420.)*(.35+1.4*leftX)+exp(-abs(y)*60.)*(.06+.4*leftX);
        col+=mix(uGold,uHot,.7)*seam*smoothstep(1.,.0,(vUv.x-uC.x)*uAspect*.9);
        col*=1.-smoothstep(.6,1.2,length((vUv-.5)*asp))*.55;
        gl_FragColor=vec4(col,1.);}`
  });
  const sky = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), backdrop); sky.frustumCulled = false; sky.renderOrder = -10; sky.name = 'Singularity'; scene.add(sky);

  // Gold sparks rushing at the camera, drawn as short motion-blurred streaks (bright head, fading tail).
  const r = rng(808), sparkN = isMobile() ? 110 : 260, sp = new Float32Array(sparkN * 6), sc = new Float32Array(sparkN * 6), sparks = [];
  for (let i = 0; i < sparkN; i++) {
    sparks.push({ x: (r() - .5) * 30, y: -r() * r() * 5 + .4, z: -60 + r() * 66, v: 5 + r() * 12 });
    const b = .35 + r() * .65; sc.set([b, b, b, 0, 0, 0], i * 6);
  }
  const sparkGeo = new THREE.BufferGeometry(); sparkGeo.setAttribute('position', new THREE.BufferAttribute(sp, 3)); sparkGeo.setAttribute('color', new THREE.BufferAttribute(sc, 3));
  const sparkMaterial = new THREE.LineBasicMaterial({ color: gold, vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
  const sparkLines = new THREE.LineSegments(sparkGeo, sparkMaterial); sparkLines.name = 'GoldSparks'; sparkLines.frustumCulled = false; scene.add(sparkLines);

  // The ship: lit metal with panel lines, swept wings, twin engines, canopy, fins.
  const keyDir = new THREE.Vector3(-.75, .1, .35).normalize(), seaDir = new THREE.Vector3(-.2, -1, .25).normalize();
  const metalMaterial = dim => new THREE.ShaderMaterial({
    uniforms: { uKey: { value: keyDir }, uSea: { value: seaDir }, uGold: { value: gold }, uHot: { value: hot }, uDim: { value: dim }, uTone: { value: new THREE.Color('#3a3e46') } },
    vertexShader: /* glsl */`varying vec3 vW;varying vec3 vN;varying vec3 vL;void main(){vec4 w=modelMatrix*vec4(position,1.);vW=w.xyz;vN=normalize(mat3(modelMatrix)*normal);vL=position;gl_Position=projectionMatrix*viewMatrix*w;}`,
    fragmentShader: /* glsl */`varying vec3 vW;varying vec3 vN;varying vec3 vL;uniform vec3 uKey,uSea,uGold,uHot,uTone;uniform float uDim;
      void main(){vec3 n=normalize(vN);if(!gl_FrontFacing)n=-n;vec3 v=normalize(cameraPosition-vW);
        float panel=1.-.45*max(step(.94,fract(vL.x*5.3)),step(.95,fract((vL.z+vL.y)*4.1)));
        float k=max(dot(n,uKey),0.),s=max(dot(n,uSea),0.),fr=pow(1.-max(dot(n,v),0.),3.);
        float spec=pow(max(dot(reflect(-uKey,n),v),0.),40.),spec2=pow(max(dot(reflect(-normalize(vec3(.3,.9,.4)),n),v),0.),18.);
        vec3 col=uTone*panel*(.06+k*.35*uGold+s*.2*uGold+max(n.y,0.)*.25)+uHot*spec*2.2+vec3(.8,.85,.95)*spec2*.6+mix(uGold,uHot,.5)*fr*.2;
        gl_FragColor=vec4(col*uDim,1.);}`,
    side: THREE.DoubleSide
  });
  const exhaustMaterial = glow(/* glsl */`varying vec2 vUv;uniform vec3 uHot,uGold;uniform float uPulse,uDim;void main(){vec2 p=vUv-vec2(1.,.5);float core=exp(-abs(p.y)*60.)*exp(p.x*.9);float g=exp(-abs(p.y)*9.)*exp(p.x*2.2);
    gl_FragColor=vec4((vec3(.8,.92,1.)*core*4.+vec3(.24,.52,1.)*g*.8)*uPulse*uDim,1.);}`, { uHot: { value: hot }, uGold: { value: gold }, uPulse: { value: 1 }, uDim: { value: 1 } }, { side: THREE.DoubleSide });
  const nozzleMaterial = glow(/* glsl */`varying vec2 vUv;uniform vec3 uHot;uniform float uPulse,uDim;void main(){float d=length(vUv-.5)*2.;gl_FragColor=vec4(vec3(.75,.88,1.)*(exp(-d*d*6.)*3.5+exp(-d*2.5)*.6)*uPulse*uDim,1.);}`, { uHot: { value: hot }, uPulse: { value: 1 }, uDim: { value: 1 } }, { side: THREE.DoubleSide });
  const fuselageGeo = new THREE.LatheGeometry([[0, -1.3], [.2, -1.25], [.3, -.7], [.3, .3], [.25, 1.1], [.14, 1.8], [0, 2.3]].map(([x, y]) => new THREE.Vector2(x, y)), 24);
  fuselageGeo.rotateZ(-Math.PI / 2); fuselageGeo.scale(1, .55, 1);
  const wingShape = new THREE.Shape([[.7, 0], [-.55, 1.55], [-.95, 1.6], [-.8, .9], [-1.05, 0]].map(([x, y]) => new THREE.Vector2(x, y)));
  const wingGeo = new THREE.ExtrudeGeometry(wingShape, { depth: .05, bevelEnabled: true, bevelThickness: .02, bevelSize: .02, bevelSegments: 1 }); wingGeo.rotateX(Math.PI / 2);
  const finShape = new THREE.Shape([[0, 0], [.45, 0], [-.05, .28], [-.18, .28]].map(([x, y]) => new THREE.Vector2(x, y)));
  const finGeo = new THREE.ExtrudeGeometry(finShape, { depth: .03, bevelEnabled: false });
  const engineGeo = new THREE.CylinderGeometry(.13, .16, .9, 20); engineGeo.rotateZ(Math.PI / 2);
  const podGeo = new THREE.BoxGeometry(.5, .08, .22);
  function buildShip(material, name, dim = 1) {
    const nozzle = dim === 1 ? nozzleMaterial : nozzleMaterial.clone(), exhaust = dim === 1 ? exhaustMaterial : exhaustMaterial.clone();
    nozzle.uniforms.uDim.value = exhaust.uniforms.uDim.value = dim;
    const g = new THREE.Group(); g.name = name;
    const add = (geo, x, y, z, part, rx = 0, sx = 1) => { const m = new THREE.Mesh(geo, material); m.position.set(x, y, z); m.rotation.x = rx; m.scale.z = sx; m.name = part; g.add(m); return m; };
    add(fuselageGeo, 0, 0, 0, 'Fuselage');
    for (const s of [-1, 1]) {
      const side = s > 0 ? 'R' : 'L';
      add(wingGeo, -.2, -.02, 0, `Wing_${side}`, 0, s);
      add(engineGeo, -.85, -.02, s * .48, `Engine_${side}`);
      add(finGeo, -1.05, .05, s * .62, `Fin_${side}`, s * -.35);
      add(podGeo, -.3, .1, s * 1.05, `Pod_${side}`);
      const noz = new THREE.Mesh(new THREE.PlaneGeometry(.36, .36), nozzle); noz.rotation.y = Math.PI / 2; noz.position.set(-1.31, -.02, s * .48); noz.name = `Nozzle_${side}`; g.add(noz);
      const trail = new THREE.Mesh(new THREE.PlaneGeometry(16, .5), exhaust); trail.name = `Exhaust_${side}`; trail.position.set(-1.3 - 8, -.02, s * .48); g.add(trail);

    }
    const canopy = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 12), metalMaterial(.35)); canopy.name = 'Canopy'; canopy.scale.set(.55, .14, .17); canopy.position.set(.75, .12, 0); g.add(canopy);
    return g;
  }
  const hullMaterial = metalMaterial(1), mirrorMaterial = metalMaterial(.18);
  const ship = buildShip(hullMaterial, 'Vessel'); scene.add(ship);
  const mirror = buildShip(mirrorMaterial, 'VesselReflection', 0); scene.add(mirror);

  function resize(w, h) {
    camera.aspect = w / h; camera.fov = w / h < 1 ? 60 : 40; camera.updateProjectionMatrix(); state.aspect = w / h;
    backdrop.uniforms.uAspect.value = state.aspect; backdrop.uniforms.uH.value = state.horizon = .47;
    const s = state.aspect < 1 ? .28 : .27; ship.scale.setScalar(s); mirror.scale.set(s, -s, s);
  }
  const fwd = new THREE.Vector3(), tmp = new THREE.Vector3(), xAxis = new THREE.Vector3(1, 0, 0), horizonPoint = new THREE.Vector3();
  function update({ time, pointer, motion, local, dt }) {
    const lp = local ?? .4, mobile = state.aspect < 1;
    // Portrait keeps the reference's large arc: the circle centre moves beyond the right edge.
    // Scroll pushes in: the shadow grows, the sea flows faster.
    backdrop.uniforms.uRs.value = (mobile ? .45 : .46) + lp * (mobile ? .02 : .04); backdrop.uniforms.uFlow.value = .08 + lp * .1; backdrop.uniforms.uTime.value = time; backdrop.uniforms.uMotion.value = motion;
    // Camera drift: a slow float on top of the pointer.
    const driftX = Math.sin(time * .13) * .006 * motion, driftY = Math.sin(time * .17 + 1) * .004 * motion;
    state.center.set((mobile ? 1.27 : .64) - pointer.x * .01 * motion + driftX, .47 + pointer.y * .008 * motion + driftY);
    camera.position.set(pointer.x * .25 * motion + Math.sin(time * .13) * .08 * motion, -pointer.y * .15 * motion + Math.sin(time * .17 + 1) * .05 * motion, 10); camera.position.y += .8; camera.lookAt(camera.position.x * .5, .8, 0); camera.updateMatrixWorld();
    for (let i = 0; i < sparkN; i++) {
      const s = sparks[i]; s.z += s.v * (dt || 0) * motion; if (s.z > 8) s.z -= 68;
      const len = .14 + s.v * .025; sp.set([s.x, s.y, s.z, s.x * (1 - len * .02), s.y * (1 - len * .02), s.z - len], i * 6);
    }
    sparkGeo.attributes.position.needsUpdate = true;
    // Ship: right of the centre, just above the sea, heading into the scene; the pointer steers it.
    state.shipTarget.set((mobile ? .8 : 1.45) + pointer.x * 1.2 * motion + Math.sin(time * .4) * .12 * motion, .02 - pointer.y * .25 * motion + Math.sin(time * .9) * .04 * motion, 1.5);
    const prevX = state.ship.x; state.ship.lerp(state.shipTarget, state.ship.lengthSq() ? .06 : 1);
    fwd.set(-.45, .045, -1).normalize();
    ship.quaternion.setFromUnitVectors(xAxis, fwd);
    state.roll += (THREE.MathUtils.clamp(-(state.ship.x - prevX) * 8 - pointer.x * .3 * motion, -.5, .5) - state.roll) * .08;
    ship.rotateX(state.roll - .2);
    // The sea plane at the ship's depth: where the horizon line crosses it. Mirror the ship about it.
    horizonPoint.set(0, state.horizon * 2 - 1, .5).unproject(camera).sub(camera.position);
    const seaY = camera.position.y + horizonPoint.y * (camera.position.z - state.ship.z) / -horizonPoint.z;
    ship.position.set(state.ship.x, seaY + .22 + state.ship.y, state.ship.z);
    mirror.position.set(ship.position.x, 2 * seaY - ship.position.y, ship.position.z); mirror.quaternion.copy(ship.quaternion);
    mirror.quaternion.set(-mirror.quaternion.x, mirror.quaternion.y, -mirror.quaternion.z, mirror.quaternion.w);
    tmp.set(-1.1, 0, 0).applyMatrix4(ship.matrixWorld.compose(ship.position, ship.quaternion, ship.scale)).project(camera); shipUv.set(tmp.x * .5 + .5, tmp.y * .5 + .5);
    exhaustMaterial.uniforms.uPulse.value = nozzleMaterial.uniforms.uPulse.value = .85 + .15 * Math.sin(time * 30 * motion);
  }
  function setPalette(color, name) { gold.set(name === 'ice' ? '#8fc6ff' : '#ffc45a'); hot.set(name === 'ice' ? '#eef6ff' : '#fff1d6'); sparkMaterial.color.copy(gold); }
  function exportGroup() {
    const g = new THREE.Group(); g.name = 'Chapitre_03_Singularite';
    const s = new THREE.Group(); s.name = 'Vessel'; ship.updateMatrixWorld(true);
    for (const c of ship.children) if (c.isMesh && c.material === hullMaterial) s.add(portable(c, { color: 0x6a6e78, metalness: .8, roughness: .3 }));
    g.add(s);
    return g;
  }
  return { name: 'singularity', scene, camera, resize, update, setPalette, exportGroup, post: { ca: 0, bloom: .95, exposure: 1.08, sat: 1.03, flare: .35 } };
}
