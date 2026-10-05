import * as THREE from 'three';
import { rng } from './kit.js';

/**
 * The screen breaks: the previous chapter's frame, cut into Voronoi glass shards, cracks from an impact point and
 * flies apart toward the camera, revealing the next chapter behind. Each shard is a thin slab of glass (bevelled
 * sides, bright bevel line, chromatic refraction of the frame, fresnel glint), like the Prism chapter. One draw call; every shard animates in the
 * vertex shader from the transition progress t (0 = intact frame, 1 = gone), so scrolling back reassembles it.
 *
 * render(renderer, texture, t, time) draws over the current render target (the next chapter already in it).
 */
export function createShatter({ isMobile }) {
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, .1, 100); camera.position.set(0, 0, 10);
  const impact = new THREE.Vector2(.6, .46);

  // Voronoi cells of the unit screen, denser around the impact.
  const r = rng(4242), sites = [], count = isMobile() ? 34 : 64;
  for (let i = 0; i < count; i++) {
    if (i < count * .55) { const a = r() * Math.PI * 2, d = Math.pow(r(), 1.5) * .42; sites.push(new THREE.Vector2(impact.x + Math.cos(a) * d * 1.2, impact.y + Math.sin(a) * d)); }
    else sites.push(new THREE.Vector2(r(), r()));
  }
  const clip = (poly, s, o) => {   // keep the half-plane closer to s than to o
    const m = s.clone().add(o).multiplyScalar(.5), n = o.clone().sub(s), out = [];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length], da = a.clone().sub(m).dot(n), db = b.clone().sub(m).dot(n);
      if (da <= 0) out.push(a);
      if (da * db < 0) out.push(a.clone().lerp(b, da / (da - db)));
    }
    return out;
  };
  const pos = [], screen = [], edge = [], center = [], rand = [], sideN = [], TH = .06;
  for (const s of sites) {
    let poly = [new THREE.Vector2(0, 0), new THREE.Vector2(1, 0), new THREE.Vector2(1, 1), new THREE.Vector2(0, 1)];
    for (const o of sites) if (o !== s && poly.length) poly = clip(poly, s, o);
    if (poly.length < 3) continue;
    const c = poly.reduce((acc, p) => acc.add(p), new THREE.Vector2()).multiplyScalar(1 / poly.length);
    const rv = [r(), r(), r(), r()];
    const vert = (p, z, e, nx = 0, ny = 0) => { pos.push(p.x - c.x, p.y - c.y, z); screen.push(p.x, p.y); edge.push(e); center.push(c.x, c.y); rand.push(...rv); sideN.push(nx, ny); };
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length];
      vert(c, TH, 0); vert(a, TH, 1); vert(b, TH, 1);
      // the broken edge: a bevelled side the thickness of the glass
      const nx = b.y - a.y, ny = a.x - b.x, l = Math.hypot(nx, ny) || 1;
      for (const [p, z] of [[a, TH], [a, -TH], [b, -TH], [a, TH], [b, -TH], [b, TH]]) vert(p, z, 2, nx / l, ny / l);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  geometry.setAttribute('aScreen', new THREE.Float32BufferAttribute(screen, 2));
  geometry.setAttribute('aEdge', new THREE.Float32BufferAttribute(edge, 1));
  geometry.setAttribute('aCenter', new THREE.Float32BufferAttribute(center, 2));
  geometry.setAttribute('aRand', new THREE.Float32BufferAttribute(rand, 4));
  geometry.setAttribute('aSideN', new THREE.Float32BufferAttribute(sideN, 2));

  const material = new THREE.ShaderMaterial({
    uniforms: { tFrame: { value: null }, uT: { value: 0 }, uTime: { value: 0 }, uHalf: { value: new THREE.Vector2(1, 1) }, uImpact: { value: impact } },
    vertexShader: /* glsl */`attribute vec2 aScreen,aCenter,aSideN;attribute float aEdge;attribute vec4 aRand;
      uniform float uT;uniform vec2 uHalf,uImpact;varying vec2 vScreen;varying float vEdge,vCrack,vFade,vFacing;varying vec3 vN;
      mat3 rot(vec3 a,float t){a=normalize(a);float c=cos(t),s=sin(t),k=1.-c;
        return mat3(c+a.x*a.x*k,a.y*a.x*k+a.z*s,a.z*a.x*k-a.y*s, a.x*a.y*k-a.z*s,c+a.y*a.y*k,a.z*a.y*k+a.x*s, a.x*a.z*k+a.y*s,a.y*a.z*k-a.x*s,c+a.z*a.z*k);}
      void main(){vScreen=aScreen;vEdge=aEdge;
        vec2 toC=(aCenter-uImpact)*vec2(uHalf.x/uHalf.y,1.);float d=length(toC);
        vCrack=smoothstep(d*.16,d*.16+.05,uT);                       // cracks run out from the impact
        float p=clamp((uT-.14-d*.22)/.7,0.,1.);p=p*p*(3.-2.*p);       // then shards leave, nearest first
        mat3 R=rot(aRand.xyz-.5,p*(1.5+aRand.w*5.)*(aRand.x>.5?1.:-1.));
        vec3 local=R*vec3(position.xy*uHalf*2.,position.z*smoothstep(0.,.1,uT));
        vec3 c=vec3((aCenter-.5)*uHalf*2.,0.);
        vec2 out2=normalize(toC+1e-4)*(1.2+aRand.y*2.5);
        vec3 move=vec3(out2*p*p*(2.+aRand.z*3.),p*(4.+aRand.w*7.))+vec3(0.,-p*p*2.5,0.);
        vN=R*(aEdge>1.5?vec3(aSideN,0.):vec3(0.,0.,1.));vFacing=abs((R*vec3(0.,0.,1.)).z);vFade=1.-smoothstep(.72,1.,p);
        gl_Position=projectionMatrix*modelViewMatrix*vec4(c+local+move,1.);}`,
    fragmentShader: /* glsl */`uniform sampler2D tFrame;uniform float uT,uTime;varying vec2 vScreen;varying float vEdge,vCrack,vFade,vFacing;varying vec3 vN;
      float sh(vec2 p){return fract(sin(dot(p,vec2(41.3,289.1)))*43758.5453);}
      void main(){vec3 n=normalize(vN);float side=step(1.5,vEdge),face=1.-side;
        float rim=(smoothstep(.95,1.,vEdge)+smoothstep(.82,1.,vEdge)*.3)*face;
        // refraction: the frame seen through tilted glass, the channels split toward the broken edges
        vec2 off=n.xy*.012*(1.-vFacing*.6)*vCrack;float split=(.002+rim*.006)*vCrack;
        vec3 col=vec3(texture2D(tFrame,vScreen+off+vec2(split,0.)).r,texture2D(tFrame,vScreen+off).g,texture2D(tFrame,vScreen+off-vec2(split,0.)).b);
        vec3 lav=vec3(.86,.82,1.);vec3 prism=.5+.5*cos(6.2831*(vScreen.x*1.7+vScreen.y+uTime*.05+vec3(0.,.33,.67)));
        float lit=.6+.4*vFacing;float fres=pow(1.-vFacing,2.);
        float glint=pow(max(dot(n,normalize(vec3(-.4,.6,.7))),0.),24.);
        float spark=step(.995,sh(floor(vScreen*vec2(420.,260.))))*vCrack*face*(1.-vFacing*.5);
        col=col*lit+(lav*1.3*rim+prism*rim*.45)*vCrack+lav*fres*.35*vCrack+vec3(1.)*glint*.9*(1.-vFacing)+lav*spark*1.5;
        col=mix(col,(lav*.45+prism*.3)*(.4+.9*glint+.6*fres)+texture2D(tFrame,vScreen).rgb*.7,side);
        gl_FragColor=vec4(col,vFade);}`,
    side: THREE.DoubleSide, transparent: true, depthTest: true, depthWrite: true
  });
  const shards = new THREE.Mesh(geometry, material); shards.frustumCulled = false; scene.add(shards);

  function resize(w, h) {
    camera.aspect = w / h; camera.updateProjectionMatrix();
    const halfH = 10 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    material.uniforms.uHalf.value.set(halfH * camera.aspect, halfH);
  }
  function render(renderer, texture, t, time) {
    material.uniforms.tFrame.value = texture; material.uniforms.uT.value = t; material.uniforms.uTime.value = time;
    renderer.render(scene, camera);
  }
  function dispose() { geometry.dispose(); material.dispose(); }
  return { resize, render, dispose };
}
