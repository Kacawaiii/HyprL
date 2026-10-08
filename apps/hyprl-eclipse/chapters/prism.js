import * as THREE from 'three';
import { NOISE, VERT_SCREEN, VERT_WORLD, glow, rng, portable } from '../lib/kit.js';

/**
 * Chapitre 04 — Prisme (référence : orbe qui éclate, verre brisé irisé, lignes de lumière, sol de planches).
 * Deep indigo atmosphere; large thin shards of broken glass (irregular polygons: triangles, slivers, trapezoids) with
 * bevelled bright edges, a crushed-glass sparkle inside, fresnel sheen and chromatic refraction of a half-resolution HDR capture of the orb, floor and atmosphere
 * behind them; defocused foreground glass; a nebula orb with a thin bright rim bursting to the right under a pink lens streak; thin light lines;
 * a reflective violet plank floor with a white glare. Scroll breaks the orb further; the pointer turns the cloud.
 */

// Procedural atmosphere, with screen UVs bottom-up. The glass samples the rendered base layer instead.
const PRISM_BG = /* glsl */`uniform vec2 uOrb;uniform float uAspect;uniform vec3 uViolet;
  vec3 prismBg(vec2 uv){vec2 asp=vec2(uAspect,1.);
    vec3 col=mix(vec3(.004,.006,.023),vec3(.0013,.002,.009),smoothstep(.3,.9,uv.y));
    col+=vec3(.024,.009,.10)*exp(-length((uv-vec2(.5,.42))*asp*vec2(.85,1.15))*3.8)*.4;
    col+=uViolet*exp(-length((uv-uOrb)*asp)*5.)*.035;
    col+=vec3(.6,.68,1.)*exp(-length((uv-vec2(.5,-.04))*asp*vec2(1.,1.7))*6.)*.12;
    return col;}`;

/** Convex outline of a broken piece, in a unit box (x across, y along): never a rectangle. */
function outline(kind, r) {
  const j = () => (r() - .5) * .2;
  switch (kind) {
    case 'tri': return [[-.5 + r() * .4, -.5], [.5, -.5 + r() * .5], [(r() - .5) * .8, .5]];
    case 'sliver': return [[-.5, -.5 + r() * .15], [.5, -.5 + r() * .45], [.5 - r() * .3, .5 - r() * .1], [-.5 + r() * .2, .5]];
    case 'trap': return [[-.5, -.5 + j()], [.5, -.5], [.25 + j(), .5], [-.3 + j(), .5 - r() * .2]];
    default: return [[-.5, -.2 + j()], [-.1 + j(), -.5], [.5, -.35 + j()], [.4 + j(), .3], [-.2 + j(), .5]];
  }
}

/**
 * Bakes broken-glass pieces into one geometry. Each piece: a thin slab (front, back, bevelled sides) with
 * aEdge (0 centre → 1 outline on the faces, 2 on the sides), aUv (local, world-scaled, for the sparkle),
 * aCenter, aDir (explosion offset per unit), aSpin (axis, rate), aLook (brightness, seed, blur).
 */
function buildShards(specs) {
  const P = [], N = [], E = [], UV = [], C = [], D = [], S = [], L = [], DOF = [];
  const q = new THREE.Quaternion(), v = new THREE.Vector3(), n = new THREE.Vector3();
  for (const s of specs) {
    const r = rng(s.seed), pts = outline(s.kind, r).map(([x, y]) => [x * s.w, y * s.h]);
    const cx = pts.reduce((a, p) => a + p[0], 0) / pts.length, cy = pts.reduce((a, p) => a + p[1], 0) / pts.length;
    q.copy(s.quat); const th = s.th / 2;let jitter=[0,0,1];
    const push = (x, y, z, nx, ny, nz, e) => {
      v.set(x, y, z).applyQuaternion(q).add(s.center); n.set(nx, ny, nz).applyQuaternion(q).normalize();
      P.push(v.x, v.y, v.z); N.push(n.x, n.y, n.z); E.push(e); UV.push(x, y); C.push(s.center.x, s.center.y, s.center.z);
      D.push(s.dir.x, s.dir.y, s.dir.z); S.push(...s.spin); L.push(...s.look); DOF.push(...jitter);
    };
    for (jitter of (s.blur ? [[0,0,.32],[-1,0,.17],[1,0,.17],[0,-1,.17],[0,1,.17]] : [[0,0,1]])) {
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      push(cx, cy, th, 0, 0, 1, 0); push(a[0]*.94, a[1]*.94, th, 0, 0, 1, 1); push(b[0]*.94, b[1]*.94, th, 0, 0, 1, 1);
      push(cx, cy, -th, 0, 0, -1, 0); push(b[0]*.94, b[1]*.94, -th, 0, 0, -1, 1); push(a[0]*.94, a[1]*.94, -th, 0, 0, -1, 1);
      const ex = b[0] - a[0], ey = b[1] - a[1], len = Math.hypot(ex, ey) || 1, nx = ey / len, ny = -ex / len;
      // Inset faces meet the thick side on an actual sloped bevel, on both sides of the slab.
      for (const sign of [-1, 1]) {
        const edge = [[a[0]*.94,a[1]*.94,th], [a[0],a[1],th*.55], [b[0],b[1],th*.55],
          [a[0]*.94,a[1]*.94,th], [b[0],b[1],th*.55], [b[0]*.94,b[1]*.94,th]];
        for (const [x,y,z] of edge) push(x,y,z*sign,nx,ny,.8*sign,2);
      }
      push(a[0], a[1], th*.55, nx, ny, 0, 2); push(a[0], a[1], -th*.55, nx, ny, 0, 2); push(b[0], b[1], -th*.55, nx, ny, 0, 2);
      push(a[0], a[1], th*.55, nx, ny, 0, 2); push(b[0], b[1], -th*.55, nx, ny, 0, 2); push(b[0], b[1], th*.55, nx, ny, 0, 2);
    }
  }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('aDof',new THREE.Float32BufferAttribute(DOF,3));
  g.setAttribute('position', new THREE.Float32BufferAttribute(P, 3)); g.setAttribute('normal', new THREE.Float32BufferAttribute(N, 3));
  g.setAttribute('aEdge', new THREE.Float32BufferAttribute(E, 1)); g.setAttribute('aUv', new THREE.Float32BufferAttribute(UV, 2));
  g.setAttribute('aCenter', new THREE.Float32BufferAttribute(C, 3)); g.setAttribute('aDir', new THREE.Float32BufferAttribute(D, 3));
  g.setAttribute('aSpin', new THREE.Float32BufferAttribute(S, 4)); g.setAttribute('aLook', new THREE.Float32BufferAttribute(L, 3));
  return g;
}

export function createPrismChapter({ isMobile }) {
  const scene = new THREE.Scene(); scene.name = 'Chapitre_04_Prisme';
  const camera = new THREE.PerspectiveCamera(45, 1, .1, 200); camera.position.set(0, 0, 12); camera.layers.enable(1); camera.layers.enable(2);
  const violet = new THREE.Color('#7a55ff'), orbPos = new THREE.Vector3(-.9, 3, -5);
  const state = { aspect: 1, explode: .5, mobile: false };
  const beamEnds=Array.from({length:6},()=>new THREE.Vector4()),caustics=Array.from({length:4},()=>new THREE.Vector4());
  const orbUv = new THREE.Vector2(.45, .7), common = { uOrb: { value: orbUv }, uAspect: { value: 1 }, uViolet: { value: violet } };
  // World point under a screen position (sx, sy from the top-left, 0..1) at depth z, for the resting camera.
  const at = (sx, sy, z) => { const h = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * (12 - z); return new THREE.Vector3((sx * 2 - 1) * h * state.aspect, (1 - sy * 2) * h, z); };
  const screenH = z => 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * (12 - z);

  const bgMaterial = new THREE.ShaderMaterial({
    uniforms: { ...common, uBeams:{value:beamEnds}, uTime: { value: 0 } }, vertexShader: VERT_SCREEN, depthWrite: false, depthTest: false,
    fragmentShader: NOISE + PRISM_BG + /* glsl */`varying vec2 vUv;uniform float uTime;uniform vec4 uBeams[6];
      void main(){vec3 col=prismBg(vUv);
        vec2 p=vUv*vec2(uAspect,1.),o=uOrb*vec2(uAspect,1.);
        float haze=fbm3(vUv*vec2(14.*uAspect,12.)+vec2(uTime*.013,0.));
        for(int i=0;i<6;i++){vec2 end=uBeams[i].xy*vec2(uAspect,1.),d=end-o;float t=dot(p-o,d)/max(dot(d,d),.001);
          float dist=length(p-o-d*t),width=.004+max(t,0.)*.027;
          float ray=exp(-pow(dist/width,2.))*smoothstep(0.,.12,t)*smoothstep(1.7,.7,t);
          col+=mix(uViolet,vec3(.35,.8,1.),float(i)*.19)*ray*(.2+haze*.8)*uBeams[i].w*.19;}
        float n=fbm(vUv*vec2(3.*uAspect,2.)+vec2(uTime*.01,0.));col+=uViolet*pow(n,3.)*.015;gl_FragColor=vec4(col,1.);}`
  });
  const bg = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), bgMaterial); bg.name = 'Backdrop'; bg.frustumCulled = false; bg.renderOrder = -10; scene.add(bg);

  // Reflective plank floor: boards running into the distance, staggered butt joints, grain, the glare near the camera.
  const floorMaterial = new THREE.ShaderMaterial({
    uniforms: { uViolet: { value: violet }, uCaustics:{value:caustics}, uTime: { value: 0 } },
    vertexShader: VERT_WORLD, transparent: true, depthWrite: false,
    fragmentShader: NOISE + /* glsl */`varying vec3 vW;uniform vec3 uViolet;uniform float uTime;uniform vec4 uCaustics[4];
      void main(){float bw=1.25;float id=floor(vW.x/bw);float fx=fract(vW.x/bw);
        float seam=1.-smoothstep(.0,.035,min(fx,1.-fx));
        float jz=fract(vW.z*.07+hash12(vec2(id,3.)));float joint=1.-smoothstep(0.,.012,min(jz,1.-jz));
        float tone=.75+.5*hash12(vec2(id,9.));float grain=.75+.25*fbm(vec2(vW.x*7.,vW.z*.35));
        vec3 v=normalize(cameraPosition-vW);float fres=pow(clamp(1.-v.y,0.,1.),4.);
        vec3 col=vec3(.013,.015,.055)*tone*grain*(.3+fres*.7);
        float glare=exp(-length(vec2(vW.x*.32,(vW.z+1.)*.16)));float centre=exp(-abs(vW.x)*.18)*exp(-abs(vW.z+6.)*.12);
        col+=mix(vec3(.3,.3,1.),vec3(.9,.92,1.),.3*glare)*(glare*.35+centre*.09)*(.45+.55*grain)*tone;
        col+=vec3(.12,.17,.55)*glare*(.6+.8*tone)+vec3(.65,.7,1.)*pow(glare,2.)*2.;
        // Moving glass caustics: ray/plane footprints from the animated hero shards.
        for(int i=0;i<4;i++){vec2 q=vW.xz-uCaustics[i].xy;float a=uCaustics[i].z;mat2 R=mat2(cos(a),-sin(a),sin(a),cos(a));q=R*q;
          float footprint=exp(-dot(q*vec2(.3,.7),q*vec2(.3,.7)));
          float fil=pow(abs(sin(q.x*4.+sin(q.y*2.+uTime*.08)*2.)),16.);
          vec3 rainbow=.5+.5*cos(q.x*2.+vec3(0.,2.1,4.2));
          col+=(vec3(.35,.44,1.)*fil+rainbow*.12)*footprint*uCaustics[i].w*.85;}
        col=col*(1.-.75*(seam+joint))+vec3(.6,.65,1.)*seam*glare*.25;
        float fade=smoothstep(-30.,-4.,vW.z);
        gl_FragColor=vec4(max(col,0.),fade);}`
  });
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(80, 60), floorMaterial); floor.name = 'PlankFloor'; floor.rotation.x = -Math.PI / 2; floor.position.set(0, -4.6, -14); scene.add(floor);

  // The orb: dark glass with a thin bright rim, a nebula swirling inside, its shell breaking away on the right.
  const orbMaterial = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uViolet: { value: violet }, uCrack: { value: .5 } },
    vertexShader: /* glsl */`varying vec3 vP;varying vec3 vN;varying vec3 vV;void main(){vP=position;vec4 mv=modelViewMatrix*vec4(position,1.);vN=normalize(normalMatrix*normal);vV=normalize(-mv.xyz);gl_Position=projectionMatrix*mv;}`,
    fragmentShader: NOISE + /* glsl */`varying vec3 vP;varying vec3 vN;varying vec3 vV;uniform float uTime,uCrack;uniform vec3 uViolet;
      vec3 h3(vec3 p){p=vec3(dot(p,vec3(127.1,311.7,74.7)),dot(p,vec3(269.5,183.3,246.1)),dot(p,vec3(113.5,271.9,124.6)));return fract(sin(p)*43758.5453);}
      void main(){vec3 n=normalize(vN);float c=max(dot(n,normalize(vV)),0.);
        vec3 p=vP*2.;vec3 i=floor(p),f=fract(p);float d1=8.,d2=8.;vec3 cell=vec3(0.);
        for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++)for(int z=-1;z<=1;z++){vec3 g=vec3(x,y,z);vec3 o=h3(i+g);float d=length(g+o-f);if(d<d1){d2=d1;d1=d;cell=i+g;}else if(d<d2)d2=d;}
        // shell pieces missing on the bursting side
        float gone=step(h3(cell).x,uCrack*1.1-.15)*smoothstep(.0,.5,n.x);if(gone>.5)discard;
        float crack=1.-smoothstep(0.,.05,d2-d1);
        vec2 q=n.xy;float rr=length(q);float ang=atan(q.y,q.x)+(1.-rr)*3.2+uTime*.06;
        float neb=fbm(vec2(cos(ang),sin(ang))*1.6+rr*3.+uTime*.02);float neb2=fbm(q*5.+neb*2.);
        float band=exp(-pow((q.y+.25-.25*q.x)/.3,2.));
        vec3 inner=mix(uViolet*.7,vec3(1.,.82,1.),smoothstep(.4,.8,neb2))*pow(neb,1.6)*band*3.;
        float rim=pow(1.-c,10.)*4.+pow(1.-c,3.)*.2;
        vec3 col=vec3(.015,.01,.05)+inner*.27+mix(uViolet,vec3(.9,.88,1.),.6)*rim*(.6+.6*smoothstep(.3,-.6,n.x+n.y*.3))+vec3(.8,.75,1.)*crack*smoothstep(.4,1.,uCrack)*.12*pow(1.-c,2.);
        gl_FragColor=vec4(col,1.);}`
  });
  const orb = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 64), orbMaterial); orb.name = 'Orb'; scene.add(orb);
  const haloMaterial = glow(/* glsl */`varying vec2 vUv;uniform vec3 uViolet;uniform float uPower,uCrack;void main(){float d=length(vUv-.5)*2.;float ring=exp(-pow((d-.5)*28.,2.))*.5+exp(-pow((d-.5)*5.,2.))*.12*step(.5,d);ring*=1.-smoothstep(.3,.9,uCrack)*smoothstep(.52,.86,vUv.x)*.8;gl_FragColor=vec4(mix(uViolet,vec3(1.),.35)*ring*uPower*smoothstep(1.,.7,d),1.);}`,
    { uViolet: { value: violet }, uPower: { value: 1 }, uCrack:{value:.5} }, { depthTest: false });
  const halo = new THREE.Mesh(new THREE.PlaneGeometry(4, 4), haloMaterial); halo.name = 'OrbHalo'; halo.renderOrder = 3; scene.add(halo);

  // Broken glass: one material for every piece.
  const shardMaterial = new THREE.ShaderMaterial({
    uniforms: { ...common, tFrame: { value: null }, uFrameTexel: { value: new THREE.Vector2(1, 1) }, uTime: { value: 0 }, uExplode: { value: .5 }, uMotion: { value: 1 }, uResolution:{value:new THREE.Vector2(1,1)} },
    vertexShader: /* glsl */`attribute vec3 aDof;uniform vec2 uResolution;varying float vDofWeight;attribute float aEdge;attribute vec2 aUv;attribute vec3 aCenter,aDir,aLook;attribute vec4 aSpin;uniform float uTime,uExplode,uMotion;
      varying vec3 vW;varying vec3 vN;varying float vEdge;varying vec2 vUv;varying vec3 vLook;varying vec4 vClip;
      mat3 rot(vec3 a,float t){a=normalize(a);float c=cos(t),s=sin(t),k=1.-c;
        return mat3(c+a.x*a.x*k,a.y*a.x*k+a.z*s,a.z*a.x*k-a.y*s, a.x*a.y*k-a.z*s,c+a.y*a.y*k,a.z*a.y*k+a.x*s, a.x*a.z*k+a.y*s,a.y*a.z*k-a.x*s,c+a.z*a.z*k);}
      void main(){mat3 R=rot(aSpin.xyz,aSpin.w*(uTime*uMotion+uExplode*3.));
        vec3 p=aCenter+aDir*uExplode+R*(position-aCenter);vec4 w=modelMatrix*vec4(p,1.);
        vW=w.xyz;vN=normalize(mat3(modelMatrix)*(R*normal));vEdge=aEdge;vUv=aUv;vLook=aLook;gl_Position=projectionMatrix*viewMatrix*w;vClip=gl_Position;
        float defocus=clamp(abs(length(cameraPosition-vW)-17.)*.42,.65,4.);
        gl_Position.xy+=aDof.xy*defocus*2./uResolution*gl_Position.w;vDofWeight=aDof.z;}`,
    fragmentShader: NOISE + PRISM_BG + /* glsl */`varying float vDofWeight;uniform float uTime;uniform sampler2D tFrame;uniform vec2 uFrameTexel;varying vec3 vW;varying vec3 vN;varying float vEdge;varying vec2 vUv;varying vec3 vLook;varying vec4 vClip;
      void main(){vec3 n=normalize(vN);vec3 v=normalize(cameraPosition-vW);if(dot(n,v)<0.)n=-n;
        float c=clamp(dot(n,v),0.,1.),fres=.0426+.9574*pow(1.-c,5.);float side=step(1.5,vEdge),bevel=smoothstep(.965,.998,vEdge)*(1.-side);
        // Refraction of the actual rendered base layer, split per channel; capture excludes glass and light overlays.
        vec2 suv=vClip.xy/vClip.w*.5+.5;
        // Dielectric glass: IOR 1.52, a longer optical path at grazing angles, indigo absorption.
        vec3 bent=refract(-v,n,1./1.52);float path=.22/max(c,.18);
        vec2 off=(mat3(viewMatrix)*(bent+v)).xy*(.035+.025*path);
        float dispersion=.0016+.0045*bevel+.0025*side;vec2 fringe=normalize(off+vec2(.002,.001))*dispersion;
        vec2 sampleUv=clamp(suv-off,uFrameTexel,1.-uFrameTexel);
        vec3 refr=vec3(texture2D(tFrame,clamp(sampleUv+fringe,uFrameTexel,1.-uFrameTexel)).r,
          texture2D(tFrame,sampleUv).g,texture2D(tFrame,clamp(sampleUv-fringe,uFrameTexel,1.-uFrameTexel)).b);
        refr*=exp(-vec3(.12,.18,.055)*path);
        vec3 lav=vec3(.84,.8,1.);float B=vLook.x;
        // broad sheen across the face, frosted cloud and crushed-glass sparkle inside
        float sheen=pow(max(dot(reflect(-v,n),normalize(vec3(-.35,-.5,.8))),0.),2.)+pow(max(dot(reflect(-v,n),normalize(vec3(.4,.6,.7))),0.),6.)*.6;
        float frost=fbm3(vUv*3.+vLook.y*17.);
        vec2 su=vUv/max(vLook.z,.05)*vec2(70.,90.)+vLook.y*31.;vec2 g=floor(su);float h=hash12(g);float tw=.5+.5*sin(uTime*2.5+h*80.+dot(v,vec3(23.,17.,11.)));
        float sparkle=step(.965-.03*B,h)*smoothstep(.45,.0,length(fract(su)-.5))*tw*tw*1.4;
        float grad=smoothstep(-.6,.8,dot(vUv,vec2(.35,.6))/max(length(vUv)+.4,.4)+frost*.5);
        float along=clamp(vUv.y/max(vLook.z,.01)+.5,0.,1.);along=fract(vLook.y*7.)>.5?along:1.-along;
        float hot=clamp(smoothstep(.2,1.,along)*.75+sheen*.5+grad*.2,0.,1.)*(.25+.9*frost);vec3 body=mix(uViolet*.45,lav,hot)*(.35+.65*frost)*B*B*(.18+.95*hot);
        vec3 film=.5+.5*cos(6.2831*(c*1.6+vLook.y+vec3(0.,.33,.67)));
        vec3 col=refr*(.78+.22*B)*(1.-fres*.5)+body*.92+lav*sparkle*(.1+4.2*B*B)*(.3+.7*hot)*(1.-side);
        float edgePower=B*B*mix(.12,1.,smoothstep(.5,.75,B));
        col+=(vec3(1.)*1.6+film*.4)*bevel*(.006+.30*edgePower)+(lav*1.3+film*.5)*side*(.004+.48*edgePower);
        float crushed=fbm3(vUv*13.+vLook.y*23.);
        float caustic=pow(frost,3.)*3.+pow(abs(sin(vUv.x*6.+vnoise(vUv*4.+vLook.y*19.)*6.+vUv.y*3.)),28.)*.22;
        col*=.78+.38*crushed;
        col+=mix(lav,film,.28)*caustic*B*B*(.08+hot*.55);
        vec3 light=normalize(vec3(-.4+sin(uTime*.17)*.12,.6,.7));
        float edgeGlint=pow(max(dot(reflect(-light,n),v),0.),36.);
        float movingGlint=pow(max(dot(reflect(normalize(vW-vec3(-1.,4.,-4.)),n),v),0.),64.);
        col+=vec3(1.,.92,1.)*movingGlint*(.08+B*B*2.2);
        col+=lav*edgeGlint*side*(.05+B*B*2.);
        col+=mix(lav,film,.3)*fres*(.1+.6*B);
        gl_FragColor=vec4(col,mix(.50+.3*hot*B,.93,max(bevel,side))*vDofWeight);}`,
    transparent: true, depthWrite: false, side: THREE.DoubleSide
  });

  const reflectionMaterial=new THREE.ShaderMaterial({uniforms:shardMaterial.uniforms,
    vertexShader:shardMaterial.vertexShader.replace('vW=w.xyz;', 'w.y=-9.2-w.y;w.z-=max(0.,-w.y-4.6)*1.6;vW=w.xyz;'),
    fragmentShader:NOISE+/* glsl */`varying vec3 vW;varying vec3 vN;varying float vEdge;varying vec2 vUv;varying vec3 vLook;uniform float uTime;
      void main(){vec3 n=normalize(vN),v=normalize(cameraPosition-vW);float glint=pow(abs(dot(n,v)),8.);
        float ripple=.5+.5*sin(vW.z*21.+vnoise(vW.xz*2.)*4.);
        vec3 c=mix(vec3(.12,.05,.5),vec3(.5,.68,1.),glint);
        c*=.35+fbm3(vUv*4.+vLook.y)*.65;
        float a=smoothstep(-16.,-5.,vW.y)*(.19+.09*ripple)*vLook.x;
        gl_FragColor=vec4(c*(.5+glint),a);}`,
    transparent:true,depthWrite:false,side:THREE.DoubleSide});

  // Pieces flung by the orb: small, mostly bursting to the right.
  const fragN = isMobile() ? 26 : 54, r = rng(303), fragSpecs = [];
  const kinds = ['tri', 'sliver', 'trap', 'penta'];
  for (let i = 0; i < fragN; i++) {
    const dir = new THREE.Vector3(.35 + r() * 1.1, (r() - .45) * .9, (r() - .5) * .8).normalize(), d0 = 1.05 + r() * 1.4, size = .06 + Math.pow(r(), 3) * .3;
    fragSpecs.push({ blur:!isMobile(), kind: kinds[i % 4], seed: 100 + i, w: size * (.3 + r() * .6), h: size * (1 + r()), th: .02 + size * .05,
      center: dir.clone().multiplyScalar(d0), quat: new THREE.Quaternion().setFromEuler(new THREE.Euler(r() * 6, r() * 6, r() * 6)),
      dir: dir.clone().multiplyScalar(1.5 + r() * 3.5), spin: [r() - .5, r() - .5, r() - .5, (r() - .5) * .8], look: [.55 + r() * .45, r(), size * 2] });
  }
  const fragGeo = buildShards(fragSpecs);
  const frags = new THREE.Mesh(fragGeo, shardMaterial); frags.name = 'OrbFragments'; frags.layers.set(1); frags.frustumCulled = false;
  const cloud = new THREE.Group(); cloud.name = 'ShardCloud'; scene.add(cloud); cloud.add(frags);

  // The large pieces, placed on the frame like the reference: [x, y (from the top), depth, length, width (screen
  // heights), angle on screen (deg), tilt, brightness, kind]. A: the huge bright sliver bottom left with the glare.
  const pieces = [
    [.35, .69, 5.5, .72, .105, -31, .25, 1, 'sliver'], [.65, .67, 3, .40, .075, 24, -.4, .95, 'sliver'], [.245, .45, 1.5, .17, .035, 78, .4, .85, 'sliver'],
    [.585, .33, -1, .27, .05, 96, -.3, 1, 'sliver'], [.11, .15, 7, .16, .08, -32, .5, .6, 'trap'], [.47, .05, 7, .15, .05, 68, -.4, .6, 'tri'],
    [.73, .05, 6, .22, .035, 32, .3, .7, 'sliver'], [.43, .49, -2.5, .30, .15, 35, .72, .4, 'tri'], [.65, .48, -2, .28, .15, -28, -.9, .32, 'tri'],
     [.96, .4, 5, .55, .13, 98, .5, .22, 'sliver'], [.9, .9, 5, .16, .07, 22, -.6, .35, 'tri'], [.04, .84, 6, .3, .08, 58, .6, .35, 'sliver'],
    [.31, .24, 1, .1, .05, -60, .7, .7, 'tri'], [.8, .27, 2, .09, .04, 40, -.7, .65, 'tri']
  ];
  const heroSpecs=[];
  let reflectionMesh=null;
  const mobilePieces = new Set([0, 1, 3, 4, 7, 9, 11]);
  let piecesMesh = null;
  function buildPieces() {
    if (piecesMesh) { piecesMesh.geometry.dispose(); scene.remove(piecesMesh); }
    const rr = rng(77), specs = [];heroSpecs.length=0;
    pieces.forEach(([sx, sy, z, len, wid, ang, tilt, bright, kind], i) => {
      if ([4, 5, 9].includes(i)) return; // these near pieces use the defocused silhouette shader
      if (state.mobile && !mobilePieces.has(i)) return;
      const H = screenH(z), x = state.mobile ? .5 + (sx - .5) * .8 : sx;
      const quat = new THREE.Quaternion().setFromEuler(new THREE.Euler(tilt * .35, tilt, THREE.MathUtils.degToRad(ang - 90)));
      const center = at(state.mobile && i === 3 ? .76 : x, sy, z);
      specs.push({ blur:!state.mobile&&(i===7||i===8), kind, seed: 500 + i, w: wid * H, h: len * H, th: .05 + wid * H * .06, center, quat,
        dir: new THREE.Vector3(center.x * .04, center.y * .03, .2), spin: [rr() - .5, rr() - .5, rr() * .3, (rr() - .5) * .04], look: [bright, rr(), len * H] });
    });
    heroSpecs.push(...specs.slice(0,4));
    if(reflectionMesh){scene.remove(reflectionMesh);}
    piecesMesh = new THREE.Mesh(buildShards(specs), shardMaterial); piecesMesh.name = 'GlassShards'; piecesMesh.layers.set(1); piecesMesh.frustumCulled = false; piecesMesh.renderOrder = 2; scene.add(piecesMesh);
    reflectionMesh=new THREE.Mesh(piecesMesh.geometry,reflectionMaterial);reflectionMesh.name='GlassFloorReflections';reflectionMesh.frustumCulled=false;reflectionMesh.renderOrder=1;scene.add(reflectionMesh);
  }

  // Debris flung from the orb: bright dust and dark grit.
  const dn = isMobile() ? 400 : 1100, dpos = new Float32Array(dn * 3), ddir = [];
  for (let i = 0; i < dn; i++) ddir.push({ v: new THREE.Vector3(.2 + r() * 1.2, (r() - .5) * .7, (r() - .5) * .7).normalize(), d: 1 + r() * 2.5, s: .5 + r() * 3 });
  const debrisGeo = new THREE.BufferGeometry(); debrisGeo.setAttribute('position', new THREE.BufferAttribute(dpos, 3));
  const debris = new THREE.Points(debrisGeo, new THREE.ShaderMaterial({uniforms:{uTime:{value:0},uMotion:{value:1},uPR:{value:1},uPower:{value:1}},
    vertexShader:/* glsl */`uniform float uTime,uMotion,uPR;varying float vLight,vBlur;void main(){vec4 p=modelViewMatrix*vec4(position,1.);gl_Position=projectionMatrix*p;
      float seed=fract(sin(dot(position,vec3(12.3,78.2,32.7)))*43217.);vLight=pow(.5+.5*sin(seed*80.+uTime*uMotion*(.6+seed)),6.);
      vBlur=clamp(abs(-p.z-17.)/18.,0.,1.);gl_PointSize=uPR*(1.2+seed*2.5+vBlur*5.)*clamp(17./-p.z,.6,2.);}`,
    fragmentShader:/* glsl */`varying float vLight,vBlur;uniform float uPower;void main(){float r=length(gl_PointCoord-.5);float a=exp(-r*r*mix(40.,10.,vBlur))*smoothstep(.5,.35,r);
      gl_FragColor=vec4(mix(vec3(.32,.56,1.),vec3(1.,.88,1.),vLight)*(.25+vLight*2.),a*uPower);}`,
    transparent:true,blending:THREE.AdditiveBlending,depthWrite:false})); debris.name = 'Debris'; debris.frustumCulled = false; cloud.add(debris);
  const grit = new THREE.Points(debrisGeo, new THREE.PointsMaterial({ color: 0x07051a, size: .055, transparent: true, opacity: .9, depthWrite: false })); grit.name = 'Grit'; grit.frustumCulled = false;  cloud.add(grit);

  const airGeo=new THREE.BufferGeometry(),airPos=[];
  for(let i=0;i<(isMobile()?90:260);i++)airPos.push((r()-.5)*28,(r()-.5)*16,-16+r()*22);
  airGeo.setAttribute('position',new THREE.Float32BufferAttribute(airPos,3));
  const air=new THREE.Points(airGeo,debris.material.clone());air.name='SparklingGlassDust';air.material.uniforms.uPower.value=.22;air.frustumCulled=false;scene.add(air);

  // Screen-space light: thin crisp light lines, the pink anamorphic streak across the orb, the glare at the bottom.
  const LINES = [[.24, 0, .62, 1], [0, .69, 1, .13], [.56, 0, .34, 1], [.3, .37, .9, .49], [0, .54, .72, .36], [.12, .64, .88, .84]];
  const overlayMaterial = new THREE.ShaderMaterial({
    uniforms: { ...common, uTime: { value: 0 }, uLines: { value: LINES.map(l => new THREE.Vector4(l[0], 1 - l[1], l[2], 1 - l[3])) }, uRes: { value: new THREE.Vector2(1, 1) }, uFlare: { value: 1 } },
    vertexShader: VERT_SCREEN, depthWrite: false, depthTest: false, transparent: true, blending: THREE.AdditiveBlending,
    fragmentShader: NOISE + /* glsl */`varying vec2 vUv;uniform vec4 uLines[6];uniform vec2 uOrb,uRes;uniform float uAspect,uTime,uFlare;uniform vec3 uViolet;
      void main(){vec2 asp=vec2(uAspect,1.);vec3 col=vec3(0.);
        for(int i=0;i<6;i++){vec2 a=uLines[i].xy*asp,b=uLines[i].zw*asp,p=vUv*asp;vec2 ab=b-a;float t=clamp(dot(p-a,ab)/dot(ab,ab),0.,1.);
          float d=length(p-a-ab*t)*uRes.y;float k=.55+.45*sin(uTime*.7+float(i)*2.1);
          float fade=smoothstep(0.,.25,t)*smoothstep(1.,.75,t)*.6+.4;
          col+=mix(vec3(.75,.7,1.),vec3(1.),.5)*(exp(-d*d*1.1)*.32+exp(-d*.7)*.015)*k*fade;}
        vec2 o=(vUv-uOrb)*asp;
        col+=vec3(1.,.55,.85)*exp(-abs(o.y)*190.)*exp(-abs(o.x)*9.)*2.1*uFlare+vec3(1.,.85,.95)*exp(-abs(o.y)*600.)*exp(-abs(o.x)*9.)*1.8*uFlare;
        col+=vec3(.85,.4,1.)*exp(-abs(o.y)*45.)*exp(-abs(o.x)*5.)*.25*uFlare;
        float burst=exp(-pow((o.y+.008)/.066,2.))*exp(-pow((o.x+.075)/.25,2.));
        float dust=fbm3(o*vec2(22.,35.)+uTime*.03);
        col+=vec3(.8,.7,1.)*burst*pow(dust,2.)*1.2*uFlare;
        vec2 g=(vUv-vec2(.51,-.015))*asp;
        col+=vec3(.9,.94,1.)*exp(-dot(g*vec2(1.,1.25),g*vec2(1.,1.25))*31.)*3.6;
        col+=vec3(.12,.16,.6)*exp(-length(g)*3.8)*.35;
        // Chromatic spill from the foreground glare across the boards and the lower lens edge.
        col+=vec3(.015,.17,.8)*exp(-pow((g.x+.34)/.18,2.)-pow((g.y+.025)/.085,2.))*.55;
        col+=vec3(.65,.03,.28)*exp(-pow((g.x-.38)/.15,2.)-pow((g.y+.015)/.065,2.))*.5;
        col+=vec3(.015,.18,.7)*exp(-abs(vUv.y)*100.)*.35;
        gl_FragColor=vec4(col,1.);}`
  });
  const overlay = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), overlayMaterial); overlay.name = 'LightLines'; overlay.layers.set(2); overlay.frustumCulled = false; overlay.renderOrder = 10; scene.add(overlay);

  const foregroundMaterial = new THREE.ShaderMaterial({
    uniforms: { ...common, uTime: { value: 0 }, uMotion: { value: 1 }, uPointer:{value:new THREE.Vector2()} }, vertexShader: VERT_SCREEN,
    depthTest: false, depthWrite: false, transparent: true, blending: THREE.AdditiveBlending,
    fragmentShader: /* glsl */`varying vec2 vUv;uniform float uTime,uMotion,uAspect;uniform vec2 uPointer;
      float shard(vec2 p,vec2 size,float angle,float blur){float co=cos(angle),si=sin(angle);p=mat2(co,-si,si,co)*p;
        vec2 a=vec2(-size.x,-size.y)*.5,b=vec2(size.x,-size.y*.3)*.5,c=vec2(size.x*.7,size.y)*.5,d=vec2(-size.x*.85,size.y*.65)*.5;
        vec2 e=b-a,f=c-b,g=d-c,h=a-d;
        float sd=max(max(dot(p-a,normalize(vec2(e.y,-e.x))),dot(p-b,normalize(vec2(f.y,-f.x)))),
          max(dot(p-c,normalize(vec2(g.y,-g.x))),dot(p-d,normalize(vec2(h.y,-h.x)))));
        return 1.-smoothstep(-blur,blur,sd);}
      vec3 nearShard(vec2 p,vec2 size,float angle,float blur,float power){
        size.x*=min(1.,uAspect/.85);
        vec2 split=vec2(blur*.55,blur*.18);
        float blue=shard(p+split,size,angle,blur),red=shard(p-split,size,angle,blur),core=shard(p,size,angle,blur*1.2);
        return (vec3(0.,.65,1.)*blue+vec3(1.,.04,.22)*red+vec3(1.)*core*.4)*power;}
      void main(){vec2 p=(vUv+uPointer*vec2(-.018,.012)*uMotion)*vec2(uAspect,1.);float drift=sin(uTime*.18)*.009*uMotion;vec3 col=vec3(0.);
        col+=nearShard(p-vec2(uAspect*.10,.85+drift),vec2(.19,.065),.3,.018,.8);
        col+=nearShard(p-vec2(uAspect*.47,1.01-drift),vec2(.09,.23),-.6,.014,1.25);
        col+=nearShard(p-vec2(uAspect*1.015,.52+drift),vec2(.19,.5),.25,.034,.32*smoothstep(.3,1.2,uAspect));
        col+=nearShard(p-vec2(uAspect*.055,-.03),vec2(.2,.34),.65,.033,.22);
        gl_FragColor=vec4(col,1.);}`
  });
  const foreground = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), foregroundMaterial);
  foreground.name = 'DefocusedForegroundGlass'; foreground.layers.set(2); foreground.frustumCulled = false; foreground.renderOrder = 11; scene.add(foreground);

  const tmp = new THREE.Vector3(), beamAxis = new THREE.Vector3(), beamCenter = new THREE.Vector3(), beamNormal = new THREE.Vector3(), beamScreen = new THREE.Vector3(), beamRay = new THREE.Vector3(), beamHit = new THREE.Vector3(), beamRotation = new THREE.Quaternion();
  function resize(w, h) {
    camera.aspect = w / h; camera.updateProjectionMatrix(); state.aspect = w / h; state.mobile = w < 600;
    common.uAspect.value = state.aspect;shardMaterial.uniforms.uResolution.value.set(w,h); overlayMaterial.uniforms.uRes.value.set(w, h);
    camera.position.set(0, 0, 12); camera.lookAt(0, 0, 0); camera.updateMatrixWorld();
    orbPos.copy(at(state.mobile ? .5 : .45, state.mobile ? .28 : .3, -5));
    const radius = (state.mobile ? .1 : .11) * screenH(-5) * (state.mobile ? state.aspect * 1.6 : 1);
    orb.position.copy(orbPos); orb.scale.setScalar(radius); halo.position.copy(orbPos); halo.scale.setScalar(radius); cloud.position.copy(orbPos); cloud.scale.setScalar(radius / 1.35);
    tmp.copy(orbPos).project(camera); orbUv.set(tmp.x * .5 + .5, tmp.y * .5 + .5);
    buildPieces();
  }
  function update({ time, pointer, motion, local, pr }) {
    const lp01 = local ?? .5; state.explode = .2 + lp01 * .75 + Math.sin(time * .25) * .04 * motion;
    camera.position.set(pointer.x * .5 * motion + Math.sin(time * .11) * .1 * motion, (-pointer.y * .35+Math.sin(time*.09)*.07) * motion, 12 - lp01 * 1.2); camera.lookAt(pointer.x * .2 * motion, 0, 0);
    camera.updateMatrixWorld();
    tmp.copy(orbPos).project(camera);orbUv.set(tmp.x*.5+.5,tmp.y*.5+.5);
    heroSpecs.forEach((s,i)=>{
      beamAxis.set(s.spin[0],s.spin[1],s.spin[2]).normalize();
      beamRotation.setFromAxisAngle(beamAxis,s.spin[3]*(time*motion+state.explode*3.));
      beamCenter.copy(s.center).addScaledVector(s.dir,state.explode);
      beamNormal.set(0,0,1).applyQuaternion(s.quat).applyQuaternion(beamRotation);
      beamScreen.copy(beamCenter).project(camera);beamRay.copy(orbPos).sub(beamCenter).normalize();
      const power=.5+Math.pow(Math.abs(beamNormal.dot(beamRay)),3.);
      beamEnds[i].set(beamScreen.x*.5+.5,beamScreen.y*.5+.5,0,power);
      beamRay.copy(beamCenter).sub(orbPos).normalize();beamRay.y=-Math.max(.35,Math.abs(beamRay.y)*1.8+.3);beamRay.z-=.65;beamRay.normalize();
      const t=(-4.6-beamCenter.y)/(beamRay.y||.001);beamHit.copy(beamCenter).addScaledVector(beamRay,Math.max(0,t));
      caustics[i].set(beamHit.x,beamHit.z,Math.atan2(beamNormal.x,beamNormal.z),power);
    });
    beamEnds[4].set(.10+pointer.x*.018*motion,.85-pointer.y*.012*motion+Math.sin(time*.18)*.009*motion,0,.9);beamEnds[5].set(.47+pointer.x*.018*motion,1.01-pointer.y*.012*motion-Math.sin(time*.18)*.009*motion,0,.8);
    air.rotation.y=time*.004*motion;air.material.uniforms.uTime.value=time;air.material.uniforms.uMotion.value=motion;air.material.uniforms.uPR.value=pr??1;
    debris.material.uniforms.uTime.value=time;debris.material.uniforms.uMotion.value=motion;debris.material.uniforms.uPR.value=pr??1;
    cloud.rotation.set(pointer.y * .2 * motion, pointer.x * .35 * motion + Math.sin(time * .05) * .1 * motion, 0);
    orb.rotation.y = time * .05 * motion; orbMaterial.uniforms.uCrack.value = state.explode; orbMaterial.uniforms.uTime.value = time;
    for (const m of [shardMaterial, bgMaterial, overlayMaterial, floorMaterial, foregroundMaterial]) m.uniforms.uTime.value = time;
    shardMaterial.uniforms.uExplode.value = state.explode; shardMaterial.uniforms.uMotion.value = motion; foregroundMaterial.uniforms.uMotion.value = motion;foregroundMaterial.uniforms.uPointer.value.set(pointer.x,pointer.y);
    haloMaterial.uniforms.uCrack.value=state.explode;haloMaterial.uniforms.uPower.value = .6 + state.explode * .5; overlayMaterial.uniforms.uFlare.value = .7 + state.explode * .5;
    for (let i = 0; i < dn; i++) { const d = ddir[i], k = d.d + state.explode * d.s; dpos[i * 3] = d.v.x * k; dpos[i * 3 + 1] = d.v.y * k + Math.sin(time * .4 + i) * .03 * motion; dpos[i * 3 + 2] = d.v.z * k; }
    debrisGeo.attributes.position.needsUpdate = true;
  }
  function setPalette() {}
  function exportGroup() {
    const g = new THREE.Group(); g.name = 'Chapitre_04_Prisme';
    g.add(portable(orb, { color: 0x1a1240, emissive: 0x6f4dff, emissiveIntensity: .25, roughness: .15, metalness: .1 }));
    const glass = { color: 0xb9aaff, transparent: true, opacity: .55, roughness: .05, metalness: 0 };
    // Glass as plain geometry (position + normal), at rest.
    const lean = m => {
      // Aperture copies are a raster effect; export one optical surface per shard.
      const pos=[],norm=[],a=m.geometry.attributes;
      for(let i=0;i<a.position.count;i++){if(a.aDof.getZ(i)<.3)continue;
        pos.push(a.position.getX(i),a.position.getY(i),a.position.getZ(i));
        norm.push(a.normal.getX(i),a.normal.getY(i),a.normal.getZ(i));}
      const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));geo.setAttribute('normal',new THREE.Float32BufferAttribute(norm,3));
      const c=new THREE.Mesh(geo,m.material);c.name=m.name;m.parent.add(c);const out=portable(c,glass);c.removeFromParent();return out;
    };
    if (piecesMesh) g.add(lean(piecesMesh)); g.add(lean(frags));
    return g;
  }
  return { name: 'prism', scene, camera, refraction: { frame: shardMaterial.uniforms.tFrame, texel: shardMaterial.uniforms.uFrameTexel }, resize, update, setPalette, exportGroup, post: { ca: .0028, bloom: 1.05, threshold: .72, exposure: .85, sat: 1.15, vignette: 1.7, flare: .32, flareTint: new THREE.Color('#ff9ae6') } };
}
