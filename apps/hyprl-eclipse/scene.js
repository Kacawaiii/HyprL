import * as THREE from 'three';
import { CSS3DRenderer } from './vendor/CSS3DRenderer.js';
import { GLTFExporter } from './vendor/GLTFExporter.js';
import { VERT_SCREEN, smooth } from './lib/kit.js';
import { createMonolithChapter } from './chapters/monolith.js';
import { createPlanetChapter } from './chapters/planet.js';
import { createSingularityChapter } from './chapters/singularity.js';
import { createPrismChapter } from './chapters/prism.js';
import { createShatter } from './lib/shatter.js';

export const CHAPTERS = ['monolith', 'planet', 'singularity', 'prism'];
// How the frame passes from one chapter to the next (scrolling back plays it in reverse).
const TRANSITION_KIND = { 'monolith>planet': 0, 'planet>singularity': 1, 'singularity>prism': 2 };

/**
 * HYPRL universe: four procedural 3D chapters behind the page, joined on scroll by a zoom into the planet,
 * a gravitational recoil/vortex and a shattering screen, composited with chromatic aberration, bloom, vignette and grain.
 * No textures, no network.
 *
 * container       fixed full-viewport element for the WebGL canvas
 * labelContainer  element covering the hero, for the readable CSS3D card overlays
 * options.hero    the hero element (scroll sync for the glass cards)
 * options.stops   elements with data-chapter (and optional data-dim), in page order
 * options.onFrame called with the frame time before each frame (the smooth scroller), so the page and
 *                 the WebGL scene move in the same frame
 */
export function createEclipseScene(container, labelContainer, { studio = false, hero = null, stops = [], onFrame = null } = {}) {
  const media = matchMedia('(prefers-reduced-motion: reduce)');
  let paused = media.matches, disposed = false, frame = 0, dirty = true, elapsed = 0, previous = performance.now(), palette = 'gold';
  let forced = studio ? 0 : null;
  const pointer = new THREE.Vector2(), smoothPointer = new THREE.Vector2();
  const isMobile = () => container.clientWidth < 600;

  let renderer;
  try { renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance', preserveDrawingBuffer: studio }); }
  catch { document.body.classList.add('scene-unavailable'); return null; }
  if (!renderer.capabilities.isWebGL2) { renderer.dispose(); document.body.classList.add('scene-unavailable'); return null; }
  renderer.setClearColor(0x000000, 1); renderer.autoClear = true; renderer.info.autoReset = false;
  container.append(renderer.domElement);
  const cssRenderer = new CSS3DRenderer(); labelContainer?.append(cssRenderer.domElement);

  const ctx = { isMobile };
  const chapters = [createMonolithChapter(ctx), createPlanetChapter(ctx), createSingularityChapter(ctx), createPrismChapter(ctx)];
  const byName = Object.fromEntries(chapters.map((c, i) => [c.name, i]));
  const shatter = createShatter({ isMobile });

  // ── Chapter transition, shared by the bloom and final passes ──
  // The camera flies into chapter A while B settles from a slight zoom, revealed from the centre by an
  // expanding, ragged ring of light (an eclipse limb opening onto the next scene).
  const TRANSITION = /* glsl */`uniform float uMix,uTime,uAspect,uHoleRadius;uniform vec3 uGravityTint;uniform vec2 uHoleCenter;uniform int uKind;
    float tHash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
    float tNoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
      return mix(mix(tHash(i),tHash(i+vec2(1,0)),f.x),mix(tHash(i+vec2(0,1)),tHash(i+vec2(1,1)),f.x),f.y);}
    // Recoil first, then differential 1/r rotation and radial contraction into the hole.
    vec2 vortexCenter(){return mix(vec2(.5,.5),uHoleCenter,smoothstep(.45,.92,uMix));}
    float recoilScale(){return exp(-smoothstep(0.,.22,uMix)*1.12-smoothstep(.28,.8,uMix)*2.8);}
    vec2 zoomA(vec2 uv){if(uKind<0)return uv;
      if(uKind==1){vec2 asp=vec2(uAspect,1.);vec2 p=(uv-vortexCenter())*asp;
        float shake=smoothstep(0.,.015,uMix)*(1.-smoothstep(.07,.24,uMix))*.012;
        p+=vec2(sin(uMix*93.),sin(uMix*137.))*shake;
        float r=length(p),spin=smoothstep(.22,.72,uMix)*.55/(r+.028);
        float a=atan(p.y,p.x)+spin;return vec2(cos(a),sin(a))*r/recoilScale()/asp+.5;}
      return (uv-.5)/(1.+.35*uMix*uMix)+.5;}
    vec2 zoomB(vec2 uv){if(uKind<0||uKind==1)return uv;float k=1.-uMix;
      if(uKind==2)return (uv-.5)/(1.+.12*k*k)+.5;return (uv-.5)/(1.+.24*k*k)+.5;}
    vec3 frameA(sampler2D tex,vec2 uv){vec2 q=zoomA(uv);
      float edge=min(min(q.x,1.-q.x),min(q.y,1.-q.y));
      return texture2D(tex,clamp(q,0.,1.)).rgb*smoothstep(0.,.07,edge);}
    vec3 gravity(sampler2D texA,sampler2D texB,vec2 uv){
      vec2 asp=vec2(uAspect,1.),p=(uv-vortexCenter())*asp;float r=length(p);
      float pull=smoothstep(.22,.72,uMix),speed=sin(3.14159*smoothstep(0.,.24,uMix));
      vec3 a=vec3(0.);float smear=.2*speed+.035*pull;
      // Outside the contracted frame no source sample contributes; avoid five inverse warps there.
      if(r<recoilScale()*length(asp)*.7+.02)for(int i=0;i<5;i++){float k=float(i)/4.-.5;float angle=k*.12*pull/(r+.03);mat2 R=mat2(cos(angle),-sin(angle),sin(angle),cos(angle));
        a+=frameA(texA,vortexCenter()+R*p*(1.+k*smear)/asp);}
      a*=.2*(1.-smoothstep(.64,.85,uMix));
      float blackR=mix(.012,uHoleRadius,smoothstep(.53,.95,uMix));
      a*=smoothstep(blackR*.72,blackR*.96,r);
      float ring=smoothstep(.3,.73,uMix),settle=smoothstep(.64,1.,uMix);
      vec3 b=texture2D(texB,uv).rgb;
      float radius=mix(.06,uHoleRadius,smoothstep(.34,.88,uMix));
      // A procedural photon rim grows from the wound-up light; the final scene enters at its real UVs.
      // Sampling a scaled target rectangle would expose its cropped top edge during ring growth.
      float winding=r-radius+.002*sin(atan(p.y,p.x)*6.+pull*18.);
      float photon=exp(-pow(winding/.0035,2.));
      float fil=.3+.7*pow(.5+.5*sin((r-radius)*760.+atan(p.y,p.x)*4.+pull*9.),3.);
      float approaching=.45+.65*smoothstep(-.5,.8,-p.x/(r+.001));
      vec3 forming=(mix(uGravityTint,vec3(1.),.55)*photon*2.8+uGravityTint*exp(-abs(winding)*50.)*fil*.5)*approaching+b*.5*smoothstep(.64,.86,uMix);
      float annulus=exp(-pow((r-radius)/(.02+.14*ring),2.));
      vec3 col=a+forming*annulus*ring*(1.-settle);
      col=mix(col,b,settle);
      // Receding light stretches into luminous, differential spiral arcs before joining the disc.
      float ang=atan(p.y,p.x),phase=ang+pull*.55/(r+.028);
      float arcs=pow(.5+.5*sin(phase*3.+r*40.),16.)*exp(-pow((r-radius*.85)/(.025+.12*pull),2.));
      col+=uGravityTint*arcs*sin(3.14159*pull)*.24;
      return col;}
    float ringEdge(){return mix(-.25,1.45,uMix);}
    float ringDist(vec2 uv){vec2 p=(uv-.5)*vec2(uAspect,1.);float r=length(p)/(.5*length(vec2(uAspect,1.)));
      vec2 d=p/(length(p)+1e-4);return r+(tNoise(d*1.6+uTime*.1)-.5)*.07+(tNoise(uv*9.+uTime*.05)-.5)*.02;}
    float reveal(float r){float e=ringEdge();return uMix>.999?1.:1.-smoothstep(e-.09,e+.008,r);}
    float revealOf(vec2 uv){if(uKind<0)return uMix;if(uKind==2)return 1.;return reveal(ringDist(uv));}`;

  // ── Post-processing chain (all hand-written, three core only) ──
  const rtOpts = { type: THREE.HalfFloatType, depthBuffer: true, samples: isMobile() ? 2 : 4 };
  const rtA = new THREE.WebGLRenderTarget(1, 1, rtOpts), rtB = new THREE.WebGLRenderTarget(1, 1, rtOpts);
  const small = { type: THREE.HalfFloatType, depthBuffer: false };
  const rtGlass = new THREE.WebGLRenderTarget(1, 1, small);
  const rtS1 = new THREE.WebGLRenderTarget(1, 1, small), rtS2 = new THREE.WebGLRenderTarget(1, 1, small);
  const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1), quadScene = new THREE.Scene();
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2)); quad.frustumCulled = false; quadScene.add(quad);
  // Optional glass capture: half-resolution HDR frame, allocated only when a chapter uses it.
  const copyMat = new THREE.ShaderMaterial({ uniforms: { tIn: { value: null } }, vertexShader: VERT_SCREEN,
    fragmentShader: 'varying vec2 vUv;uniform sampler2D tIn;void main(){gl_FragColor=texture2D(tIn,vUv);}', depthTest: false, depthWrite: false });
  const brightMat = new THREE.ShaderMaterial({ uniforms: { tA: { value: null }, tB: { value: null }, uMix: { value: 0 }, uKind: { value: 0 }, uTime: { value: 0 }, uAspect: { value: 1 }, uHoleCenter: { value: new THREE.Vector2(.64,.47) }, uHoleRadius: { value: .48 }, uGravityTint: { value: new THREE.Color() }, uTexel: { value: new THREE.Vector2() } }, vertexShader: VERT_SCREEN, depthTest: false, depthWrite: false,
    fragmentShader: /* glsl */`varying vec2 vUv;uniform sampler2D tA,tB;uniform vec2 uTexel;${TRANSITION}
      vec3 s(vec2 uv){if(uKind==1&&uMix>.001)return gravity(tA,tB,uv);if(uMix<.001)return texture2D(tA,uv).rgb;return mix(texture2D(tA,zoomA(uv)).rgb,texture2D(tB,zoomB(uv)).rgb,revealOf(uv));}
      void main(){vec3 c=(s(vUv+uTexel*vec2(-1,-1))+s(vUv+uTexel*vec2(1,-1))+s(vUv+uTexel*vec2(-1,1))+s(vUv+uTexel*vec2(1,1)))*.25;
        float l=dot(c,vec3(.2126,.7152,.0722));gl_FragColor=vec4(c*smoothstep(.35,1.2,l),1.);}` });
  const blurMat = new THREE.ShaderMaterial({ uniforms: { tIn: { value: null }, uDir: { value: new THREE.Vector2() } }, vertexShader: VERT_SCREEN, depthTest: false, depthWrite: false,
    fragmentShader: /* glsl */`varying vec2 vUv;uniform sampler2D tIn;uniform vec2 uDir;
      void main(){vec3 c=texture2D(tIn,vUv).rgb*.227;c+=(texture2D(tIn,vUv+uDir*1.385).rgb+texture2D(tIn,vUv-uDir*1.385).rgb)*.316;c+=(texture2D(tIn,vUv+uDir*3.23).rgb+texture2D(tIn,vUv-uDir*3.23).rgb)*.07;gl_FragColor=vec4(c,1.);}` });
  const finalMat = new THREE.ShaderMaterial({
    uniforms: { tA: { value: null }, tB: { value: null }, tBloom: { value: null }, uMix: { value: 0 }, uKind: { value: 0 }, uCAa: { value: 0 }, uCAb: { value: 0 }, uBloom: { value: 1 }, uExpo: { value: 1 }, uDim: { value: 0 }, uVig: { value: .85 }, uSat: { value: 1 }, uTime: { value: 0 }, uAspect: { value: 1 }, uHoleCenter: { value: new THREE.Vector2(.64,.47) }, uHoleRadius: { value: .48 }, uGravityTint: { value: new THREE.Color() }, uAccent: { value: new THREE.Color() }, uFlare: { value: 0 }, uFlareTint: { value: new THREE.Color(1, 1, 1) }, uRes: { value: new THREE.Vector2() } },
    vertexShader: VERT_SCREEN, depthTest: false, depthWrite: false,
    fragmentShader: /* glsl */`varying vec2 vUv;uniform sampler2D tA,tB,tBloom;uniform float uCAa,uCAb,uBloom,uExpo,uDim,uSat,uVig,uFlare;uniform vec2 uRes;uniform vec3 uAccent,uFlareTint;${TRANSITION}
      vec3 ca(sampler2D t,vec2 uv,float s){vec2 d=(uv-.5)*s;return vec3(texture2D(t,uv+d).r,texture2D(t,uv).g,texture2D(t,uv-d).b);}
      vec3 rblur(sampler2D t,vec2 uv,float s){vec3 c=vec3(0.);vec2 d=(uv-.5)*s;for(int i=0;i<7;i++)c+=texture2D(t,uv-d*(float(i)/6.)).rgb;return c/7.;}
      vec3 aces(vec3 x){return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.);}
      float h(vec2 p){return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453);}
      void main(){vec3 col;float seam=0.;
        if(uMix<.001)col=ca(tA,vUv,uCAa);
        else if(uKind==2)col=ca(tB,zoomB(vUv),uCAb);   // shatter: B underneath, A's shards were drawn over it
        else{vec2 ua=zoomA(vUv),ub=zoomB(vUv);float pulse=sin(3.14159*uMix);
          if(uKind<0)col=mix(ca(tA,vUv,uCAa),ca(tB,vUv,uCAb),uMix);
          else if(uKind==1)col=gravity(tA,tB,vUv);
          else{vec3 a=mix(ca(tA,ua,uCAa),rblur(tA,ua,.09*uMix),smoothstep(0.,.35,uMix))*(1.-.4*uMix);
            vec3 b=mix(ca(tB,ub,uCAb),rblur(tB,ub,.07*(1.-uMix)),smoothstep(0.,.35,1.-uMix))*(.65+.35*uMix);
            float r=ringDist(vUv),e=ringEdge();col=mix(a,b,reveal(r));
            seam=(exp(-pow((r-e)/.009,2.))*.75+exp(-pow((r-e)/.04,2.))*.16)*pulse;}}
        col+=texture2D(tBloom,vUv).rgb*uBloom;
        // Anamorphic flare: the brightest points smeared into a thin horizontal streak (chapters that ask for it).
        if(uFlare>.001){vec3 fl=vec3(0.);for(int i=1;i<=10;i++){float o=float(i*i)*.0032;fl+=(texture2D(tBloom,vUv+vec2(o,0.)).rgb+texture2D(tBloom,vUv-vec2(o,0.)).rgb)*exp(-float(i)*.3);}
          float fl2=dot(fl,vec3(.3,.5,.2))*.12;col+=uFlareTint*fl2*fl2*uFlare;}
        col*=uExpo;col=mix(vec3(dot(col,vec3(.2126,.7152,.0722))),col,uSat);
        col+=mix(vec3(1.),uAccent,.45)*seam;
        vec2 p=vUv-.5;col*=1.-dot(p,p)*uVig;col*=1.-uDim;
        col=aces(col);col=pow(col,vec3(1./2.2));
        col+=(h(vUv*uRes+fract(uTime*7.))-.5)*.022;
        gl_FragColor=vec4(col,1.);}` });

  const white = new THREE.Color(1, 1, 1);
  let W = 1, H = 1, PR = 1, quality = 1, slowFrames = 0, sampled = 0;
  function resize() {
    W = container.clientWidth || innerWidth; H = container.clientHeight || innerHeight; if (!W || !H) return;
    PR = Math.min(devicePixelRatio, isMobile() ? 1.25 : 1.5) * quality;
    renderer.setPixelRatio(PR); renderer.setSize(W, H, false);
    renderer.domElement.style.width = '100%'; renderer.domElement.style.height = '100%';
    const pw = Math.round(W * PR), ph = Math.round(H * PR);
    rtA.setSize(pw, ph); rtB.setSize(pw, ph); rtS1.setSize(pw >> 2, ph >> 2); rtS2.setSize(pw >> 2, ph >> 2);
    finalMat.uniforms.uRes.value.set(pw, ph); brightMat.uniforms.uTexel.value.set(1 / pw, 1 / ph);
    finalMat.uniforms.uAspect.value = brightMat.uniforms.uAspect.value = W / H;
    const heroH = hero ? hero.offsetHeight : H;
    chapters[0].resize(W, H, heroH); for (const c of chapters.slice(1)) c.resize(W, H);
    shatter.resize(W, H);
    if (labelContainer) cssRenderer.setSize(W, heroH);
    dirty = true;
  }

  // ── Scroll → chapter blend ──────────────────────────────────────
  // Where the page is: a fractional stop index f and each chapter's own progress (0..1).
  function scrollTarget() {
    const vh = innerHeight; let f = 0; const rects = stops.map(el => el.getBoundingClientRect());
    for (let i = 1; i < stops.length; i++) f += smooth(vh * .9, vh * .2, rects[i].top);
    const local = {};
    for (const name of CHAPTERS) {
      const idx = stops.map((el, i) => el.dataset.chapter === name ? i : -1).filter(i => i >= 0); if (!idx.length) continue;
      const top = rects[idx[0]].top, bottom = rects[idx[idx.length - 1]].bottom; local[name] = Math.min(Math.max((vh - top) / (bottom - top + vh), 0), 1);
    }
    return { f, local };
  }
  function blendAt(f, local) {
    const i0 = Math.min(Math.floor(f), stops.length - 1), i1 = Math.min(i0 + 1, stops.length - 1), t = f - i0;
    const dimOf = el => parseFloat(el.dataset.dim || 0);
    return { a: byName[stops[i0].dataset.chapter] ?? 0, b: byName[stops[i1].dataset.chapter] ?? 0, t, dim: dimOf(stops[i0]) * (1 - t) + dimOf(stops[i1]) * t, local };
  }
  // The scene glides towards the page position instead of jumping with each wheel notch, anchor or key.
  // Paused or reduced motion (dt = 0): it snaps, so a still frame always matches the page.
  const glide = { f: null, local: {} };
  function readScroll(dt = 0) {
    if (forced !== null || !stops.length) { glide.f = null; return { a: forced ?? 0, b: forced ?? 0, t: 0, dim: 0, local: {} }; }
    const target = scrollTarget(), k = glide.f === null || dt <= 0 ? 1 : 1 - Math.exp(-dt * 4.5);
    glide.f = glide.f === null ? target.f : glide.f + (target.f - glide.f) * k;
    if (Math.abs(target.f - glide.f) < 1e-4) glide.f = target.f;
    for (const [name, value] of Object.entries(target.local)) {
      const g = glide.local[name]; glide.local[name] = g === undefined ? value : g + (value - g) * k;
    }
    return blendAt(glide.f, { ...glide.local });
  }

  function renderChapter(c, target, s, pr) {
    const scroll = hero ? Math.max(0, -hero.getBoundingClientRect().top) : 0;
    c.update({ time: elapsed, dt: s.dt, pointer: smoothPointer, motion: paused ? 0 : 1, local: s.local[c.name], scroll, viewH: H, pr });
    if (!c.refraction) { renderer.setRenderTarget(target); renderer.render(c.scene, c.camera); return; }
    // Base layer → frame copy → glass → light overlays. The glass never samples its own render target.
    const layers = c.camera.layers.mask;
    c.camera.layers.set(0); renderer.setRenderTarget(target); renderer.render(c.scene, c.camera);
    const gw = Math.max(1, target.width >> 1), gh = Math.max(1, target.height >> 1);
    if (rtGlass.width !== gw || rtGlass.height !== gh) rtGlass.setSize(gw, gh);
    quad.material = copyMat; copyMat.uniforms.tIn.value = target.texture;
    renderer.setRenderTarget(rtGlass); renderer.render(quadScene, quadCam);
    c.refraction.frame.value = rtGlass.texture; c.refraction.texel.value.set(1 / gw, 1 / gh);
    renderer.setRenderTarget(target); renderer.autoClear = false;
    c.camera.layers.set(1); renderer.render(c.scene, c.camera);
    c.camera.layers.set(2); renderer.render(c.scene, c.camera);
    renderer.autoClear = true; c.camera.layers.mask = layers;
  }
  let lastBlend = null;
  function render(dt = 0) {
    renderer.info.reset();
    const s = readScroll(dt); s.dt = dt; lastBlend = s;
    const A = chapters[s.a], B = chapters[s.b], mixT = s.a === s.b ? 0 : s.t;
    const kind = media.matches && mixT > .001 ? -1 : mixT > .001 ? TRANSITION_KIND[`${A.name}>${B.name}`] ?? TRANSITION_KIND[`${B.name}>${A.name}`] ?? 0 : 0;
    renderChapter(A, rtA, s, PR); if (mixT > .001) renderChapter(B, rtB, s, PR);
    if (kind === 2) { renderer.setRenderTarget(rtB); renderer.autoClear = false; renderer.clearDepth(); shatter.render(renderer, rtA.texture, mixT, elapsed); renderer.autoClear = true; }
    brightMat.uniforms.uKind.value = finalMat.uniforms.uKind.value = kind;
    if(kind===1){for(const m of [brightMat,finalMat]){m.uniforms.uHoleCenter.value.copy(B.transition.center);m.uniforms.uHoleRadius.value=B.transition.radius.value;}}
    const pa = A.post, pb = mixT > .001 ? B.post : A.post, lerp = (x, y) => x + (y - x) * mixT;
    quad.material = brightMat; Object.assign(brightMat.uniforms.tA, { value: rtA.texture }); brightMat.uniforms.tB.value = rtB.texture; brightMat.uniforms.uMix.value = kind === 2 ? 1 : mixT; brightMat.uniforms.uTime.value = elapsed;
    renderer.setRenderTarget(rtS1); renderer.render(quadScene, quadCam);
    quad.material = blurMat;
    for (const [src, dst, dir] of [[rtS1, rtS2, [1, 0]], [rtS2, rtS1, [0, 1]], [rtS1, rtS2, [2, 0]], [rtS2, rtS1, [0, 2]]]) {
      blurMat.uniforms.tIn.value = src.texture; blurMat.uniforms.uDir.value.set(dir[0] / rtS1.width, dir[1] / rtS1.height); renderer.setRenderTarget(dst); renderer.render(quadScene, quadCam);
    }
    quad.material = finalMat; const u = finalMat.uniforms;
    u.tA.value = rtA.texture; u.tB.value = rtB.texture; u.tBloom.value = rtS1.texture; u.uMix.value = kind === 2 ? 1 : mixT;
    u.uCAa.value = pa.ca; u.uCAb.value = pb.ca; u.uBloom.value = lerp(pa.bloom, pb.bloom); u.uExpo.value = lerp(pa.exposure, pb.exposure); u.uDim.value = s.dim; u.uSat.value = lerp(pa.sat ?? 1, pb.sat ?? 1); u.uVig.value = lerp(pa.vignette ?? .85, pb.vignette ?? .85); u.uFlare.value = lerp(pa.flare ?? 0, pb.flare ?? 0); u.uFlareTint.value.copy(mixT > .5 ? pb.flareTint ?? white : pa.flareTint ?? white); u.uTime.value = elapsed;
    renderer.setRenderTarget(null); renderer.render(quadScene, quadCam);
    // Readable overlays only while the first chapter (the hero) is on screen.
    const heroOn = (s.a === 0 && (1 - mixT) > .02) || (s.b === 0 && mixT > .02);
    const heroWeight = s.a === 0 ? 1 - mixT : (s.b === 0 ? mixT : 0);
    if (labelContainer) {
      const style = cssRenderer.domElement.style;
      style.visibility = heroOn ? 'visible' : 'hidden'; style.opacity = smooth(.55, 1, heroWeight).toFixed(3);  // gone before the limb reaches them
      // The card texts follow the transition zoom of the WebGL frame (about the viewport centre).
      const zoom = mixT < .001 ? 1 : s.a === 0 ? 1 + .35 * mixT * mixT : 1 + .24 * (1 - mixT) ** 2;
      if (zoom === 1) style.transform = '';
      else { const heroScroll = hero ? Math.max(0, -hero.getBoundingClientRect().top) : 0; style.transformOrigin = `${W / 2}px ${heroScroll + H / 2}px`; style.transform = `scale(${zoom.toFixed(4)})`; }
      if (heroOn) cssRenderer.render(chapters[0].labels, chapters[0].labelCamera);
    }
    dirty = false;
  }

  const resizeObserver = new ResizeObserver(resize); resizeObserver.observe(container); if (hero) resizeObserver.observe(hero);
  function onScroll() { dirty = true; }
  addEventListener('scroll', onScroll, { passive: true });
  function pointerMove(e) { pointer.set(e.clientX / innerWidth * 2 - 1, e.clientY / innerHeight * 2 - 1); }
  function pointerLeave() { pointer.set(0, 0); }
  const interaction = studio ? container.parentElement : window;
  interaction.addEventListener('pointermove', pointerMove, { passive: true }); document.documentElement.addEventListener('pointerleave', pointerLeave);

  function animate(now) {
    if (disposed) return; frame = requestAnimationFrame(animate);
    if (document.hidden) { previous = now; return; }
    onFrame?.(now);
    const dt = Math.min((now - previous) / 1000, .05); previous = now;
    if (!paused) {
      elapsed += dt; smoothPointer.lerp(pointer, .045); render(dt);
      // Adaptive resolution: drop render scale on slow GPUs (never below 60 %).
      if (!studio && quality > .6 && ++sampled > 30) { if (dt > .028) slowFrames++; else slowFrames = Math.max(0, slowFrames - 1); if (slowFrames > 45) { quality = Math.max(.6, quality - .15); slowFrames = 0; resize(); } }
    }
    else if (dirty) render(0);
  }
  function setPaused(value) { paused = Boolean(value); if (paused) { pointer.set(0, 0); smoothPointer.set(0, 0); } previous = performance.now(); dirty = true; }
  const gold = new THREE.Color('#dac09a'), ice = new THREE.Color('#a0c9e8');
  function setPalette(value) { palette = value === 'ice' ? 'ice' : 'gold'; const color = palette === 'ice' ? ice : gold; for (const c of chapters) c.setPalette(color, palette); finalMat.uniforms.uAccent.value.copy(color);brightMat.uniforms.uGravityTint.value.copy(color);finalMat.uniforms.uGravityTint.value.copy(color); document.documentElement.style.setProperty('--accent', `#${color.getHexString()}`); dirty = true; }
  function setChapter(name) { forced = name == null ? null : (typeof name === 'number' ? name : byName[name] ?? 0); dirty = true; }
  function mediaChange() { setPaused(media.matches); container.dispatchEvent(new CustomEvent('motionchange', { detail: { paused } })); }
  media.addEventListener('change', mediaChange);
  function contextLost(event) { event.preventDefault(); document.body.classList.remove('scene-ready'); document.body.classList.add('scene-unavailable'); }
  renderer.domElement.addEventListener('webglcontextlost', contextLost);
  renderer.domElement.addEventListener('webglcontextrestored', () => { document.body.classList.remove('scene-unavailable'); document.body.classList.add('scene-ready'); resize(); });

  function download(blob, filename) { const url = URL.createObjectURL(blob), a = document.createElement('a'); a.href = url; a.download = filename; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
  async function exportGLB() {
    // Portable geometry + PBR materials for every chapter. Shaders, glow and animation stay in code.
    const asset = new THREE.Scene(); asset.name = 'HYPRL_Universe_Portable';
    chapters.forEach((c, i) => { const g = c.exportGroup(); g.position.x = i * 40; asset.add(g); });
    asset.userData = { description: 'HYPRL — 4 chapitres (Monolithe, Planète, Singularité, Prisme). Géométrie éditable ; shaders, lumière et animation dans chapters/*.js.', palette };
    const result = await new GLTFExporter().parseAsync(asset, { binary: true });
    download(new Blob([result], { type: 'model/gltf-binary' }), 'hyprl-universe.glb');
    asset.traverse(o => { if (o.material) for (const m of [].concat(o.material)) m.dispose(); });
  }
  function exportPNG() { render(0); renderer.domElement.toBlob(blob => { if (blob) download(blob, `hyprl-${CHAPTERS[forced ?? 0]}.png`); }); }

  setPalette('gold'); resize(); render(0); document.body.classList.add('scene-ready'); frame = requestAnimationFrame(animate);
  return {
    setPaused, setPalette, setChapter, exportGLB, exportPNG,
    get paused() { return paused; }, get chapters() { return chapters; }, get chapter() { if (forced !== null || !stops.length) return CHAPTERS[forced ?? 0]; const { f, local } = scrollTarget(), b = blendAt(f, local); return CHAPTERS[b.t > .5 ? b.b : b.a]; },
    get stats() { return { drawCalls: renderer.info.render.calls, triangles: renderer.info.render.triangles }; },
    dispose() {
      disposed = true; cancelAnimationFrame(frame); resizeObserver.disconnect(); removeEventListener('scroll', onScroll); interaction.removeEventListener('pointermove', pointerMove);
      document.documentElement.removeEventListener('pointerleave', pointerLeave); media.removeEventListener('change', mediaChange);
      for (const c of chapters) c.scene.traverse(o => { o.geometry?.dispose(); if (o.material) for (const m of [].concat(o.material)) m.dispose(); });
      shatter.dispose(); copyMat.dispose(); for (const t of [rtA, rtB, rtS1, rtS2, rtGlass]) t.dispose(); renderer.dispose(); renderer.domElement.remove(); cssRenderer.domElement.remove();
    }
  };
}
