import * as THREE from 'three';
import { CSS3DRenderer } from './vendor/CSS3DRenderer.js';
import { GLTFExporter } from './vendor/GLTFExporter.js';
import { VERT_SCREEN, smooth } from './lib/kit.js';
import { createMonolithChapter } from './chapters/monolith.js';
import { createPlanetChapter } from './chapters/planet.js';
import { createSingularityChapter } from './chapters/singularity.js';
import { createPrismChapter } from './chapters/prism.js';
import { createShatter } from './lib/shatter.js';
import { createSurfacePullback } from './lib/pullback.js';
import { motionPref } from './lib/motion.js';

export const CHAPTERS = ['monolith', 'planet', 'singularity', 'prism'];
// How the frame passes from one chapter to the next (scrolling back plays it in reverse).
const TRANSITION_KIND = { 'monolith>planet': 0, 'planet>singularity': 1, 'singularity>prism': 2 };

/**
 * HYPRL universe: four procedural 3D chapters behind the page, joined on scroll by an atmospheric pullback,
 * a receding planet / hyperspace arrival and a shattering screen, composited with chromatic aberration, bloom, vignette and grain.
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
  let userPaused = false, paused = motionPref.reduced, disposed = false, frame = 0, dirty = true, elapsed = 0, previous = performance.now(), palette = 'gold';
  let forced = studio ? 0 : null;
  const pointer = new THREE.Vector2(), smoothPointer = new THREE.Vector2();
  const isMobile = () => container.clientWidth < 600;

  let renderer;
  try {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('webgl2', { alpha: false, antialias: false, powerPreference: 'high-performance', preserveDrawingBuffer: studio });
    if (!context) { document.body.classList.add('scene-unavailable'); return null; }
    renderer = new THREE.WebGLRenderer({ canvas, context });
  }
  catch { document.body.classList.add('scene-unavailable'); return null; }
  if (!renderer.capabilities.isWebGL2) { renderer.dispose(); document.body.classList.add('scene-unavailable'); return null; }
  renderer.setClearColor(0x000000, 1); renderer.autoClear = true; renderer.info.autoReset = false;
  container.append(renderer.domElement);
  const cssRenderer = new CSS3DRenderer(); labelContainer?.append(cssRenderer.domElement);

  const ctx = { isMobile };
  const chapters = [createMonolithChapter(ctx), createPlanetChapter(ctx), createSingularityChapter(ctx), createPrismChapter(ctx)];
  const pullback = createSurfacePullback(chapters[0], chapters[1]);
  const byName = Object.fromEntries(chapters.map((c, i) => [c.name, i]));
  const shatter = createShatter({ isMobile });

  // The surface pullback is one world. Warp blends at the flash; shatter retains its incoming zoom.
  const TRANSITION = /* glsl */`uniform float uMix,uTime,uAspect;uniform vec3 uTravelTint;uniform vec2 uTravelOrigin;uniform int uKind;
    float tHash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
    vec2 zoomB(vec2 uv){float k=1.-uMix;return uKind==2?(uv-.5)/(1.+.12*k*k)+.5:uv;}
    float blendWeight(){if(uKind<0)return uMix;if(uKind==2)return 1.;
      return uKind==0?0.:smoothstep(.48,.60,uMix);}
    float streak(vec2 p,float stretch){float r=length(p),a=atan(p.y,p.x)*56.;float lane=floor(a);
      float seed=tHash(vec2(lane,13.));float jitter=.15+.7*tHash(vec2(lane,7.));
      float d=abs(fract(a)-jitter)*r/56.;
      float phase=fract(log(r+.035)*1.7-smoothstep(0.,1.,uMix)*6.+seed*9.);
      float tail=smoothstep(1.-stretch,1.-stretch*.18,phase)*(1.-smoothstep(.97,1.,phase));
      return exp(-pow(d/.00085,2.))*tail*step(.79,seed)*smoothstep(.04,.15,r);}
    vec3 travelLight(vec2 uv){vec2 p=(uv-.5)*vec2(uAspect,1.);
      if(uKind==1){float speed=smoothstep(.10,.43,uMix)*(1.-smoothstep(.58,.96,uMix));
        float stretch=.08+.78*speed;
        vec2 stream=(uv-uTravelOrigin)*vec2(uAspect,1.);
        vec3 lines=vec3(streak(stream*1.006,stretch),streak(stream,stretch),streak(stream*.994,stretch));
        float flash=exp(-pow((uMix-.54)/.052,2.));
        float point=exp(-dot(p,p)*4000.)*smoothstep(.32,.48,uMix)*(1.-smoothstep(.54,.62,uMix));
        return lines*mix(vec3(.65,.78,1.),uTravelTint,.25)*speed*.85+vec3(1.,.85,.66)*(flash*.36+point*3.);}
      return vec3(0.);}
    `;

  // ── Post-processing chain (all hand-written, three core only) ──
  const rtOpts = { type: THREE.HalfFloatType, depthBuffer: true, samples: isMobile() ? 0 : 2 };
  const rtA = new THREE.WebGLRenderTarget(1, 1, rtOpts), rtB = new THREE.WebGLRenderTarget(1, 1, rtOpts);
  const small = { type: THREE.HalfFloatType, depthBuffer: false };
  const rtGlass = new THREE.WebGLRenderTarget(1, 1, small);
  const rtS1 = new THREE.WebGLRenderTarget(1, 1, small), rtS2 = new THREE.WebGLRenderTarget(1, 1, small);
  const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1), quadScene = new THREE.Scene();
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2)); quad.frustumCulled = false; quadScene.add(quad);
  // Optional glass capture: half-resolution HDR frame, allocated only when a chapter uses it.
  const copyMat = new THREE.ShaderMaterial({ uniforms: { tIn: { value: null } }, vertexShader: VERT_SCREEN,
    fragmentShader: 'varying vec2 vUv;uniform sampler2D tIn;void main(){gl_FragColor=texture2D(tIn,vUv);}', depthTest: false, depthWrite: false });
  const brightMat = new THREE.ShaderMaterial({ uniforms: { tA: { value: null }, tB: { value: null }, uMix: { value: 0 }, uKind: { value: 0 }, uTime: { value: 0 }, uAspect: { value: 1 }, uTravelTint: { value: new THREE.Color() }, uTravelOrigin: { value: new THREE.Vector2(.5,.5) }, uThreshold: { value: .35 }, uTexel: { value: new THREE.Vector2() } }, vertexShader: VERT_SCREEN, depthTest: false, depthWrite: false,
    fragmentShader: /* glsl */`varying vec2 vUv;uniform sampler2D tA,tB;uniform vec2 uTexel;uniform float uThreshold;${TRANSITION}
      vec3 s(vec2 uv){if(uMix<.001)return texture2D(tA,uv).rgb;return mix(texture2D(tA,uv).rgb,texture2D(tB,zoomB(uv)).rgb,blendWeight())+(uKind==0?vec3(0.):travelLight(uv));}
      void main(){vec3 c=(s(vUv+uTexel*vec2(-1,-1))+s(vUv+uTexel*vec2(1,-1))+s(vUv+uTexel*vec2(-1,1))+s(vUv+uTexel*vec2(1,1)))*.25;
        float l=dot(c,vec3(.2126,.7152,.0722));gl_FragColor=vec4(c*smoothstep(uThreshold,uThreshold+.85,l),1.);}` });
  const blurMat = new THREE.ShaderMaterial({ uniforms: { tIn: { value: null }, uDir: { value: new THREE.Vector2() } }, vertexShader: VERT_SCREEN, depthTest: false, depthWrite: false,
    fragmentShader: /* glsl */`varying vec2 vUv;uniform sampler2D tIn;uniform vec2 uDir;
      void main(){vec3 c=texture2D(tIn,vUv).rgb*.227;c+=(texture2D(tIn,vUv+uDir*1.385).rgb+texture2D(tIn,vUv-uDir*1.385).rgb)*.316;c+=(texture2D(tIn,vUv+uDir*3.23).rgb+texture2D(tIn,vUv-uDir*3.23).rgb)*.07;gl_FragColor=vec4(c,1.);}` });
  const finalMat = new THREE.ShaderMaterial({
    uniforms: { tA: { value: null }, tB: { value: null }, tBloom: { value: null }, uMix: { value: 0 }, uKind: { value: 0 }, uCAa: { value: 0 }, uCAb: { value: 0 }, uBloom: { value: 1 }, uExpo: { value: 1 }, uDim: { value: 0 }, uGrain: { value: .012 }, uVig: { value: .85 }, uSat: { value: 1 }, uTime: { value: 0 }, uAspect: { value: 1 }, uTravelTint: { value: new THREE.Color() }, uTravelOrigin: { value: new THREE.Vector2(.5,.5) }, uAccent: { value: new THREE.Color() }, uFlare: { value: 0 }, uFlareTint: { value: new THREE.Color(1, 1, 1) }, uRes: { value: new THREE.Vector2() } },
    vertexShader: VERT_SCREEN, depthTest: false, depthWrite: false,
    fragmentShader: /* glsl */`varying vec2 vUv;uniform sampler2D tA,tB,tBloom;uniform float uCAa,uCAb,uBloom,uExpo,uDim,uSat,uVig,uFlare,uGrain;uniform vec2 uRes;uniform vec3 uAccent,uFlareTint;${TRANSITION}
      vec3 ca(sampler2D t,vec2 uv,float s){vec2 d=(uv-.5)*s;return vec3(texture2D(t,uv+d).r,texture2D(t,uv).g,texture2D(t,uv-d).b);}
      vec3 aces(vec3 x){return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.);}
      float h(vec2 p){return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453);}
      void main(){vec3 col;
        if(uMix<.001)col=ca(tA,vUv,uCAa);
        else if(uKind==2)col=ca(tB,zoomB(vUv),uCAb);   // shatter: B underneath, A's shards were drawn over it
        else col=mix(ca(tA,vUv,uCAa),ca(tB,vUv,uCAb),blendWeight())+travelLight(vUv);
        col+=texture2D(tBloom,vUv).rgb*uBloom;
        // Anamorphic flare: the brightest points smeared into a thin horizontal streak (chapters that ask for it).
        if(uFlare>.001){vec3 fl=vec3(0.);for(int i=1;i<=10;i++){float o=float(i*i)*.0032;fl+=(texture2D(tBloom,vUv+vec2(o,0.)).rgb+texture2D(tBloom,vUv-vec2(o,0.)).rgb)*exp(-float(i)*.3);}
          float fl2=dot(fl,vec3(.3,.5,.2))*.12;col+=uFlareTint*fl2*fl2*uFlare;}
        col*=uExpo;col=mix(vec3(dot(col,vec3(.2126,.7152,.0722))),col,uSat);
        vec2 p=vUv-.5;col*=1.-dot(p,p)*uVig;col*=1.-uDim;
        col=aces(col);col=pow(col,vec3(1./2.2));
        col+=(h(vUv*uRes+fract(uTime*7.))-.5)*uGrain;
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
    return { a: byName[stops[i0].dataset.chapter] ?? 0, b: byName[stops[i1].dataset.chapter] ?? 0, t, stopB: i1, dim: dimOf(stops[i0]) * (1 - t) + dimOf(stops[i1]) * t, local };
  }
  // The page already carries wheel damping. A second glide would make the camera lag the copy
  // and would change the scene after keyboard navigation has finished.
  function readScroll() {
    if (forced !== null || !stops.length) return { a: forced ?? 0, b: forced ?? 0, t: 0, dim: 0, local: {} };
    const { f, local } = scrollTarget(); return blendAt(f, local);
  }

  // Freeze the approved framing at the two edges of the transition. Otherwise the chapter's local
  // forward dolly would fight the backward travel, especially at the ends and on reverse scrolling.
  function travelLocal(c, s, role) {
    const rects = stops.filter(el => el.dataset.chapter === c.name).map(el => el.getBoundingClientRect());
    if (!rects.length) return s.local[c.name];
    const vh = innerHeight, delta = stops[s.stopB].getBoundingClientRect().top - vh * (role === 'out' ? .9 : .2);
    return THREE.MathUtils.clamp((vh - rects[0].top + delta) / (rects.at(-1).bottom - rects[0].top + vh), 0, 1);
  }
  function planetEntryLocal(s) {
    const entry = stops.findIndex(el => el.dataset.chapter === 'planet');
    return entry < 0 || motionPref.reduced ? undefined : travelLocal(chapters[1], { ...s, stopB: entry }, 'in');
  }
  const warmed = new Set(), stillLocal = { monolith: .5, planet: .4, singularity: .4, prism: .5 };
  function renderChapter(c, target, s, pr, transition = null, visible = true) {
    const scroll = hero ? Math.max(0, -hero.getBoundingClientRect().top) : 0;
    c.update({ time: elapsed, dt: s.dt, pointer: smoothPointer, motion: paused ? 0 : 1, local: motionPref.reduced ? stillLocal[c.name] : transition ? travelLocal(c, s, transition.role) : s.local[c.name], transition, entryLocal: c.name === 'planet' ? planetEntryLocal(s) : undefined, scroll, viewH: H, pr });
    if (transition?.kind === 0 && transition.role === 'out') {
      chapters[1].update({ time: elapsed, pointer: smoothPointer, motion: paused ? 0 : 1, local: travelLocal(chapters[1], s, 'in'), entryLocal: planetEntryLocal(s) });
      pullback.update(transition.progress);
    }
    if (!visible && warmed.has(c)) return;
    warmed.add(c);
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
    const s = readScroll(); s.dt = dt; lastBlend = s;
    const A = chapters[s.a], B = chapters[s.b], mixT = s.a === s.b ? 0 : s.t;
    const kind = motionPref.reduced && mixT > .001 ? -1 : mixT > .001 ? TRANSITION_KIND[`${A.name}>${B.name}`] ?? TRANSITION_KIND[`${B.name}>${A.name}`] ?? 0 : 0;
    const travel = !motionPref.reduced && mixT > .001 && kind < 2;
    pullback.reset();
    renderChapter(A, rtA, s, PR, travel ? { role: 'out', progress: mixT, kind } : null, !travel || kind === 0 || mixT < .60);
    if (mixT > .001 && !(travel && kind === 0)) renderChapter(B, rtB, s, PR, travel ? { role: 'in', progress: mixT, kind } : null, !travel || mixT > .48);
    if (kind === 2) { renderer.setRenderTarget(rtB); renderer.autoClear = false; renderer.clearDepth(); shatter.render(renderer, rtA.texture, mixT); renderer.autoClear = true; }
    brightMat.uniforms.uKind.value = finalMat.uniforms.uKind.value = kind;
    if (kind === 1) for (const m of [brightMat, finalMat]) {
      const origin = B.transition?.arrivalCenter;
      m.uniforms.uTravelOrigin.value.set(origin?.x ?? .5, origin?.y ?? .5);
    }

    const pa = A.post, pb = mixT > .001 ? B.post : A.post, lerp = (x, y) => x + (y - x) * mixT;
    brightMat.uniforms.uThreshold.value = lerp(pa.threshold ?? .35, pb.threshold ?? .35);
    quad.material = brightMat; Object.assign(brightMat.uniforms.tA, { value: rtA.texture }); brightMat.uniforms.tB.value = rtB.texture; brightMat.uniforms.uMix.value = kind === 2 ? 1 : mixT; brightMat.uniforms.uTime.value = elapsed;
    renderer.setRenderTarget(rtS1); renderer.render(quadScene, quadCam);
    quad.material = blurMat;
    const blurPasses = [[rtS1, rtS2, [1, 0]], [rtS2, rtS1, [0, 1]], [rtS1, rtS2, [2, 0]], [rtS2, rtS1, [0, 2]]];
    for (const [src, dst, dir] of (isMobile() ? blurPasses.slice(0, 2) : blurPasses)) {
      blurMat.uniforms.tIn.value = src.texture; blurMat.uniforms.uDir.value.set(dir[0] / rtS1.width, dir[1] / rtS1.height); renderer.setRenderTarget(dst); renderer.render(quadScene, quadCam);
    }
    quad.material = finalMat; const u = finalMat.uniforms;
    u.tA.value = rtA.texture; u.tB.value = rtB.texture; u.tBloom.value = rtS1.texture; u.uMix.value = kind === 2 ? 1 : mixT;
    u.uCAa.value = pa.ca; u.uCAb.value = pb.ca; u.uBloom.value = lerp(pa.bloom, pb.bloom); u.uExpo.value = lerp(pa.exposure, pb.exposure); u.uDim.value = s.dim; u.uSat.value = lerp(pa.sat ?? 1, pb.sat ?? 1); u.uVig.value = lerp(pa.vignette ?? .85, pb.vignette ?? .85); u.uGrain.value = isMobile() ? .008 : .012; u.uFlare.value = isMobile() ? 0 : lerp(pa.flare ?? 0, pb.flare ?? 0); u.uFlareTint.value.copy(mixT > .5 ? pb.flareTint ?? white : pa.flareTint ?? white); u.uTime.value = elapsed;
    renderer.setRenderTarget(null); renderer.render(quadScene, quadCam);
    // Readable overlays only while the first chapter (the hero) is on screen.
    const heroOn = (s.a === 0 && (1 - mixT) > .02) || (s.b === 0 && mixT > .02);
    const heroWeight = s.a === 0 ? 1 - mixT : (s.b === 0 ? mixT : 0);
    if (labelContainer) {
      const style = cssRenderer.domElement.style;
      style.visibility = heroOn ? 'visible' : 'hidden'; style.opacity = smooth(travel ? .82 : .55, 1, heroWeight).toFixed(3);
      style.transform = '';
      if (heroOn) cssRenderer.render(chapters[0].labels, chapters[0].labelCamera);
    }
    dirty = false;
  }

  const resizeObserver = new ResizeObserver(resize); resizeObserver.observe(container); if (hero) resizeObserver.observe(hero);
  function onScroll() { dirty = true; }
  addEventListener('scroll', onScroll, { passive: true });
  function pointerMove(e) { if (e.pointerType === 'touch') return; pointer.set(e.clientX / innerWidth * 2 - 1, e.clientY / innerHeight * 2 - 1); }
  function pointerLeave() { pointer.set(0, 0); }
  const interaction = studio ? container.parentElement : window;
  interaction.addEventListener('pointermove', pointerMove, { passive: true }); document.documentElement.addEventListener('pointerleave', pointerLeave);

  function animate(now) {
    if (disposed) return; frame = 0;
    if (document.hidden) { previous = now; return; }
    frame = requestAnimationFrame(animate);
    const scrolling = onFrame?.(now);
    const dt = Math.max(0, Math.min((now - previous) / 1000, .05)); previous = now;
    if (!paused) {
      elapsed += dt; smoothPointer.lerp(pointer, 1 - Math.exp(-dt * 9)); render(dt);
      // Adaptive resolution: drop render scale on slow GPUs (never below 60 %).
      if (!studio && quality > .6 && ++sampled > 30) { if (dt > .028) slowFrames++; else slowFrames = Math.max(0, slowFrames - 1); if (slowFrames > 45) { quality = Math.max(.6, quality - .15); slowFrames = 0; resize(); } }
    }
    else if (dirty || scrolling) render(0);
  }
  function setPaused(value) { userPaused = Boolean(value); paused = userPaused || motionPref.reduced; if (paused) { pointer.set(0, 0); smoothPointer.set(0, 0); } previous = performance.now(); dirty = true; }
  const gold = new THREE.Color('#dac09a'), ice = new THREE.Color('#a0c9e8');
  function setPalette(value) { palette = value === 'ice' ? 'ice' : 'gold'; const color = palette === 'ice' ? ice : gold; for (const c of chapters) c.setPalette(color, palette); finalMat.uniforms.uAccent.value.copy(color);brightMat.uniforms.uTravelTint.value.copy(color);finalMat.uniforms.uTravelTint.value.copy(color); document.documentElement.style.setProperty('--accent', `#${color.getHexString()}`); dirty = true; }
  function setChapter(name) { forced = name == null ? null : (typeof name === 'number' ? name : byName[name] ?? 0); dirty = true; }
  function mediaChange() { setPaused(userPaused); container.dispatchEvent(new CustomEvent('motionchange', { detail: { paused } })); }
  const offMotion = motionPref.onChange(mediaChange);
  function visibilityChange() {
    cancelAnimationFrame(frame); frame = 0; previous = performance.now();
    if (!document.hidden && !disposed) { dirty = true; frame = requestAnimationFrame(animate); }
  }
  document.addEventListener('visibilitychange', visibilityChange);
  function contextLost(event) { event.preventDefault(); document.body.classList.remove('scene-ready'); document.body.classList.add('scene-unavailable'); }
  renderer.domElement.addEventListener('webglcontextlost', contextLost);
  function contextRestored() { document.body.classList.remove('scene-unavailable'); document.body.classList.add('scene-ready'); resize(); }
  renderer.domElement.addEventListener('webglcontextrestored', contextRestored);

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
    get paused() { return paused; }, get userPaused() { return userPaused; }, get chapters() { return chapters; }, get chapter() { if (forced !== null || !stops.length) return CHAPTERS[forced ?? 0]; const { f, local } = scrollTarget(), b = blendAt(f, local); return CHAPTERS[b.t > .5 ? b.b : b.a]; },
    get stats() { return { drawCalls: renderer.info.render.calls, triangles: renderer.info.render.triangles }; },
    get travelState() { return { blend: lastBlend, pullback }; },
    dispose() {
      disposed = true; cancelAnimationFrame(frame); resizeObserver.disconnect(); removeEventListener('scroll', onScroll); interaction.removeEventListener('pointermove', pointerMove);
      document.documentElement.removeEventListener('pointerleave', pointerLeave); offMotion();
      document.removeEventListener('visibilitychange', visibilityChange);
      renderer.domElement.removeEventListener('webglcontextlost', contextLost); renderer.domElement.removeEventListener('webglcontextrestored', contextRestored);
      chapters[0].labels.traverse(o => { if (o.element) o.element.remove(); });
      for (const m of [brightMat, blurMat, finalMat]) m.dispose(); quad.geometry.dispose();
      for (const c of chapters) c.scene.traverse(o => { o.geometry?.dispose(); if (o.material) for (const m of [].concat(o.material)) m.dispose(); });
      pullback.dispose(); shatter.dispose(); copyMat.dispose(); for (const t of [rtA, rtB, rtS1, rtS2, rtGlass]) t.dispose(); renderer.dispose(); renderer.domElement.remove(); cssRenderer.domElement.remove();
    }
  };
}
