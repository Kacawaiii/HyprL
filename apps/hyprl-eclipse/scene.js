import * as THREE from 'three';
import { CSS3DRenderer } from './vendor/CSS3DRenderer.js';
import { GLTFExporter } from './vendor/GLTFExporter.js';
import { VERT_SCREEN, smooth } from './lib/kit.js';
import { createEclipseChapter } from './chapters/eclipse.js';
import { createMonolithChapter } from './chapters/monolith.js';
import { createPrismChapter } from './chapters/prism.js';
import { createSingularityChapter } from './chapters/singularity.js';

export const CHAPTERS = ['eclipse', 'monolith', 'prism', 'singularity'];

/**
 * HYPRL universe: four procedural 3D chapters behind the page, cross-faded on scroll,
 * composited with chromatic aberration, bloom, vignette and grain. No textures, no network.
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
  const chapters = [createEclipseChapter(ctx), createMonolithChapter(ctx), createPrismChapter(ctx), createSingularityChapter(ctx)];
  const byName = Object.fromEntries(chapters.map((c, i) => [c.name, i]));

  // ── Chapter transition, shared by the bloom and final passes ──
  // The camera flies into chapter A while B settles from a slight zoom, revealed from the centre by an
  // expanding, ragged ring of light (an eclipse limb opening onto the next scene).
  const TRANSITION = /* glsl */`uniform float uMix,uTime,uAspect;
    float tHash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
    float tNoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
      return mix(mix(tHash(i),tHash(i+vec2(1,0)),f.x),mix(tHash(i+vec2(0,1)),tHash(i+vec2(1,1)),f.x),f.y);}
    vec2 zoomA(vec2 uv){return (uv-.5)/(1.+.35*uMix*uMix)+.5;}
    vec2 zoomB(vec2 uv){float k=1.-uMix;return (uv-.5)/(1.+.24*k*k)+.5;}
    float ringEdge(){return mix(-.25,1.45,uMix);}
    float ringDist(vec2 uv){vec2 p=(uv-.5)*vec2(uAspect,1.);float r=length(p)/(.5*length(vec2(uAspect,1.)));
      vec2 d=p/(length(p)+1e-4);return r+(tNoise(d*1.6+uTime*.1)-.5)*.07+(tNoise(uv*9.+uTime*.05)-.5)*.02;}
    float reveal(float r){float e=ringEdge();return uMix>.999?1.:1.-smoothstep(e-.09,e+.008,r);}`;

  // ── Post-processing chain (all hand-written, three core only) ──
  const rtOpts = { type: THREE.HalfFloatType, depthBuffer: true, samples: isMobile() ? 2 : 4 };
  const rtA = new THREE.WebGLRenderTarget(1, 1, rtOpts), rtB = new THREE.WebGLRenderTarget(1, 1, rtOpts);
  const small = { type: THREE.HalfFloatType, depthBuffer: false };
  const rtS1 = new THREE.WebGLRenderTarget(1, 1, small), rtS2 = new THREE.WebGLRenderTarget(1, 1, small);
  const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1), quadScene = new THREE.Scene();
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2)); quad.frustumCulled = false; quadScene.add(quad);
  const brightMat = new THREE.ShaderMaterial({ uniforms: { tA: { value: null }, tB: { value: null }, uMix: { value: 0 }, uTime: { value: 0 }, uAspect: { value: 1 }, uTexel: { value: new THREE.Vector2() } }, vertexShader: VERT_SCREEN, depthTest: false, depthWrite: false,
    fragmentShader: /* glsl */`varying vec2 vUv;uniform sampler2D tA,tB;uniform vec2 uTexel;${TRANSITION}
      vec3 s(vec2 uv){if(uMix<.001)return texture2D(tA,uv).rgb;return mix(texture2D(tA,zoomA(uv)).rgb,texture2D(tB,zoomB(uv)).rgb,reveal(ringDist(uv)));}
      void main(){vec3 c=(s(vUv+uTexel*vec2(-1,-1))+s(vUv+uTexel*vec2(1,-1))+s(vUv+uTexel*vec2(-1,1))+s(vUv+uTexel*vec2(1,1)))*.25;
        float l=dot(c,vec3(.2126,.7152,.0722));gl_FragColor=vec4(c*smoothstep(.35,1.2,l),1.);}` });
  const blurMat = new THREE.ShaderMaterial({ uniforms: { tIn: { value: null }, uDir: { value: new THREE.Vector2() } }, vertexShader: VERT_SCREEN, depthTest: false, depthWrite: false,
    fragmentShader: /* glsl */`varying vec2 vUv;uniform sampler2D tIn;uniform vec2 uDir;
      void main(){vec3 c=texture2D(tIn,vUv).rgb*.227;c+=(texture2D(tIn,vUv+uDir*1.385).rgb+texture2D(tIn,vUv-uDir*1.385).rgb)*.316;c+=(texture2D(tIn,vUv+uDir*3.23).rgb+texture2D(tIn,vUv-uDir*3.23).rgb)*.07;gl_FragColor=vec4(c,1.);}` });
  const finalMat = new THREE.ShaderMaterial({
    uniforms: { tA: { value: null }, tB: { value: null }, tBloom: { value: null }, uMix: { value: 0 }, uCAa: { value: 0 }, uCAb: { value: 0 }, uBloom: { value: 1 }, uExpo: { value: 1 }, uDim: { value: 0 }, uSat: { value: 1 }, uTime: { value: 0 }, uAspect: { value: 1 }, uAccent: { value: new THREE.Color() }, uRes: { value: new THREE.Vector2() } },
    vertexShader: VERT_SCREEN, depthTest: false, depthWrite: false,
    fragmentShader: /* glsl */`varying vec2 vUv;uniform sampler2D tA,tB,tBloom;uniform float uCAa,uCAb,uBloom,uExpo,uDim,uSat;uniform vec2 uRes;uniform vec3 uAccent;${TRANSITION}
      vec3 ca(sampler2D t,vec2 uv,float s){vec2 d=(uv-.5)*s;return vec3(texture2D(t,uv+d).r,texture2D(t,uv).g,texture2D(t,uv-d).b);}
      vec3 rblur(sampler2D t,vec2 uv,float s){vec3 c=vec3(0.);vec2 d=(uv-.5)*s;for(int i=0;i<7;i++)c+=texture2D(t,uv-d*(float(i)/6.)).rgb;return c/7.;}
      vec3 aces(vec3 x){return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.);}
      float h(vec2 p){return fract(sin(dot(p,vec2(12.9898,78.233)))*43758.5453);}
      void main(){vec3 col;float seam=0.;
        if(uMix<.001)col=ca(tA,vUv,uCAa);
        else{vec2 ua=zoomA(vUv),ub=zoomB(vUv);
          vec3 a=mix(ca(tA,ua,uCAa),rblur(tA,ua,.09*uMix),smoothstep(0.,.35,uMix))*(1.-.4*uMix);
          vec3 b=mix(ca(tB,ub,uCAb),rblur(tB,ub,.07*(1.-uMix)),smoothstep(0.,.35,1.-uMix))*(.65+.35*uMix);
          float r=ringDist(vUv),e=ringEdge(),pulse=sin(3.14159*uMix);col=mix(a,b,reveal(r));
          seam=(exp(-pow((r-e)/.009,2.))*.75+exp(-pow((r-e)/.04,2.))*.16)*pulse;}
        col+=texture2D(tBloom,vUv).rgb*uBloom;col*=uExpo;col=mix(vec3(dot(col,vec3(.2126,.7152,.0722))),col,uSat);
        col+=mix(vec3(1.),uAccent,.45)*seam;
        vec2 p=vUv-.5;col*=1.-dot(p,p)*.85;col*=1.-uDim;
        col=aces(col);col=pow(col,vec3(1./2.2));
        col+=(h(vUv*uRes+fract(uTime*7.))-.5)*.022;
        gl_FragColor=vec4(col,1.);}` });

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
    chapters[0].resize(W, heroH, H); for (const c of chapters.slice(1)) c.resize(W, H);
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
    renderer.setRenderTarget(target); renderer.render(c.scene, c.camera);
  }
  let lastBlend = null;
  function render(dt = 0) {
    renderer.info.reset();
    const s = readScroll(dt); s.dt = dt; lastBlend = s;
    const A = chapters[s.a], B = chapters[s.b], mixT = s.a === s.b ? 0 : s.t;
    renderChapter(A, rtA, s, PR); if (mixT > .001) renderChapter(B, rtB, s, PR);
    const pa = A.post, pb = mixT > .001 ? B.post : A.post, lerp = (x, y) => x + (y - x) * mixT;
    quad.material = brightMat; Object.assign(brightMat.uniforms.tA, { value: rtA.texture }); brightMat.uniforms.tB.value = rtB.texture; brightMat.uniforms.uMix.value = mixT; brightMat.uniforms.uTime.value = elapsed;
    renderer.setRenderTarget(rtS1); renderer.render(quadScene, quadCam);
    quad.material = blurMat;
    for (const [src, dst, dir] of [[rtS1, rtS2, [1, 0]], [rtS2, rtS1, [0, 1]], [rtS1, rtS2, [2, 0]], [rtS2, rtS1, [0, 2]]]) {
      blurMat.uniforms.tIn.value = src.texture; blurMat.uniforms.uDir.value.set(dir[0] / rtS1.width, dir[1] / rtS1.height); renderer.setRenderTarget(dst); renderer.render(quadScene, quadCam);
    }
    quad.material = finalMat; const u = finalMat.uniforms;
    u.tA.value = rtA.texture; u.tB.value = rtB.texture; u.tBloom.value = rtS1.texture; u.uMix.value = mixT;
    u.uCAa.value = pa.ca; u.uCAb.value = pb.ca; u.uBloom.value = lerp(pa.bloom, pb.bloom); u.uExpo.value = lerp(pa.exposure, pb.exposure); u.uDim.value = s.dim; u.uSat.value = lerp(pa.sat ?? 1, pb.sat ?? 1); u.uTime.value = elapsed;
    renderer.setRenderTarget(null); renderer.render(quadScene, quadCam);
    // Readable overlays only while the eclipse chapter is on screen.
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
  function setPalette(value) { palette = value === 'ice' ? 'ice' : 'gold'; const color = palette === 'ice' ? ice : gold; for (const c of chapters) c.setPalette(color, palette); finalMat.uniforms.uAccent.value.copy(color); document.documentElement.style.setProperty('--accent', `#${color.getHexString()}`); dirty = true; }
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
    asset.userData = { description: 'HYPRL — 4 chapitres (Éclipse, Monolithe, Prisme, Singularité). Géométrie éditable ; shaders, lumière et animation dans chapters/*.js.', palette };
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
      for (const t of [rtA, rtB, rtS1, rtS2]) t.dispose(); renderer.dispose(); renderer.domElement.remove(); cssRenderer.domElement.remove();
    }
  };
}
