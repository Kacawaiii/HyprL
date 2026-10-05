import * as THREE from 'three';

/** Shared GLSL: hash, value noise, fbm, ACES. Pure procedural, no textures. */
export const NOISE = /* glsl */`
float hash12(vec2 p){vec3 p3=fract(vec3(p.xyx)*.1031);p3+=dot(p3,p3.yzx+33.33);return fract((p3.x+p3.y)*p3.z);}
float vnoise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);
  return mix(mix(hash12(i),hash12(i+vec2(1,0)),f.x),mix(hash12(i+vec2(0,1)),hash12(i+vec2(1,1)),f.x),f.y);}
float fbm(vec2 p){float v=0.,a=.5;for(int i=0;i<5;i++){v+=a*vnoise(p);p=p*2.03+vec2(1.7,9.2);a*=.5;}return v;}
float fbm3(vec2 p){float v=0.,a=.5;for(int i=0;i<3;i++){v+=a*vnoise(p);p=p*2.03+vec2(1.7,9.2);a*=.5;}return v;}
`;

export const VERT_UV = /* glsl */`varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}`;
export const VERT_SCREEN = /* glsl */`varying vec2 vUv;void main(){vUv=uv;gl_Position=vec4(position.xy,0.,1.);}`;
/** World position + world normal, instancing aware. */
export const VERT_WORLD = /* glsl */`varying vec3 vW;varying vec3 vN;varying vec2 vUv;
void main(){mat4 m=modelMatrix;
  #ifdef USE_INSTANCING
  m=modelMatrix*instanceMatrix;
  #endif
  vec4 w=m*vec4(position,1.);vW=w.xyz;vN=normalize(mat3(m)*normal);vUv=uv;gl_Position=projectionMatrix*viewMatrix*w;}`;

/** Smooth normal perturbed by a height field h(p) through screen derivatives: per-pixel relief, no triangle facets. */
export const BUMP = /* glsl */`vec3 bumpN(vec3 n,vec3 p,float h){vec3 sx=dFdx(p),sy=dFdy(p);vec3 r1=cross(sy,n),r2=cross(n,sx);float det=dot(sx,r1);vec3 g=sign(det)*(dFdx(h)*r1+dFdy(h)*r2);return normalize(abs(det)*n-g);}`;

/** Additive glow material (light, flares, rings). */
export function glow(fragmentShader, uniforms = {}, extra = {}) {
  return new THREE.ShaderMaterial({ uniforms, vertexShader: extra.vertexShader || VERT_UV, fragmentShader,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: extra.side ?? THREE.FrontSide, depthTest: extra.depthTest ?? true });
}

/** Deterministic random: stable composition on every reload. */
export function rng(seed) { let s = seed % 2147483647; if (s <= 0) s += 2147483646; return () => { s = (s * 16807) % 2147483647; return (s - 1) / 2147483646; }; }

function hash3(x, y, z) { const h = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453; return h - Math.floor(h); }
export function noise3(x, y, z) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const s = t => t * t * (3 - 2 * t), fx = s(x - xi), fy = s(y - yi), fz = s(z - zi);
  const l = (a, b, t) => a + (b - a) * t;
  const c = (i, j, k) => hash3(xi + i, yi + j, zi + k);
  return l(l(l(c(0,0,0), c(1,0,0), fx), l(c(0,1,0), c(1,1,0), fx), fy), l(l(c(0,0,1), c(1,0,1), fx), l(c(0,1,1), c(1,1,1), fx), fy), fz);
}
export const noise2 = (x, y) => noise3(x, y, .5);

/** Irregular asteroid: displaced icosahedron, smooth shaded with procedural surface relief. */
export function rockGeometry(seed, detail = 2) {
  const g = new THREE.IcosahedronGeometry(1, detail), p = g.attributes.position, v = new THREE.Vector3();
  const r = rng(seed * 97 + 13), sx = .75 + r() * .5, sy = .6 + r() * .45, sz = .8 + r() * .4;
  for (let i = 0; i < p.count; i++) {
    v.fromBufferAttribute(p, i).normalize();
    const d = .78 + .42 * noise3(v.x * 1.7 + seed, v.y * 1.7, v.z * 1.7) + .16 * noise3(v.x * 4.3, v.y * 4.3 + seed, v.z * 4.3) - .12 * Math.abs(noise3(v.x * 9, v.y * 9, v.z * 9 + seed) - .5);
    p.setXYZ(i, v.x * d * sx, v.y * d * sy, v.z * d * sz);
  }
  g.computeVertexNormals(); g.name = `Rock_${seed}`;
  return g;
}

/** Back-lit rock (rim light from a point: eclipse, sun, disc), smooth normals plus fine relief in object space. */
export function rockMaterial({ light = new THREE.Vector3(), rim = new THREE.Color('#dac09a'), fill = new THREE.Color('#8caed8'), base = new THREE.Color('#0a0b0e') } = {}) {
  return new THREE.ShaderMaterial({
    uniforms: { uLight: { value: light }, uRim: { value: rim }, uFill: { value: fill }, uBase: { value: base }, uFade: { value: 1 } },
    vertexShader: /* glsl */`varying vec3 vW;varying vec3 vN;varying vec3 vL;
      void main(){mat4 m=modelMatrix;
        #ifdef USE_INSTANCING
        m=modelMatrix*instanceMatrix;
        #endif
        vec4 w=m*vec4(position,1.);vW=w.xyz;vN=normalize(mat3(m)*normal);vL=position;gl_Position=projectionMatrix*viewMatrix*w;}`,
    fragmentShader: NOISE + BUMP + /* glsl */`varying vec3 vW;varying vec3 vN;varying vec3 vL;uniform vec3 uLight,uRim,uFill,uBase;uniform float uFade;
      void main(){float h=fbm(vL.xy*5.+vL.z*3.1)*.05+fbm(vL.yz*13.)*.018;vec3 n=bumpN(normalize(vN),vW,h);
        vec3 v=normalize(cameraPosition-vW);vec3 l=normalize(uLight-vW);
        float diff=max(dot(n,l),0.);float rim=pow(clamp(1.-dot(n,v),0.,1.),2.2);
        float front=max(dot(n,normalize(vec3(-.35,.45,1.))),0.);
        float edge=pow(rim,1.4);vec3 col=uBase*(.3+front*.7)+uFill*front*.018+uRim*(diff*diff*.16+edge*(.12+diff*1.1));
        gl_FragColor=vec4(col*uFade,1.);}`
  });
}

/** Twinkling star field as points. */
export function starField({ count, spread, depth, seed = 27, color = '#a8b8d0', size = [1.6, 4] }) {
  const r = rng(seed), pos = [], aSize = [], aPhase = [];
  for (let i = 0; i < count; i++) { pos.push((r() - .5) * spread[0], (r() - .5) * spread[1], depth[0] - r() * (depth[1] - depth[0])); aSize.push(size[0] + Math.pow(r(), 3) * (size[1] - size[0])); aPhase.push(r()); }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('aSize', new THREE.Float32BufferAttribute(aSize, 1));
  g.setAttribute('aPhase', new THREE.Float32BufferAttribute(aPhase, 1));
  const m = new THREE.ShaderMaterial({
    uniforms: { uTime: { value: 0 }, uPR: { value: 1 }, uColor: { value: new THREE.Color(color) }, uFade: { value: 1 } },
    vertexShader: /* glsl */`attribute float aSize;attribute float aPhase;uniform float uTime,uPR;varying float vA;
      void main(){vec4 mv=modelViewMatrix*vec4(position,1.);gl_Position=projectionMatrix*mv;gl_PointSize=aSize*uPR*clamp(24./-mv.z,.6,3.);vA=.55+.45*sin(uTime*(.5+aPhase)+aPhase*40.);}`,
    fragmentShader: /* glsl */`varying float vA;uniform vec3 uColor;uniform float uFade;void main(){float d=length(gl_PointCoord-.5);float a=smoothstep(.5,0.,d);gl_FragColor=vec4(uColor*a*a*vA*uFade*1.6,1.);}`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending
  });
  const p = new THREE.Points(g, m); p.name = 'Stars'; p.frustumCulled = false;
  return p;
}

/** Visible half-size of the frustum at depth z for a camera on the z axis. */
export function halfSize(camera, z, aspect = camera.aspect) {
  const h = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * (camera.position.z - z);
  return { h, w: h * aspect };
}

export const smooth = (a, b, x) => { const t = Math.min(Math.max((x - a) / (b - a), 0), 1); return t * t * (3 - 2 * t); };

/** Portable copy for glTF: geometry + a standard PBR material. */
export function portable(mesh, params) {
  const m = new THREE.Mesh(mesh.geometry, new THREE.MeshStandardMaterial(params));
  m.name = mesh.name; mesh.updateWorldMatrix(true, false); m.applyMatrix4(mesh.matrixWorld);
  return m;
}
export function portableInstances(inst, params, name, geometry = inst.geometry) {
  const g = new THREE.Group(); g.name = name; const mat = new THREE.MeshStandardMaterial(params), m4 = new THREE.Matrix4();
  inst.updateWorldMatrix(true, false);
  for (let i = 0; i < inst.count; i++) { inst.getMatrixAt(i, m4); const m = new THREE.Mesh(geometry, mat); m.name = `${name}_${i}`; m.applyMatrix4(m4.premultiply(inst.matrixWorld)); g.add(m); }
  return g;
}
