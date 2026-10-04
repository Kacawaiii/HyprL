import * as THREE from 'three';
import { CSS3DRenderer, CSS3DObject } from './vendor/CSS3DRenderer.js';
import { GLTFExporter } from './vendor/GLTFExporter.js';

/** Procedural, editable scene. No image textures or generation service. */
export function createEclipseScene(container, labelContainer, { studio = false } = {}) {
  const media = matchMedia('(prefers-reduced-motion: reduce)');
  let paused = media.matches, visible = true, disposed = false, frame = 0;
  let elapsed = 0, previous = performance.now(), palette = 'gold';
  const pointer = new THREE.Vector2(), smoothPointer = new THREE.Vector2();
  const mobile = () => container.clientWidth < 600;
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true, powerPreference: 'low-power', preserveDrawingBuffer: studio }); }
  catch { document.body.classList.add('scene-unavailable'); return null; }
  renderer.setPixelRatio(Math.min(devicePixelRatio, mobile() ? 1.4 : 1.75));
  renderer.setClearColor(0x08090b, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.append(renderer.domElement);
  const cssRenderer = new CSS3DRenderer();
  labelContainer.append(cssRenderer.domElement);
  const scene = new THREE.Scene(), labels = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(43, 1, .1, 80);
  camera.position.set(0, 0, 14);
  const world = new THREE.Group(); world.name = 'HYPRL_Eclipse'; scene.add(world);
  const ambient = new THREE.HemisphereLight(0xc5d6f4, 0x08090b, 1.5); scene.add(ambient);
  const key = new THREE.PointLight(0xeddbc1, 45, 35, 2); key.position.set(0, 1.5, 6); scene.add(key);
  const fill = new THREE.PointLight(0x8caed8, 22, 25, 2); fill.position.set(-6, -1, 5); scene.add(fill);
  const gold = new THREE.Color('#dac09a'), ice = new THREE.Color('#a0c9e8');

  const vertex = 'varying vec2 vUv; void main(){vUv=uv; gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}';
  const atmosphereMaterial = new THREE.ShaderMaterial({
    uniforms: { uTime: {value:0}, uColor: {value:gold.clone()} },
    vertexShader: vertex,
    fragmentShader: `varying vec2 vUv; uniform float uTime; uniform vec3 uColor;
      float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
      float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);}
      float fbm(vec2 p){float f=0.;float a=.5;for(int i=0;i<4;i++){f+=a*noise(p);p=p*2.02+vec2(4.3,2.7);a*=.5;}return f;}
      void main(){vec2 p=(vUv-.5)*vec2(1.75,1.);float n=fbm(p*5.+vec2(uTime*.009,-uTime*.015));
        float upper=exp(-pow(p.x*3.2,2.)-pow((p.y-.05)*1.7,2.));
        float haze=exp(-pow(p.x*1.9,2.)-pow((p.y+.2)*6.,2.));
        vec3 color=mix(vec3(.22,.32,.48),uColor,.36)*(upper*.19+haze*.2)*(n*.8+.25);
        gl_FragColor=vec4(color,(upper*.55+haze*.45)*n);}`,
    transparent:true, depthWrite:false, blending:THREE.AdditiveBlending
  });
  const atmosphere = new THREE.Mesh(new THREE.PlaneGeometry(33, 22), atmosphereMaterial);
  atmosphere.name='Atmosphere'; atmosphere.position.set(0, 0, -9); world.add(atmosphere);

  const beamMaterial = new THREE.ShaderMaterial({
    uniforms:{uColor:{value:gold.clone()},uTime:{value:0}},vertexShader:vertex,
    fragmentShader:`varying vec2 vUv;uniform vec3 uColor;uniform float uTime;
      void main(){float x=abs(vUv.x-.5);float core=exp(-x*240.);float inner=exp(-x*44.)*.24;float outer=exp(-x*9.)*.035;
        float fade=smoothstep(0.,.13,vUv.y)*(1.-smoothstep(.65,1.,vUv.y));float pulse=.94+.06*sin(uTime*.45);
        gl_FragColor=vec4(uColor,(core+inner+outer)*fade*pulse);}`,
    transparent:true,depthWrite:false,blending:THREE.AdditiveBlending
  });
  const beam = new THREE.Mesh(new THREE.PlaneGeometry(4.2, 22),beamMaterial);
  beam.name='LightBeam';beam.position.set(0, 3, -6);world.add(beam);

  const flareMaterial=new THREE.ShaderMaterial({
    uniforms:{uColor:{value:gold.clone()}},vertexShader:vertex,
    fragmentShader:`varying vec2 vUv;uniform vec3 uColor;void main(){vec2 p=vUv-.5;float core=exp(-dot(p*vec2(18.,20.),p*vec2(18.,20.)));float streak=exp(-dot(p*vec2(3.,65.),p*vec2(3.,65.)));float glow=exp(-dot(p*vec2(5.,6.),p*vec2(5.,6.)));gl_FragColor=vec4(uColor,core*.42+streak*.14+glow*.08);}`,
    transparent:true,depthWrite:false,blending:THREE.AdditiveBlending
  });
  const flare=new THREE.Mesh(new THREE.PlaneGeometry(5,3),flareMaterial);flare.name='HorizonFlare';flare.position.set(0,-.82,.5);world.add(flare);

  const planetMaterial = new THREE.ShaderMaterial({
    uniforms:{uColor:{value:gold.clone()}},
    vertexShader:`varying vec3 vNormal;varying vec3 vView;varying vec3 vPosition;
      void main(){vec4 pos=modelViewMatrix*vec4(position,1.);vNormal=normalize(normalMatrix*normal);vView=normalize(-pos.xyz);vPosition=position;gl_Position=projectionMatrix*pos;}`,
    fragmentShader:`varying vec3 vNormal;varying vec3 vView;varying vec3 vPosition;uniform vec3 uColor;
      void main(){float facing=max(dot(normalize(vNormal),normalize(vView)),0.);float rim=pow(1.-facing,5.5);
        float north=smoothstep(-3.,8.,vPosition.y);float veins=sin(vPosition.x*14.+sin(vPosition.y*9.))*sin(vPosition.z*12.);
        vec3 body=vec3(.013,.016,.024)+veins*.0009;vec3 light=mix(vec3(.20,.31,.48),uColor*.57,.42);
        gl_FragColor=vec4(body+light*rim*north*.88,1.);}`
  });
  const planet = new THREE.Mesh(new THREE.SphereGeometry(9,96,64), planetMaterial);
  planet.name='Horizon';planet.position.set(0,-10.1,-3);world.add(planet);
  // Atmospheric silhouette just outside the sphere; depth keeps the interior dark.
  const haloMaterial = new THREE.ShaderMaterial({
    uniforms:{uColor:{value:gold.clone()}},
    vertexShader:`varying vec3 vNormal;varying vec3 vView;varying vec3 vPosition;void main(){vec4 p=modelViewMatrix*vec4(position,1.);vNormal=normalize(normalMatrix*normal);vView=normalize(-p.xyz);vPosition=position;gl_Position=projectionMatrix*p;}`,
    fragmentShader:`varying vec3 vNormal;varying vec3 vView;varying vec3 vPosition;uniform vec3 uColor;void main(){float rim=pow(1.-abs(dot(normalize(vNormal),normalize(vView))),6.);float north=smoothstep(-1.,8.,vPosition.y);gl_FragColor=vec4(mix(vec3(.34,.48,.69),uColor,.45),rim*north*.24);}`,
    transparent:true,depthWrite:false,side:THREE.BackSide,blending:THREE.AdditiveBlending
  });
  const halo = new THREE.Mesh(new THREE.SphereGeometry(9.08,80,48), haloMaterial);
  halo.name='HorizonAtmosphere';halo.position.copy(planet.position);world.add(halo);

  // Deterministic stars: stable composition on reload, no textures to download.
  let seed=27;const random=()=>{seed=(seed*16807)%2147483647;return(seed-1)/2147483646;};
  const starsGeometry=new THREE.BufferGeometry();const stars=[];
  for(let i=0;i<(mobile()?65:150);i++) stars.push((random()-.5)*32, random()*13-3,-8-random()*8);
  starsGeometry.setAttribute('position',new THREE.Float32BufferAttribute(stars,3));
  const starField=new THREE.Points(starsGeometry,new THREE.PointsMaterial({color:0xa8b8d0,size:.018,transparent:true,opacity:.47,sizeAttenuation:true,depthWrite:false}));
  starField.name='Stars';world.add(starField);

  function roundedShape(w,h,r){const s=new THREE.Shape(),x=-w/2,y=-h/2;s.moveTo(x+r,y);s.lineTo(x+w-r,y);s.quadraticCurveTo(x+w,y,x+w,y+r);s.lineTo(x+w,y+h-r);s.quadraticCurveTo(x+w,y+h,x+w-r,y+h);s.lineTo(x+r,y+h);s.quadraticCurveTo(x,y+h,x,y+h-r);s.lineTo(x,y+r);s.quadraticCurveTo(x,y,x+r,y);return s;}
  const panels=[];
  const chart=`<svg class="card-chart" viewBox="0 0 210 90" fill="none"><path d="M0 20H210M0 45H210M0 70H210" stroke="#aabbcf" stroke-opacity=".1"/><path d="M0 78L13 72L23 76L35 58L45 62L58 41L70 48L86 31L100 39L116 20L130 29L145 14L160 25L177 10L190 18L210 5" stroke="#b6c9e2" stroke-width="1.3"/><path d="M0 85L16 80L34 82L48 74L67 78L89 67L113 70L138 59L158 63L185 51L210 48" stroke="#d1bd9c" stroke-opacity=".45"/></svg>`;
  const specs=[
    {name:'LeftPanel',x:-5.1,y:-3.1,z:.6,ry:.25,rz:.075,width:490,height:282,html:`<div class="card-heading"><span><i class="card-symbol">⌁</i> Marchés</span><span class="card-tag">WATCHLIST</span></div><div class="card-subtitle">CRYPTO · FOREX · OR</div><div class="card-title">En perspective.</div><div class="card-meta">Votre univers, en un regard</div>${chart}<div class="card-footer"><span>VUE MULTI-ACTIFS<b>Une même interface</b></span><span>ESPACE<b>Personnalisé</b></span></div>`},
    {name:'RightPanel',x:5.1,y:-3.1,z:.6,ry:-.25,rz:-.075,width:490,height:282,html:`<div class="card-heading"><span><i class="card-symbol">◎</i> Gestion du risque</span><span class="card-tag">RISK VIEW</span></div><div class="card-subtitle">EXPOSITION · ALLOCATION</div><div class="card-title">Garder le contrôle.</div><div class="card-meta">Le détail fait la différence</div>${chart}<div class="card-footer"><span>PRIORITÉ<b>Votre exposition</b></span><span>APPROCHE<b>Structurée</b></span></div>`},
    {name:'CenterPanel',x:0,y:-2.45,z:2,ry:0,rz:0,width:530,height:320,html:`<div class="card-heading"><span><i class="card-symbol">H</i> HYPRL Workspace</span><span class="card-tag">APERÇU</span></div><div class="card-subtitle">STRATEGY OVERVIEW</div><div class="card-title">La vue d'ensemble.</div><div class="card-meta">Des informations qui font sens</div>${chart}<div class="card-footer"><span>MARCHÉS<b>Crypto · Forex · Or</b></span><span>STRATÉGIES<b>Votre espace</b></span><span>FOCUS<b>Clarté & risque</b></span></div><div class="card-glare"></div>`}
  ];
  for(const spec of specs){
    const group=new THREE.Group();group.name=spec.name;
    const shape=roundedShape(spec.width*.01,spec.height*.01,.22);
    const geometry=new THREE.ExtrudeGeometry(shape,{depth:.075,bevelEnabled:true,bevelSegments:3,steps:1,bevelSize:.025,bevelThickness:.02,curveSegments:12});
    const glass=new THREE.MeshPhysicalMaterial({color:0x222a38,metalness:.42,roughness:.23,transparent:true,opacity:.62,clearcoat:1,clearcoatRoughness:.1,side:THREE.DoubleSide});
    group.add(new THREE.Mesh(geometry,glass));
    const edge=new THREE.LineSegments(new THREE.EdgesGeometry(geometry,22),new THREE.LineBasicMaterial({color:0x93a9c6,transparent:true,opacity:.13}));group.add(edge);
    world.add(group);
    const element=document.createElement('div');element.className=`scene-card ${spec.name==='CenterPanel'?'center':''}`;element.innerHTML=spec.html;
    const label=new CSS3DObject(element);label.name=`${spec.name}_ReadableOverlay`;labels.add(label);
    panels.push({spec,group,label});
  }
  function positionPanels(){
    const scale=mobile()?.0073:.0085;
    for(const {spec,group,label} of panels){
      const ratio=scale/.01;group.scale.setScalar(ratio);
      group.position.set(spec.x*(mobile()?.69:1),spec.y+(mobile()?.06:0),spec.z);
      group.rotation.set(-.05,spec.ry,spec.rz);
      label.scale.setScalar(scale);label.position.copy(group.position);label.rotation.copy(group.rotation);label.position.z+=.10;
      // Side panels remain a visual frame; central panel stays legible on a phone.
      group.visible=label.visible=!mobile()||spec.name==='CenterPanel';
    }
  }
  function resize(){
    const w=container.clientWidth,h=container.clientHeight;if(!w||!h)return;
    renderer.setSize(w,h);cssRenderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();
    positionPanels();render();
  }
  function render(){renderer.render(scene,camera);cssRenderer.render(labels,camera);}
  const resizeObserver=new ResizeObserver(resize);resizeObserver.observe(container);
  const intersectionObserver=new IntersectionObserver(([entry])=>{visible=entry.isIntersecting;if(visible)previous=performance.now();});intersectionObserver.observe(container);
  function pointerMove(e){const r=container.getBoundingClientRect();pointer.set((e.clientX-r.left)/r.width*2-1,(e.clientY-r.top)/r.height*2-1);}
  function pointerLeave(){pointer.set(0,0);}
  const interaction=container.parentElement;interaction.addEventListener('pointermove',pointerMove);interaction.addEventListener('pointerleave',pointerLeave);
  function animate(now){
    if(disposed)return;frame=requestAnimationFrame(animate);
    if(!visible||document.hidden){previous=now;return;}
    const dt=Math.min((now-previous)/1000,.05);previous=now;
    if(paused)return;
    elapsed+=dt;atmosphereMaterial.uniforms.uTime.value=elapsed;beamMaterial.uniforms.uTime.value=elapsed;
    smoothPointer.lerp(pointer,.035);camera.position.x=smoothPointer.x*.13;camera.position.y=-smoothPointer.y*.07;camera.lookAt(0,0,0);
    for(const {spec,group,label} of panels){
      group.position.y=spec.y+(mobile()?.06:0)+Math.sin(elapsed*.38+spec.x)*.07;
      group.rotation.z=spec.rz+Math.sin(elapsed*.23+spec.x)*.008;
      label.position.copy(group.position);label.position.z+=.10;label.rotation.copy(group.rotation);
    }
    render();
  }
  function setPaused(value){paused=Boolean(value);if(paused){pointer.set(0,0);smoothPointer.set(0,0);camera.position.set(0,0,14);camera.lookAt(0,0,0);positionPanels();render();}previous=performance.now();}
  function setPalette(value){palette=value==='ice'?'ice':'gold';const color=palette==='ice'?ice:gold;for(const material of[atmosphereMaterial,beamMaterial,planetMaterial,haloMaterial,flareMaterial])material.uniforms.uColor.value.copy(color);key.color.copy(color);document.documentElement.style.setProperty('--accent',`#${color.getHexString()}`);render();}
  function mediaChange(){setPaused(media.matches);container.dispatchEvent(new CustomEvent('motionchange',{detail:{paused}}));}media.addEventListener('change',mediaChange);
  function contextLost(event){event.preventDefault();document.body.classList.remove('scene-ready');document.body.classList.add('scene-unavailable');}
  renderer.domElement.addEventListener('webglcontextlost',contextLost);
  renderer.domElement.addEventListener('webglcontextrestored',()=>{document.body.classList.remove('scene-unavailable');document.body.classList.add('scene-ready');resize();});
  function download(blob,filename){const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  async function exportGLB(){
    // Export portable geometry + PBR materials, not WebGL-only shader programs.
    const asset=new THREE.Scene();asset.name='HYPRL_Eclipse_Portable';
    const portablePlanet=new THREE.Mesh(planet.geometry,new THREE.MeshStandardMaterial({color:0x080b12,roughness:.8,metalness:.18}));portablePlanet.position.copy(planet.position);portablePlanet.name='Horizon';asset.add(portablePlanet);
    for(const {group} of panels){const copy=new THREE.Group();copy.name=group.name;copy.position.copy(group.position);copy.rotation.copy(group.rotation);copy.scale.copy(group.scale);copy.add(group.children[0].clone());asset.add(copy);}
    // Beam glow and atmosphere require re-lighting in the target renderer.
    const portableBeam=new THREE.Mesh(beam.geometry,new THREE.MeshStandardMaterial({color:palette==='ice'?ice:gold,emissive:palette==='ice'?ice:gold,emissiveIntensity:.35,transparent:true,opacity:.06,side:THREE.DoubleSide}));portableBeam.name='LightBeam';portableBeam.position.copy(beam.position);asset.add(portableBeam);
    asset.userData={description:'Editable HYPRL geometry. Procedural atmosphere, HTML overlays and animation are in scene.js.',palette};
    try{const result=await new GLTFExporter().parseAsync(asset,{binary:true});download(new Blob([result],{type:'model/gltf-binary'}),'hyprl-eclipse.glb');}
    finally{portablePlanet.material.dispose();portableBeam.material.dispose();}
  }
  function exportPNG(){render();renderer.domElement.toBlob(blob=>{if(blob)download(blob,'hyprl-eclipse-background.png');});}
  resize();render();document.body.classList.add('scene-ready');frame=requestAnimationFrame(animate);
  return {setPaused,setPalette,exportGLB,exportPNG,get paused(){return paused;},get stats(){return{drawCalls:renderer.info.render.calls,triangles:renderer.info.render.triangles};},dispose(){disposed=true;cancelAnimationFrame(frame);resizeObserver.disconnect();intersectionObserver.disconnect();interaction.removeEventListener('pointermove',pointerMove);interaction.removeEventListener('pointerleave',pointerLeave);media.removeEventListener('change',mediaChange);scene.traverse(o=>{o.geometry?.dispose();if(o.material)for(const m of Array.isArray(o.material)?o.material:[o.material])m.dispose();});renderer.dispose();renderer.domElement.remove();cssRenderer.domElement.remove();}};
}
