const {chromium}=require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const fs=require('fs'),crypto=require('crypto'),assert=require('assert');
const http=require('http'),path=require('path');
const server=http.createServer((req,res)=>{const file=path.resolve(__dirname,'..','.'+decodeURIComponent(req.url.split('?')[0]==='/'?'/index.html':req.url.split('?')[0]));const f=file.endsWith('/')?file+'index.html':file;fs.readFile(f,(err,data)=>{if(err){res.writeHead(404);res.end();return;}const ext=path.extname(f);res.setHeader('Content-Type',({'.js':'text/javascript','.html':'text/html','.css':'text/css','.svg':'image/svg+xml','.woff2':'font/woff2'})[ext]||'application/octet-stream');res.end(data)})});
let base;
const out=process.env.ECLIPSE_REPORT_DIR || require('os').tmpdir();
fs.mkdirSync(out,{recursive:true});
async function run(){
 await new Promise(r=>server.listen(0,'127.0.0.1',r));
 base=`http://127.0.0.1:${server.address().port}/`;
 const browser=await chromium.launch({headless:true,args:['--use-angle=swiftshader','--enable-unsafe-swiftshader','--ignore-gpu-blocklist']});
 try {
  for(const [w,h,tag] of [[1440,900,'desktop'],[390,844,'mobile']]){
   if(process.argv[2]&&process.argv[2]!==tag)continue;
   const page=await browser.newPage({viewport:{width:w,height:h},deviceScaleFactor:1});
   const errors=[],external=[];page.on('pageerror',e=>errors.push(e.message));page.on('console',m=>{if(m.type()==='error')errors.push(m.text())});
   await page.route('**/*',r=>{if(r.request().url().startsWith(base))return r.continue();external.push(r.request().url());return r.abort()});
   await page.addInitScript(()=>{let seq=0,q=new Map();let now=performance.now();requestAnimationFrame=cb=>{q.set(++seq,cb);return seq};cancelAnimationFrame=id=>q.delete(id);window.step=()=>{now+=16.6667;const old=[...q.values()];q.clear();for(const cb of old)cb(now);};});
   await page.goto(base,{waitUntil:'networkidle'});await page.evaluate(()=>{hyprlScene.setPaused(true);step()});assert.deepEqual(errors,[]);
   await page.evaluate(async()=>{
    const THREE=await import('./vendor/three.module.js'),zero=new THREE.Vector2();
    const [mono,planet,sing]=hyprlScene.chapters,{pullback}=hyprlScene.travelState;
    const captured=[],readback=document.createElement('canvas');readback.width=innerWidth;readback.height=innerHeight;
    const readContext=readback.getContext('2d',{willReadFrequently:true});
    window.captureCanvas=(direction,j)=>{
     const canvas=document.querySelector('#scene canvas'),blend=hyprlScene.travelState.blend;
     const pose=[blend.a,blend.b,blend.t,blend.dim];
     for(const index of (blend.t>.001?[blend.a,blend.b]:[blend.a])){
      const c=hyprlScene.chapters[index].camera;pose.push(...c.position.toArray(),...c.quaternion.toArray(),...c.up.toArray(),...c.projectionMatrix.elements);
     }
     readContext.drawImage(canvas,0,0,innerWidth,innerHeight);
     const pixels=readContext.getImageData(0,0,innerWidth,innerHeight).data;
     let check=null;
     if(direction==='forward')captured[j]={pose,pixels};
     else {
      const expected=captured[8-j];let maxDelta=0,changedPixels=0,poseMaxDelta=0;
      for(let i=0;i<pose.length;i++)poseMaxDelta=Math.max(poseMaxDelta,Math.abs(pose[i]-expected.pose[i]));
      for(let i=0;i<pixels.length;i+=4){let changed=false;for(let c=0;c<3;c++){const delta=Math.abs(pixels[i+c]-expected.pixels[i+c]);maxDelta=Math.max(maxDelta,delta);changed ||= delta>0;}if(changed)changedPixels++;}
      check={j,poseMaxDelta,maxDelta,changedPixels,totalPixels:innerWidth*innerHeight};
     }
     return {png:canvas.toDataURL(),check};
    };
    window.edgeLocal=(name,id,role)=>{
     const group=[...document.querySelectorAll('[data-chapter]')].filter(el=>el.dataset.chapter===name);
     const y=document.getElementById(id).getBoundingClientRect().top+scrollY-innerHeight*(role==='out'?.9:.2);
     const top=group[0].getBoundingClientRect().top+scrollY,bottom=group.at(-1).getBoundingClientRect().bottom+scrollY;
     return Math.max(0,Math.min(1,(innerHeight-top+y)/(bottom-top+innerHeight)));
    };
    function sphere(camera,center,radius){
     camera.updateMatrixWorld(true);
     const d=camera.position.distanceTo(center),f=innerHeight/2/Math.tan(THREE.MathUtils.degToRad(camera.fov/2));
     const c=center.clone().applyMatrix4(camera.matrixWorldInverse),z=-c.z;
     const angular=f*radius/Math.sqrt(d*d-radius*radius);
     // Perspective silhouette conic: major semiaxis. Near the surface the disc crosses
     // the camera plane; cap the unbounded extent to the viewport circumradius.
     const major=z>radius?f*radius*Math.sqrt(d*d-radius*radius)/(z*z-radius*radius):Infinity;
     return {planetAngularRadiusPx:angular,planetDiscRadiusPx:Math.min(Math.hypot(innerWidth,innerHeight)/2,major),planetUnclippedDiscRadiusPx:Number.isFinite(major)?major:null};
    }
    window.measureCurrent=(kind)=>{
     const blend=hyprlScene.travelState.blend;
     const transitionProgress=kind==='pullback'?(blend.a===0?blend.t:1):(blend.a===1?blend.t:1);
     if(kind==='pullback'){
      const camera=transitionProgress===1?planet.camera:mono.camera;
      const center=transitionProgress===1?new THREE.Vector3():pullback.center;
      const radius=transitionProgress===1?planet.radius:pullback.radius;
      const corners=[];mono.surface.monolith.updateMatrixWorld(true);
      for(const x of [-.9,.9])for(const y of [-2.45,2.45])for(const z of [-.275,.275]){
       const p=new THREE.Vector3(x,y,z).applyMatrix4(mono.surface.monolith.matrixWorld).project(mono.camera);corners.push(p.y*innerHeight/2);
      }
      return {transitionProgress,monolithHeightPx:transitionProgress===1?null:Math.max(...corners)-Math.min(...corners),...sphere(camera,center,radius)};
     }
     const u=sing.scene.children.find(o=>o.material?.uniforms?.uRs).material.uniforms;
     return {transitionProgress,...sphere(planet.camera,new THREE.Vector3(),planet.radius),singularityRingRadiusPx:u.uRs.value*(transitionProgress===0?1.08:u.uArrival.value)*innerHeight,ringRevealed:transitionProgress>.48};
    };
    window.samplePlanetRest=(t)=>{
     const entry=edgeLocal('planet','vision','in'),exit=edgeLocal('planet','horizon','out');
     planet.update({time:0,dt:0,pointer:zero,motion:0,local:entry+(exit-entry)*t,entryLocal:entry});
     return sphere(planet.camera,new THREE.Vector3(),planet.radius);
    };
    window.sampleSubjects=(kind,t)=>{
     const common={time:0,dt:0,pointer:zero,motion:0,pr:1};
     if(kind==='pullback'){
      pullback.reset();mono.update({...common,local:edgeLocal('monolith','vision','out')});
      planet.update({...common,local:edgeLocal('planet','vision','in')});pullback.update(t);
      const corners=[];mono.surface.monolith.updateMatrixWorld(true);
      for(const x of [-.9,.9])for(const y of [-2.45,2.45])for(const z of [-.275,.275]){
       const p=new THREE.Vector3(x,y,z).applyMatrix4(mono.surface.monolith.matrixWorld).project(mono.camera);corners.push(p.y*innerHeight/2);
      }
      return {monolithHeightPx:Math.max(...corners)-Math.min(...corners),...sphere(mono.camera,pullback.center,pullback.radius)};
     }
     planet.update({...common,local:edgeLocal('planet','horizon','out'),entryLocal:edgeLocal('planet','vision','in'),transition:{role:'out',progress:t,kind:1}});
     sing.update({...common,local:edgeLocal('singularity','horizon','in'),transition:{role:'in',progress:t,kind:1}});
     const backdrop=sing.scene.children.find(o=>o.material?.uniforms?.uRs).material.uniforms;
     return {...sphere(planet.camera,new THREE.Vector3(),planet.radius),singularityRingRadiusPx:backdrop.uRs.value*backdrop.uArrival.value*innerHeight};
    };
   });
   const rest=await page.evaluate(()=>Array.from({length:101},(_,i)=>({t:i/100,...samplePlanetRest(i/100)})));
   for(let i=1;i<rest.length;i++)assert(rest[i].planetDiscRadiusPx<=rest[i-1].planetDiscRadiusPx*1.02,`${tag} planet chapter grows`);
   const records=[],dense=[],reversal=[];
   const keys={pullback:['monolithHeightPx','planetAngularRadiusPx','planetDiscRadiusPx'],warp:['planetAngularRadiusPx','planetDiscRadiusPx','singularityRingRadiusPx']};
   for(const kind of ['pullback','warp']){
    if(process.argv[3]&&kind!==process.argv[3])continue;
    const samples=await page.evaluate(kind=>Array.from({length:201},(_,i)=>({t:i/200,...sampleSubjects(kind,i/200)})),kind);
    for(const key of keys[kind])for(let i=1;i<samples.length;i++)assert(samples[i][key]<=samples[i-1][key]*1.02,`${tag} ${kind} ${key} grows at ${samples[i].t}: ${samples[i-1][key]} -> ${samples[i][key]}`);
    dense.push({kind,samples});
   }
   for(const {kind,samples} of dense){
    const a=kind==='pullback'?samples.at(-1).planetDiscRadiusPx:rest.at(-1).planetDiscRadiusPx;
    const b=kind==='pullback'?rest[0].planetDiscRadiusPx:samples[0].planetDiscRadiusPx;
    assert(Math.abs(a-b)<1e-7,`${tag} ${kind}: planet size changes at chapter boundary`);
   }
   await page.evaluate(()=>{dispatchEvent(new Event('scroll'));step()});
   for(const [id,kind]of[['vision','pullback'],['horizon','warp']]){
    if(process.argv[3]&&kind!==process.argv[3])continue;
    const top=await page.evaluate(id=>document.getElementById(id).getBoundingClientRect().top+scrollY,id);
    const hashes={forward:[],backward:[]},pixelChecks=[];
    for(const direction of (process.argv[4]==='debug'?['forward']:['forward','backward']))for(let j=0;j<9;j++){
     const t=(direction==='forward'?j:8-j)/8;let lo=0,hi=1;for(let n=0;n<30;n++){const m=(lo+hi)/2;if(m*m*(3-2*m)<t)lo=m;else hi=m}
     await page.evaluate(y=>{scrollTo({top:y,behavior:'instant'});dispatchEvent(new Event('scroll'))},top-h*(.9-.7*(lo+hi)/2));await page.waitForTimeout(50);const capture=await page.evaluate(({direction,j})=>{step();return captureCanvas(direction,j)},{direction,j}),canvas=capture.png;
     if(capture.check)pixelChecks.push(capture.check);
     const file=`${out}/norezoom-${tag}-${kind}-${direction}-${j}.jpg`;
     await page.screenshot({path:file,type:'jpeg',quality:87});
     hashes[direction].push(crypto.createHash('sha256').update(canvas).digest('hex'));
     if(process.env.ECLIPSE_CANVAS_DEBUG)fs.writeFileSync(file.replace('.jpg','-canvas.png'),Buffer.from(canvas.split(',')[1],'base64'));
     const m=await page.evaluate(kind=>measureCurrent(kind),kind);
     records.push({tag,kind,direction,j,t,...m});console.log(JSON.stringify(records.at(-1)));
    }
    assert(new Set(hashes.forward).size>=4,`${tag} ${kind}: canvas samples must show different rendered frames`);
    if(hashes.backward.length){
     for(const check of pixelChecks){
      assert(check.poseMaxDelta<1e-10,`${tag} ${kind}: reverse scene pose differs`);
      // Half-float bloom/MSAA in SwiftShader can round one RGB unit differently on a handful of pixels.
      // This is stricter than a perceptual screenshot diff and cannot conceal a camera or scale change.
      assert(check.maxDelta<=1&&check.changedPixels<=check.totalPixels*.00001,`${tag} ${kind}: reverse pixels differ beyond 8-bit rounding: ${JSON.stringify(check)}`);
     }
     reversal.push({kind,identicalScenePoses:true,identicalPixels:pixelChecks.every(c=>c.changedPixels===0),pixelChecks,canvasHashes:hashes.forward});
     console.log(JSON.stringify(reversal.at(-1)));
    }
   }
   for(const kind of ['pullback','warp'])for(const direction of ['forward','backward']){
    const frames=records.filter(r=>r.kind===kind&&r.direction===direction);
    for(const key of keys[kind])for(let i=1;i<frames.length;i++){
     if(frames[i][key]==null||frames[i-1][key]==null)continue;
     const a=direction==='forward'?frames[i-1][key]:frames[i][key],b=direction==='forward'?frames[i][key]:frames[i-1][key];
     assert(b<=a*1.02,`${tag} ${kind} ${direction} frame ${i}: ${key} grows`);
    }
   }
   const reduced=[];
   if(process.argv[4]!=='debug'){
    await page.emulateMedia({reducedMotion:'reduce'});await page.waitForTimeout(50);
    for(const [id,kind]of[['vision','pullback'],['horizon','warp']]){
     if(process.argv[3]&&kind!==process.argv[3])continue;
     const top=await page.evaluate(id=>document.getElementById(id).getBoundingClientRect().top+scrollY,id);
     const poses=[];
     for(const t of [.25,.5,.75]){
      let lo=0,hi=1;for(let n=0;n<30;n++){const m=(lo+hi)/2;if(m*m*(3-2*m)<t)lo=m;else hi=m}
      await page.evaluate(y=>{scrollTo({top:y,behavior:'instant'});dispatchEvent(new Event('scroll'))},top-h*(.9-.7*(lo+hi)/2));
      await page.waitForTimeout(40);await page.evaluate(()=>step());
      poses.push(await page.evaluate(()=>({paused:hyprlScene.paused,surface:hyprlScene.travelState.pullback.globe.visible,cameras:hyprlScene.chapters.map(c=>[...c.camera.position.toArray(),...c.camera.quaternion.toArray()]),arrival:hyprlScene.chapters[2].scene.children.find(o=>o.material?.uniforms?.uArrival).material.uniforms.uArrival.value})));
      if(t===.5)await page.screenshot({path:`${out}/norezoom-${tag}-${kind}-reduced.jpg`,type:'jpeg',quality:87});
     }
     for(const pose of poses){assert(pose.paused);assert(!pose.surface);assert.equal(pose.arrival,1);}
     const indices=kind==='pullback'?[0,1]:[1,2];for(const i of indices){assert.deepEqual(poses[0].cameras[i],poses[1].cameras[i]);assert.deepEqual(poses[1].cameras[i],poses[2].cameras[i]);}
     reduced.push({kind,fixedPoses:true,simpleFade:true});
    }
    await page.locator('[data-rail="prism"]').focus();await page.keyboard.press('Enter');await page.waitForTimeout(60);await page.evaluate(()=>step());
    assert.equal(await page.evaluate(()=>document.activeElement.id),'plateforme');assert.equal(await page.evaluate(()=>hyprlScene.chapter),'prism');
    await page.emulateMedia({reducedMotion:'no-preference'});await page.waitForTimeout(40);await page.evaluate(()=>step());assert(await page.evaluate(()=>hyprlScene.paused));
   }
   assert.deepEqual(errors,[]);assert.deepEqual(external,[]);fs.writeFileSync(`${out}/norezoom-${tag}-${process.argv[3]||'all'}-sizes.json`,JSON.stringify({records,dense,planetRest:rest,reversal,reduced,errors,external},null,2));await page.close();
  }
 }finally{await browser.close();server.close()}
}
run().catch(e=>{console.error(e);process.exit(1)});
