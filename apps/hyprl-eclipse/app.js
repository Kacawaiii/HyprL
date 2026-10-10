import { createEclipseScene } from './scene.js';
import { createSmoothScroll } from './lib/smooth-scroll.js';
import { motionPref } from './lib/motion.js';
const studio=document.body.classList.contains('studio-page');
// Inertial scroll (wheel, fine pointer): the scene drives it so the page and the WebGL frame move together.
const scroller=studio?null:createSmoothScroll();
const scene=createEclipseScene(document.querySelector('#scene'),document.querySelector('#scene-labels'),{studio,hero:document.querySelector('.hero'),stops:[...document.querySelectorAll('[data-chapter]')],onFrame:scroller?.update});
if(scene&&scroller)scroller.drive();
// A single, short discovery reveal. Retargetable opacity/transform transitions; no blur or scale.
const revealGroups=[['.hero','.eyebrow,h1,.hero-copy>p,.hero-actions'],['.vision','.vision-inner'],['.platform','.section-heading,.feature'],['.interlude','.interlude-copy'],['.approach','.approach>div']];
if(!studio&&'IntersectionObserver'in window){
  if(!motionPref.reduced)document.documentElement.classList.add('reveal-ready');
  const revealer=new IntersectionObserver(entries=>{for(const entry of entries)if(entry.isIntersecting){entry.target.classList.add('is-in');revealer.unobserve(entry.target);}},{rootMargin:'0px 0px -4% 0px',threshold:.08});
  for(const[section,selector]of revealGroups)document.querySelectorAll(section).forEach(root=>root.querySelectorAll(selector).forEach((el,i)=>{el.dataset.reveal='';el.style.setProperty('--reveal-delay',`${Math.min(i,3)*40}ms`);revealer.observe(el);}));
  motionPref.onChange(()=>{if(motionPref.reduced){document.documentElement.classList.remove('reveal-ready');document.querySelectorAll('[data-reveal]').forEach(el=>el.classList.add('is-in'));}});
}
// Keyboard focus and navigation respond immediately; pointer discovery retains the small reveal.
addEventListener('keydown',event=>{if(!event.metaKey&&!event.ctrlKey&&!event.altKey)document.documentElement.classList.add('keyboard-input');});
for(const type of ['pointerdown','wheel'])addEventListener(type,()=>document.documentElement.classList.remove('keyboard-input'),{passive:true});
// Chapter rail: highlights the chapter on screen (index) or switches chapter (studio).
const rail=[...document.querySelectorAll('[data-rail]')];
function updateRail(){const current=scene?.chapter??([...document.querySelectorAll('[data-chapter]')].filter(el=>el.getBoundingClientRect().top<=innerHeight*.55).at(-1)?.dataset.chapter||'monolith');rail.forEach(link=>link.dataset.rail===current?link.setAttribute('aria-current','true'):link.removeAttribute('aria-current'));}
if(!studio&&rail.length){addEventListener('scroll',()=>requestAnimationFrame(updateRail),{passive:true});setTimeout(updateRail,300);}
const studioChapters=[...document.querySelectorAll('[data-studio-chapter]')];
studioChapters.forEach(button=>button.addEventListener('click',()=>{scene?.setChapter(button.dataset.studioChapter);studioChapters.forEach(other=>other.setAttribute('aria-pressed',String(other===button)));const label=document.querySelector('[data-studio-title]');if(label){label.textContent=button.dataset.title;document.querySelector('[data-studio-overline]').textContent=`HYPRL / SCÈNE ${button.dataset.index}`;document.querySelector('[data-studio-text]').textContent=button.dataset.text;}}));
// Exposed for local scene inspection and integration; no account/API connection.
window.hyprlScene=scene;
const motion=document.querySelector('#motion');
function updateMotion(){if(!scene){motion.disabled=true;return;}motion.disabled=false;const optin=motionPref.reduced;motion.classList.toggle('motion-optin',optin);motion.setAttribute('aria-pressed',String(scene.paused));motion.setAttribute('aria-label',optin?'Activer les animations (votre système demande de les réduire)':scene.paused?'Reprendre l’animation':'Mettre l’animation en pause');motion.textContent=optin?'Activer les animations':scene.paused?'▷':'Ⅱ';}
updateMotion();motion.addEventListener('click',()=>{if(!scene)return;if(motionPref.reduced){motionPref.setOverride(true);scene.setPaused(false);}else scene.setPaused(!scene.paused);updateMotion();});document.querySelector('#scene').addEventListener('motionchange',updateMotion);
document.querySelectorAll('[data-palette]').forEach(button=>button.addEventListener('click',()=>{scene?.setPalette(button.dataset.palette);document.querySelectorAll('[data-palette]').forEach(other=>other.setAttribute('aria-pressed',String(other===button)));}));
if(studio){
  const status=document.querySelector('.export-status');
  for(const [id,action,message]of[['export-glb','exportGLB','Géométrie des 4 chapitres exportée. Shaders, lumière et animations restent dans le code.'],['export-png','exportPNG','Image du chapitre exportée sans les textes HTML.']]){
    const button=document.getElementById(id);button.disabled=!scene;button.addEventListener('click',async()=>{button.disabled=true;status.textContent='Préparation…';try{await scene[action]();status.textContent=message;}catch(error){status.textContent='L’export a échoué. Réessayez avec un navigateur compatible WebGL.';console.error(error);}finally{button.disabled=false;}});
  }
}else{
  const dialog=document.querySelector('#workspace'),panel=document.querySelector('#workspace-panel');
  let restoreFocus=null, wasPaused=true;
  const chart=`<svg viewBox="0 0 760 200" preserveAspectRatio="none" role="img" aria-label="Évolution illustrative, aucune performance réelle"><defs><linearGradient id="workspace-fill" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#a9c1df" stop-opacity=".2"/><stop offset="1" stop-color="#a9c1df" stop-opacity="0"/></linearGradient></defs><path d="M0 35H760M0 85H760M0 135H760M0 185H760" stroke="#91a6c4" stroke-opacity=".1"/><path d="M0 158L35 149L65 164L95 120L133 132L166 103L205 111L234 91L270 106L308 77L345 84L380 67L415 82L452 48L490 54L530 34L568 47L607 28L644 35L683 17L723 25L760 8V200H0Z" fill="url(#workspace-fill)"/><path d="M0 158L35 149L65 164L95 120L133 132L166 103L205 111L234 91L270 106L308 77L345 84L380 67L415 82L452 48L490 54L530 34L568 47L607 28L644 35L683 17L723 25L760 8" fill="none" stroke="#abc2dd" stroke-width="2"/></svg>`;
  const overview=`<span class="workspace-badge">APERÇU DE L'INTERFACE</span><h2 id="workspace-title">Votre perspective, aujourd'hui.</h2><p class="lead">Un espace pour relier les marchés, vos stratégies et vos décisions.</p><div class="workspace-metrics"><div class="metric"><small>Univers suivis</small><strong>03</strong><span>Crypto · Forex · Or</span></div><div class="metric"><small>Listes personnelles</small><strong>02</strong><span>Exemple d'organisation</span></div><div class="metric"><small>Mode de l'espace</small><strong>Démo</strong><span>Aucune exécution</span></div></div><div class="workspace-chart"><div class="chart-top">Vue d'une stratégie <small>COURBE ILLUSTRATIVE</small></div>${chart}<div class="chart-axis"><span>JAN</span><span>FÉV</span><span>MAR</span><span>AVR</span><span>MAI</span><span>JUIN</span></div></div>`;
  const markets=`<span class="workspace-badge">VOTRE WATCHLIST</span><h2 id="workspace-title">Les marchés, à votre façon.</h2><p class="lead">Une liste d'exemple. Les cours ne sont pas connectés dans cet aperçu.</p><div class="workspace-chart"><table class="watchlist"><thead><tr><th>ACTIF</th><th>UNIVERS</th><th>STATUT</th></tr></thead><tbody><tr><td>Bitcoin · BTC/USD</td><td>Crypto</td><td>Non connecté</td></tr><tr><td>Ethereum · ETH/USD</td><td>Crypto</td><td>Non connecté</td></tr><tr><td>Euro · EUR/USD</td><td>Forex</td><td>Non connecté</td></tr><tr><td>Or · XAU/USD</td><td>Métaux</td><td>Non connecté</td></tr></tbody></table></div>`;
  let entries=[];try{const saved=JSON.parse(localStorage.getItem('hyprl-eclipse-journal')||'[]');if(Array.isArray(saved))entries=saved.filter(x=>typeof x.text==='string'&&typeof x.date==='string').slice(0,30);}catch{}
  function renderEntries(){const list=document.querySelector('.journal-entries');for(const entry of entries){const article=document.createElement('article'),time=document.createElement('time'),text=document.createElement('span');time.textContent=new Date(entry.date).toLocaleString('fr-FR');time.dateTime=entry.date;text.textContent=entry.text;article.append(time,text);list.append(article);}}
  function setTab(tab){
    const tabs=[...document.querySelectorAll('[data-tab]')];tabs.forEach(button=>{const selected=button.dataset.tab===tab;button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;});panel.setAttribute('aria-labelledby',`tab-${tab}`);
    document.querySelector('.workspace-breadcrumb').textContent=`ESPACE / ${tab==='overview'?"VUE D'ENSEMBLE":tab==='markets'?'MARCHÉS':'JOURNAL'}`;
    if(tab==='overview')panel.innerHTML=overview;
    if(tab==='markets')panel.innerHTML=markets;
    if(tab==='journal'){
      panel.innerHTML=`<span class="workspace-badge">JOURNAL PERSONNEL</span><h2 id="workspace-title">Le fil de vos décisions.</h2><p class="lead">Vos notes restent dans ce navigateur.</p><form class="journal-form"><label for="note">Nouvelle note</label><textarea id="note" maxlength="2000" placeholder="Une observation, une idée, une décision…" required></textarea><button type="submit">Enregistrer la note ↗</button><div class="journal-status" role="status"></div></form><div class="journal-entries" aria-label="Notes enregistrées"></div>`;renderEntries();
      document.querySelector('.journal-form').addEventListener('submit',event=>{event.preventDefault();const input=document.querySelector('#note'),text=input.value.trim();if(!text)return;const next=[{text,date:new Date().toISOString()},...entries].slice(0,30);try{localStorage.setItem('hyprl-eclipse-journal',JSON.stringify(next));entries=next;input.value='';document.querySelector('.journal-status').textContent='Note enregistrée dans ce navigateur.';document.querySelector('.journal-entries').replaceChildren();renderEntries();}catch{document.querySelector('.journal-status').textContent='Stockage indisponible. Votre note reste dans le champ.';}});
    }
  }
  document.querySelectorAll('[data-workspace]').forEach(button=>button.addEventListener('click',()=>{restoreFocus=button;wasPaused=scene?.userPaused??true;setTab('overview');dialog.showModal();document.body.style.overflow='hidden';scene?.setPaused(true);updateMotion();}));
  document.querySelector('.close-workspace').addEventListener('click',()=>dialog.close());
  dialog.addEventListener('close',()=>{document.body.style.overflow='';scene?.setPaused(wasPaused);updateMotion();restoreFocus?.focus();});
  document.querySelectorAll('[data-tab]').forEach(button=>button.addEventListener('click',()=>setTab(button.dataset.tab)));
  document.querySelector('.workspace-tabs').addEventListener('keydown',event=>{const tabs=[...document.querySelectorAll('[data-tab]')];const current=tabs.indexOf(document.activeElement);if(current<0)return;let index=current;if(event.key==='ArrowRight'||event.key==='ArrowDown')index=(current+1)%tabs.length;else if(event.key==='ArrowLeft'||event.key==='ArrowUp')index=(current+tabs.length-1)%tabs.length;else if(event.key==='Home')index=0;else if(event.key==='End')index=tabs.length-1;else return;event.preventDefault();setTab(tabs[index].dataset.tab);tabs[index].focus();});
}
