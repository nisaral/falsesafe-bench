/* FalseSafe-Bench plain-language layer: "What is this?", "What am I looking at?", guided tour,
   hover tooltips on HUD terms, colour legend, glossary. Purely additive; no demo feature is changed.
   Usage: <script src="../common/explain.js" data-page="gym|viewer3d|illustrative|landing|glossary" data-up="../"></script> */
(function(){
const me=document.currentScript, PAGE=(me&&me.dataset.page)||'illustrative', UP=(me&&me.dataset.up)||'../';
const G=window.FSB_GLOSSARY=[
['Hamilton-Jacobi (HJ) reachability','A way to compute, for every starting state, whether a robot can always avoid a crash no matter what. Like working out from every spot on an icy hill whether you can still stop before the cliff.',/\bHJ\b|Hamilton|reachab/i],
['Value function (V)','One number per state. Positive means "you can still stay safe from here", negative means "a crash is unavoidable". V_true uses the real physics; V̂ (V-hat) uses what the robot learned.',/V_true|V̂|V\^|value function|value height|\bV\b/],
['Safe set','All states where the value is ≥ 0, i.e. where staying safe is still possible. Think of it as the "green zone" on a map.',/safe set|safe vol|true safe/i],
['Certifier','The part that says "safe" or "not safe" for a state, based on a learned model of the physics. This benchmark grades certifiers.',/certif/i],
['False-safe (FSV)','The certifier says "safe" but under the real physics a crash is unavoidable. FSV = false-safe volume, the fraction of the state space where that happens. This is the dangerous mistake; lower is better.',/FSV|false-safe|false safe|FALSE-SAFE/i],
['False-unsafe (FUV)','The certifier says "unsafe" where the robot was actually fine. Over-caution: annoying and costs performance, but not dangerous. FUV = false-unsafe volume.',/FUV|false-unsafe|false unsafe/i],
['Hidden physics / coverage gap','A region the robot never saw during training, where the physics are different (for example weak brakes or a drift). Like a patch of black ice that was not there when you practised.',/gap|hidden/i],
['gap-FSV','False-safe volume measured only inside the hidden-physics gap. Our main finding is that most false-safe error lives here.',/gap-FSV/i],
['k (inflation)','How much the certifier pads its uncertainty. k=0 trusts the learned model as-is; k=2 adds two "standard deviations" of caution.',/\bk=|\bk\b|inflation/],
['Ensemble','Train several models and compare them. Where they disagree, the robot is probably unsure. Like asking five friends for directions.',/ensemble/i],
['Conformal prediction','A statistical recipe that turns past errors into an error bar with a stated coverage rate (for example "right 90% of the time"), under the assumption that the future looks like the past.',/conformal/i],
['Distance-aware','A certifier that gets more cautious the further a state is from any training data.',/distance/i],
['Shield / safety filter','A guard that lets you drive normally but overrides your input when the certifier thinks you are about to leave the safe set. It is only as good as the certifier it trusts.',/shield|filter|INTERVENING|override/i],
['World model','A learned simulator of how the world responds to actions. Robots plan "in imagination" inside it.',/world model/i],
['Latent space','A compressed internal representation a learned model uses instead of raw sensor data (for example images).',/latent/i],
['Oracle','A cheating reference certifier that knows the true physics. It shows the best possible result.',/oracle/i],
['Dubins car','A simple car model that drives at constant speed and can only steer. A classic robotics test case.',/dubins/i],
['Double integrator','The simplest moving object: position and velocity, and you control acceleration. A cart on a track.',/double integrator|\bdi_|cart/i],
['IoU','Intersection over union: how much the learned safe set overlaps the true one (100% is a perfect match).',/IoU/],
['Preregistration','Writing down the hypotheses, metrics and pass/fail bar before running the experiment, so the results cannot be cherry-picked.',/prereg/i],
['Confidence interval (95% BCa)','A range that probably contains the true value. BCa is a bootstrap method that corrects for bias and skew. P1: 40.34% [36.47%, 44.73%].',/confidence|BCa|interval/i],
['Rollout / trajectory','One simulated run of the robot from a start state, step by step.',/trajector|rollout/i],
];
const LOOK={
gym:'You are driving a small system (cart, drone, car...) yourself. The big view shows the world; the minimap on the top right is the state space (position vs velocity): green = truly safe, magenta = FALSE-SAFE (the certifier says safe, but a crash is unavoidable), dashed box = hidden-physics gap. With the shield ON, the certifier overrides you when it thinks you are about to become unsafe. Where it is wrong, you can still crash.',
viewer3d:'A 3D plot of two value functions computed by a real HJ solver: the true one and the one the learned certifier believes. Where the certifier surface is above zero but the true one is below, the floor is coloured as false-safe. Use the sliders to change the amount of training data, the gap or k, and watch the false-safe area grow or shrink. The numbers come from exported solver runs, not from the browser.',
illustrative:'This is an older illustrative page: hand-made geometry or toy in-browser maths, for intuition only. It is not solver output. For real numbers use the 3D viewer or the gym.',
landing:'This site collects the FalseSafe-Bench demos.', glossary:'Plain-language explanations of the jargon used across the site.'};
const WHAT='<b>The problem.</b> Robots and AI systems increasingly <i>learn</i> their own physics and their own safety checks. A learned check can confidently say <b>"safe"</b> in places it never saw, where in reality a crash is unavoidable. <b>FalseSafe-Bench measures how often and how badly that happens</b>: it computes the true answer exactly (HJ reachability on a grid) and compares it with what each learned certifier claims.<br><br><b>Honest headline.</b> Adding uncertainty padding (k=2) cut false-safe volume by <b>40.34%</b> [95% CI 36.47–44.73%] on the double integrator, below the preregistered 50% bar, so P1 is only <b>partially supported</b>. It removed 89.1% of the error where the model had data but only 33.3% inside the unseen gap, which holds 87.3% of the error. Source: claims_ledger.md.';
const TOURS={
gym:[[null,'Welcome! This gym lets you drive a robot whose safety guard was learned from data. Take the 30-second tour or skip it.'],
['#scn','Pick a scenario: each has a hazard (wall, cliff, ground...) and a hidden-physics gap the model never saw.'],
['#cert','Pick the certifier, the learned safety check the shield trusts. "oracle" knows the true physics; the others learned it.'],
['#met','The numbers for this certifier, computed offline by the Python solver: FSV = how much of the space it wrongly calls safe.'],
['#shieldBtn','Turn the shield on or off (Space). ON: it overrides your input near danger. OFF: you are on your own.'],
['#map','The state-space map: green = safe, magenta = false-safe (looks safe, is not), dashed box = hidden gap. The dot is you.'],
['#hud','Live readout: both value functions at your current state, and whether you are in the gap or a false-safe region. Hover any term for an explanation.'],
['#fsb-try','Try these experiments. You can replay this tour any time from the "?" button.']],
viewer3d:[[null,'This 3D viewer shows real solver output: the true value function vs the learned one. Quick tour?'],
['#ds','Choose a dataset (an exported experiment).'],['#sliders','Move through the precomputed sweep: training data, gap size, inflation k.'],
['#layers','Toggle surfaces and layers: V_true, V̂, the false-safe floor, the gap box, trajectories.'],
['#met','Metrics for this exact case, from the exported JSON. Hover any term for its meaning.'],
['#c3d','Drag to rotate, scroll to zoom. Magenta on the floor = false-safe region.']],
illustrative:[[null,'This is an illustrative demo (toy maths, not solver output). Use "?" for explanations and the glossary.']],
landing:[[null,'Start with the Interactive gym (drive it yourself) or the 3D viewer (see the real solver output). The glossary explains every term.']]};
const LEGEND='<div><i style="background:#2ea043"></i>truly safe, certifier agrees</div><div><i style="background:#ff2fd0"></i><b>false-safe</b>: says safe, crash unavoidable</div><div><i style="background:#d29922"></i>false-unsafe: says unsafe, actually fine</div><div><i style="background:#3b4048"></i>truly unsafe</div><div><i style="background:#8b1e1e"></i>hazard (wall, cliff, ground)</div><div><i style="border:2px dashed #79c0ff"></i>hidden-physics gap</div><p style="margin:6px 0 0;color:#8b949e">Exact colours vary slightly per demo; the minimap and floor legends on each page are authoritative.</p>';
const TRY=['Pick <b>Cart vs wall</b>, keep the shield ON with the <b>naive</b> certifier (k=0), and drive hard towards the wall through the drift band. Does the shield save you?','Switch the certifier to <b>oracle</b> and repeat. The oracle knows the true physics: no crashes with the shield on.','Turn the shield <b>OFF</b> (Space) and try to stop just short of the hazard yourself.','Find a <b>magenta</b> cell on the minimap and start from it ("Random start" until the log says you start in a false-safe region). Watch V̂ say safe while V_true says no.','Try an inflated certifier (k=2): fewer false-safe cells, but the shield intervenes more often (the false-unsafe cost).','Run <b>Batch</b> with 100 episodes and compare crash counts with the shield ON vs OFF.'];
const css=`#fsb-x-btn{position:fixed;right:12px;bottom:12px;z-index:10001;display:flex;gap:6px}#fsb-x-btn button{background:#1f6feb;color:#fff;border:0;border-radius:16px;padding:6px 12px;font:600 13px system-ui;cursor:pointer;box-shadow:0 2px 8px #0008}#fsb-x-btn button.sec{background:#30363d}
.fsb-x-panel{position:fixed;right:12px;bottom:56px;z-index:10002;width:min(440px,92vw);max-height:70vh;overflow:auto;background:#161b22;color:#e6edf3;border:1px solid #30363d;border-radius:10px;padding:14px 16px;font:14px/1.5 system-ui;box-shadow:0 8px 30px #000a}.fsb-x-panel h3{margin:.2em 0 .4em;font-size:16px}.fsb-x-panel a{color:#58a6ff}.fsb-x-close{float:right;cursor:pointer;color:#8b949e}
.fsb-x-legend div{margin:3px 0}.fsb-x-legend i{display:inline-block;width:14px;height:14px;border-radius:3px;margin-right:8px;vertical-align:-2px}
#fsb-tip{position:fixed;z-index:10005;max-width:320px;background:#0d1117;color:#e6edf3;border:1px solid #58a6ff;border-radius:8px;padding:8px 10px;font:13px/1.45 system-ui;pointer-events:none;display:none;box-shadow:0 4px 16px #000a}#fsb-tip b{color:#79c0ff}
.fsb-term{text-decoration:underline dotted #79c0ff 1px;text-underline-offset:3px;cursor:help}
#fsb-tour-hl{position:fixed;z-index:10003;border:3px solid #f0b72f;border-radius:8px;box-shadow:0 0 0 9999px rgba(0,0,0,.55);pointer-events:none;transition:all .25s}
#fsb-tour-box{position:fixed;z-index:10004;width:min(340px,90vw);background:#161b22;color:#e6edf3;border:1px solid #f0b72f;border-radius:10px;padding:12px 14px;font:14px/1.5 system-ui}#fsb-tour-box button{margin:8px 6px 0 0;background:#238636;color:#fff;border:0;border-radius:6px;padding:5px 12px;cursor:pointer}#fsb-tour-box button.skip{background:#30363d}
#fsb-try{margin:10px 0;padding:10px;border:1px solid #f0b72f55;border-radius:8px;background:#f0b72f12;font-size:13px}#fsb-try h2{margin:0 0 6px}#fsb-try li{margin:4px 0}`;
function el(t,a,h){const e=document.createElement(t);Object.assign(e,a||{});if(h!=null)e.innerHTML=h;return e;}
function panel(id,title,body){let p=document.getElementById(id);if(p){p.remove();return;}document.querySelectorAll('.fsb-x-panel').forEach(x=>x.remove());p=el('div',{id,className:'fsb-x-panel'},`<span class="fsb-x-close" title="close">✕</span><h3>${title}</h3>${body}`);p.querySelector('.fsb-x-close').onclick=()=>p.remove();document.body.appendChild(p);return p;}
function glossHTML(){return '<dl>'+G.map(g=>`<dt><b>${g[0]}</b></dt><dd style="margin:0 0 8px 0">${g[1]}</dd>`).join('')+'</dl>';}
function match(t){if(!t||t.length>70)return null;for(const g of G)if(g[2].test(t))return g;return null;}
const SCOPE='[class*=metric],[class*=label],[class*=stat],[class*=badge],#side,#hud,#mapcap,#met,#trinfo,#layers,#score,#batchres,#banner,.metric,.hud,.panel,.card,table,label,h2';
function mark(){document.querySelectorAll(SCOPE).forEach(r=>r.querySelectorAll('td,th,label,span,b,div,h2,dt').forEach(e=>{if(e.children.length||e.classList.contains('fsb-term')||e.closest('.fsb-x-panel,#fsb-tour-box,#fsb-nav'))return;if(match(e.textContent.trim()))e.classList.add('fsb-term');}));}
let tip;function onOver(ev){const e=ev.target;if(!(e instanceof Element)||e.closest('.fsb-x-panel,#fsb-tour-box'))return;const g=e.children.length===0&&match(e.textContent.trim());if(!g){tip.style.display='none';return;}tip.innerHTML=`<b>${g[0]}</b><br>${g[1]}`;tip.style.display='block';const x=Math.min(ev.clientX+14,innerWidth-340),y=Math.min(ev.clientY+14,innerHeight-tip.offsetHeight-8);tip.style.left=x+'px';tip.style.top=y+'px';}
let step=0,steps=[];function tour(start){steps=(TOURS[PAGE]||TOURS.illustrative).filter(s=>!s[0]||document.querySelector(s[0]));step=start||0;show();}
function endTour(){['fsb-tour-hl','fsb-tour-box'].forEach(i=>{const e=document.getElementById(i);if(e)e.remove();});try{localStorage.setItem('fsb-tour-'+PAGE,'done');}catch(e){}}
function show(){endTour();if(step>=steps.length)return;const [sel,txt]=steps[step];const box=el('div',{id:'fsb-tour-box'},`<div style="color:#f0b72f;font-size:12px">Tour ${step+1}/${steps.length}</div>${txt}<br><button class="next">${step<steps.length-1?'Next':'Done'}</button><button class="skip">Skip tour</button>`);
const t=sel&&document.querySelector(sel);if(t){const r=t.getBoundingClientRect();const hl=el('div',{id:'fsb-tour-hl'});Object.assign(hl.style,{left:r.left-4+'px',top:r.top-4+'px',width:r.width+8+'px',height:Math.min(r.height,innerHeight)+8+'px'});document.body.appendChild(hl);const bx=r.right+360<innerWidth?r.right+12:Math.max(8,r.left-352);box.style.left=(bx<0||r.width>innerWidth*0.6?Math.max(8,innerWidth/2-170):bx)+'px';box.style.top=Math.max(8,Math.min(r.top,innerHeight-220))+'px';}else{box.style.left=Math.max(8,innerWidth/2-170)+'px';box.style.top='30%';}
document.body.appendChild(box);box.querySelector('.next').onclick=()=>{step++;step<steps.length?show():endTour();};box.querySelector('.skip').onclick=endTour;}
function init(){document.head.appendChild(el('style',{},css));tip=el('div',{id:'fsb-tip'});document.body.appendChild(tip);document.addEventListener('mouseover',onOver);
if(PAGE==='glossary'){const m=document.getElementById('gloss');if(m)m.innerHTML=glossHTML();}
const bar=el('div',{id:'fsb-x-btn'});const B=(l,f,c)=>{const b=el('button',{className:c||''},l);b.onclick=f;bar.appendChild(b);};
B('? What is this',()=>panel('fsb-what','What is this?',WHAT+`<h3 style="margin-top:12px">What am I looking at?</h3>${LOOK[PAGE]||LOOK.illustrative}<p><a href="${UP}glossary.html">Full glossary</a></p>`));
if(PAGE!=='glossary')B('Tour',()=>tour(0),'sec');B('Legend',()=>panel('fsb-legend','Colour legend','<div class="fsb-x-legend">'+LEGEND+'</div>'),'sec');B('Glossary',()=>panel('fsb-gloss','Glossary (plain language)',glossHTML()+`<p><a href="${UP}glossary.html">Open as a page</a></p>`),'sec');document.body.appendChild(bar);
if(PAGE==='gym'){const side=document.getElementById('side'),anchor0=document.getElementById('prov'),anchor=anchor0&&anchor0.previousElementSibling&&anchor0.previousElementSibling.tagName==='H2'?anchor0.previousElementSibling:anchor0;if(side){const t=el('div',{id:'fsb-try'},'<h2>Try this</h2><ol>'+TRY.map(x=>`<li>${x}</li>`).join('')+'</ol>');anchor&&anchor.parentNode===side?side.insertBefore(t,anchor):side.appendChild(t);}}
mark();setInterval(mark,1500);let seen=null;try{seen=localStorage.getItem('fsb-tour-'+PAGE);}catch(e){}
const auto=!/[?&](notour|nohelp)=1/.test(location.search)&&!seen&&PAGE!=='glossary';const go=()=>{const h=document.getElementById('help');if(h&&h.offsetParent!==null&&getComputedStyle(h).display!=='none')return setTimeout(go,700);tour(0);};if(auto)setTimeout(go,1200);
window.__fsbExplain={tour,endTour,glossary:G,page:PAGE};}
document.readyState==='loading'?document.addEventListener('DOMContentLoaded',init):init();
})();
