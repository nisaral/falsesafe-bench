/* FalseSafe "Learned from pixels (world model)" panel, shared by viewer3d and the gym.
   Loads OPTIONAL exported data (scripts/export_worldmodel_demo.py -> common/worldmodel/*.js,
   schema falsesafe.worldmodel_demo/v1, docs/WORLDMODEL_DEMO.md).  With no data it shows an empty state.
   Everything shown here is EXPLORATORY, from dev seeds, and the preregistered gates were NOT passed.
   The decode/classify part is pure JS and is also run in node by tests/test_worldmodel_demo.py. */
(function(root){
'use strict';
function b64(s){if(typeof Buffer!=='undefined'&&typeof atob==='undefined'){const b=Buffer.from(s,'base64');return new Uint8Array(b.buffer,b.byteOffset,b.length);}
  const t=atob(s),u=new Uint8Array(t.length);for(let i=0;i<t.length;i++)u[i]=t.charCodeAt(i);return u;}
function grid(g){if(!g)return null;const u=b64(g.b64);let a;if(g.dtype==='int8')a=new Int8Array(u.buffer,u.byteOffset,u.length);
  else if(g.dtype==='uint16'){a=new Uint16Array(u.length/2);for(let i=0;i<a.length;i++)a[i]=u[2*i]|(u[2*i+1]<<8);}else a=u;
  const s=g.scale||1,f=new Float32Array(a.length);for(let i=0;i<a.length;i++)f[i]=a[i]*s;return f;}
// 0 truly safe, 1 FALSE-SAFE (vhat>=0 but vstar<0), 2 false-unsafe, 3 truly unsafe  (same as the Python exporter)
function classify(vs,vh){const c=new Uint8Array(vs.length);for(let i=0;i<vs.length;i++){const s=vs[i]>=0,h=vh[i]>=0;c[i]=s&&h?0:!s&&h?1:s?2:3;}return c;}
function counts(c){const n=[0,0,0,0];for(const k of c)n[k]++;return {TS:n[0],FS:n[1],FU:n[2],TU:n[3],n:c.length,FSV:n[1]/c.length,FUV:n[2]/c.length};}
function decode(d){const g=d.grids||{},o={vstar:grid(g.vstar),v_latent:grid(g.v_latent),v_enc:grid(g.v_enc),inhull:grid(g.inhull)};
  if(o.vstar&&o.v_latent){o.cls=classify(o.vstar,o.v_latent);o.counts=counts(o.cls);}
  if(o.vstar&&o.v_enc){o.cls_enc=classify(o.vstar,o.v_enc);o.counts_enc=counts(o.cls_enc);}
  const L=d.latent||{};if(L.points){const q=grid({b64:L.points.b64,dtype:'uint16'}),P=new Float32Array(q.length);
    for(let i=0;i<q.length;i+=2){P[i]=L.lo[0]+(L.hi[0]-L.lo[0])*q[i]/65535;P[i+1]=L.lo[1]+(L.hi[1]-L.lo[1])*q[i+1]/65535;}o.zpts=P;o.zcls=b64(L.points.class_b64);}
  if(L.V0)o.V0=grid(L.V0);return o;}
function median(a){const b=a.filter(x=>x!=null).sort((x,y)=>x-y),n=b.length;return n?(n%2?b[(n-1)/2]:(b[n/2-1]+b[n/2])/2):null;}
const API={b64,grid,classify,counts,decode,median};
if(typeof module!=='undefined'&&module.exports){module.exports=API;return;}
if(typeof document==='undefined'){root.FSBWorldModelCore=API;return;}

// ------------------------------------------------------------------------------------------ UI
const COL=[[0x2e,0xa0,0x43],[0xff,0x2b,0xd6],[0xd2,0x99,0x22],[0x30,0x36,0x3d]];
const NAMES=['truly safe','FALSE-SAFE','false-unsafe','truly unsafe'];
const pct=x=>x==null?'–':(100*x).toFixed(x<0.01?2:1)+'%';
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const CSS=`#fsb-wm{position:fixed;inset:0;z-index:10001;background:#0d1117f2;color:#e6edf3;display:none;overflow:auto;font:13.5px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
#fsb-wm.show{display:block}#fsb-wm .wrap{max-width:1100px;margin:0 auto;padding:16px}
#fsb-wm h2{margin:0;font-size:19px;text-transform:none;letter-spacing:0;color:#e6edf3}#fsb-wm h3{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#8b949e;margin:14px 0 6px}
#fsb-wm .badge{position:sticky;top:4px;z-index:2;box-shadow:0 2px 8px #000a;display:inline-block;background:#3d2a00;color:#ffd27a;border:1px solid #9e6a03;border-radius:999px;padding:2px 10px;font-size:12px;font-weight:600;margin:6px 0}
#fsb-wm .top{display:flex;gap:10px;align-items:center;flex-wrap:wrap}#fsb-wm .top .sp{flex:1}
#fsb-wm select,#fsb-wm button{background:#21262d;color:#e6edf3;border:1px solid #30363d;border-radius:6px;padding:6px 10px;font:inherit;min-height:34px}
#fsb-wm button.on{background:#1f6feb;border-color:#1f6feb}#fsb-wm .grid{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:18px}
@media (max-width:820px){#fsb-wm .grid{grid-template-columns:1fr}}#fsb-wm #fsb-wm-map{width:min(100%,62vh,520px)}
#fsb-wm label{display:inline-flex;gap:6px;align-items:center;margin:0;color:#c9d1d9}#fsb-wm input[type=checkbox]{width:auto;margin:0}#fsb-wm .try{border-left:3px solid #58a6ff;background:#161b22;padding:6px 10px;margin:8px 0;border-radius:4px}
#fsb-wm canvas{width:100%;max-width:520px;aspect-ratio:1;image-rendering:pixelated;border:1px solid #30363d;border-radius:6px;display:block;touch-action:none}
#fsb-wm .leg span{display:inline-block;width:11px;height:11px;border-radius:2px;margin:0 4px -1px 10px}#fsb-wm .leg span:first-child{margin-left:0}
#fsb-wm .src{color:#8b949e;font-size:12px}#fsb-wm table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}#fsb-wm td,#fsb-wm th{padding:3px 6px;border-bottom:1px solid #21262d;text-align:left}
#fsb-wm td.n{text-align:right}#fsb-wm .bar{display:flex;height:22px;border-radius:5px;overflow:hidden;background:#21262d;margin:4px 0}
#fsb-wm .bar div{height:100%}#fsb-wm .row{display:grid;grid-template-columns:110px 1fr 64px;gap:8px;align-items:center;margin:3px 0}
#fsb-wm .row .b{height:14px;background:#21262d;border-radius:4px;overflow:hidden}#fsb-wm .row .b div{height:100%}
#fsb-wm .empty{border:1px dashed #30363d;border-radius:10px;padding:22px;margin-top:14px}#fsb-wm code{background:#161b22;padding:1px 5px;border-radius:4px}
#fsb-wm .imgs{display:flex;gap:4px;flex-wrap:wrap}#fsb-wm .imgs canvas{width:52px;height:52px;border-radius:3px}
#fsb-wm .seeds{display:flex;flex-wrap:wrap;gap:4px}#fsb-wm .seeds button{min-width:44px;padding:4px 6px}#fsb-wm .seeds button.bad{border-color:#f85149;color:#ffa198}
#fsb-wm .seeds button.bad.on{color:#fff}#fsb-wm .strip svg{width:100%;max-width:560px;height:auto;display:block}#fsb-wm .tune{display:inline-block;background:#30215a;color:#d2a8ff;border:1px solid #6e40c9;border-radius:999px;padding:1px 9px;font-size:12px;font-weight:600}
#fsb-wm .col{border-left:3px solid #f85149;background:#25171a;padding:6px 10px;margin:8px 0;border-radius:4px}#fsb-wm .no{color:#ffa198;font-weight:600}#fsb-wm .yes{color:#7ee787}#fsb-wm .arm{border-left:3px solid #a371f7;background:#1c1530;padding:6px 10px;margin:8px 0;border-radius:4px}
#fsb-wm .rd{min-height:20px;font-variant-numeric:tabular-nums}#fsb-wm .warn{border-left:3px solid #9e6a03;padding:6px 10px;background:#161b22;margin:10px 0}`;
let host=null,st={base:'',files:null,meta:null,cur:null,dec:null,mode:'latent',hull:true},onClose=null;
function el(h){const t=document.createElement('div');t.innerHTML=h.trim();return t.firstChild;}
function loadScript(src){return new Promise((ok,bad)=>{const s=document.createElement('script');s.src=src;s.onload=ok;s.onerror=bad;document.head.appendChild(s);});}
async function loadIndex(){if(st.files)return st.files;try{if(!window.FSB_WM_FILES)await loadScript(st.base+'worldmodel/index.js');}catch(e){}
  st.files=Array.isArray(window.FSB_WM_FILES)?window.FSB_WM_FILES:[];st.meta=window.FSB_WM_META||{};return st.files;}
const ARMNAME={main:'Camera-only encoder (main)',sup:'Different encoder: trained with the true state'};
const fsvOf=f=>f&&f.FSV;
function seedNote(f){const m=[];if(f.collapsed)m.push('collapsed: certifies (almost) everything as safe');if(f.folded)m.push('latent map folds over itself (fold '+(100*f.fold).toFixed(0)+'%)');if(f.tuning_seed&&f.condition==='QA')m.push('tuning seed — not representative');return m;}
function strip(rows,cur){const W=560,H=86,X=v=>14+(W-28)*Math.sqrt(Math.max(0,Math.min(v,0.32))/0.32),med=median(rows.map(fsvOf));
  const ticks=[0,0.01,0.03,0.1,0.2,0.3].map(t=>`<line x1="${X(t)}" x2="${X(t)}" y1="18" y2="62" stroke="${t===0.03?'#2ea043':'#30363d'}" ${t===0.03?'stroke-dasharray="3 2"':''}/><text x="${X(t)}" y="74" fill="#8b949e" font-size="10" text-anchor="middle">${(100*t).toFixed(0)}%</text>`).join('');
  const dots=rows.map((f,i)=>{const bad=f.collapsed,y=26+(i%3)*14,c=bad?'#f85149':f.folded?'#d29922':'#58a6ff',on=cur&&f.file===cur;
    return `<g data-seed="${f.seed}" style="cursor:pointer"><title>seed ${f.seed}: false-safe ${pct(f.FSV)}${seedNote(f).length?' — '+seedNote(f).join('; '):''}</title><circle cx="${X(f.FSV||0)}" cy="${y}" r="${on?7:5}" fill="${c}" stroke="${on?'#fff':'#0d1117'}" stroke-width="${on?2:1}"/><text x="${X(f.FSV||0)}" y="${y+3}" font-size="7" text-anchor="middle" fill="#0d1117">${f.seed}</text></g>`;}).join('');
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="False-safe volume of the full world model for each seed; median ${pct(med)}">${ticks}<text x="${X(0.03)+3}" y="12" fill="#7ee787" font-size="10">G1 limit 3%</text>${med!=null?`<line x1="${X(med)}" x2="${X(med)}" y1="18" y2="62" stroke="#e6edf3" stroke-width="2"/><text x="${X(med)}" y="85" fill="#e6edf3" font-size="10" text-anchor="middle">median ${pct(med)}</text>`:''}${dots}</svg>`;}
function gateTable(){const G=(st.meta||{}).doc_gates||{},D=(st.meta||{}).gate_defs||{},yn=b=>b?'<span class="yes">pass</span>':'<span class="no">not passed</span>';
  const arms=Object.keys(G);if(!arms.length)return '';
  const r=(k,lab,mean)=>`<tr><td title="${esc(D[k]||'')}">${lab}</td>${arms.map(a=>{const g=G[a],p=(g.per_seed||{})[k],m=(g.seed_mean||{})[k];
    return `<td class="n">${p?p.pass+'/'+p.n:'–'}</td><td class="n">${m?mean(m)+' · '+yn(m.pass):'–'}</td>`;}).join('')}</tr>`;
  return `<h3>Preregistered gates (condition QA) — none passed</h3><table class="gates"><tr><th>gate</th>${arms.map(a=>`<th colspan="2">${esc(ARMNAME[a]||a)}<br><span class="src">seeds passing · on the seed mean</span></th>`).join('')}</tr>
   ${r('G1','G1 few wrong answers (false-safe ≤3% and false-unsafe ≤3%)',m=>'FSV '+pct(m.FSV)+', FUV '+pct(m.FUV))}
   ${r('G4','G4 latent keeps state order (ρ≥0.9) and encoder adds ≤1% false-safe',m=>'ρ '+m.spearman.toFixed(2)+', FSV_enc '+pct(m.FSV_enc))}
   ${r('C2.5','C2.5 latent map does not fold (fold ≤5%)',m=>'fold '+pct(m.fold_frac))}</table>
   <p class="src">Exact numbers from <code>worldmodel-dev-results.md</code> (aggregate_latent_dev.py). The two encoders are never pooled. Only the encoder part of G4 (FSV_enc ≤1%): ${arms.map(a=>{const p=(G[a].per_seed||{}).G4_FSV_enc_part;return esc(a)+' '+(p?p.pass+'/'+p.n:'–');}).join(', ')} seeds.</p>`;}
function controls(d){const F=st.files,arm=d?d.arm&&d.arm.arm||'main':'main',cur=d&&F.find(f=>f.file===d.id)||F[0];
  const arms=[...new Set(F.map(f=>f.arm))],conds=[...new Set(F.filter(f=>f.arm===cur.arm).map(f=>f.condition))];
  const rows=F.filter(f=>f.arm===cur.arm&&f.condition===cur.condition).sort((a,b)=>a.seed-b.seed);
  const M=st.meta||{},ex=(M.excluded||[]).map(e=>`<b>${esc(e.tag)}</b> not shown: ${esc(e.reason)}.`).join(' ');
  const notes=seedNote(cur);
  return `<div class="top" style="margin-top:10px" role="group" aria-label="Which encoder"><span class="src">Encoder:</span>${arms.map(a=>`<button data-arm="${a}" class="${a===cur.arm?'on':''}" aria-pressed="${a===cur.arm}">${esc(ARMNAME[a]||a)}</button>`).join('')}
    <label>Condition <select data-a="cond" aria-label="Condition">${conds.map(c=>{const f=F.find(x=>x.arm===cur.arm&&x.condition===c);return `<option value="${c}"${c===cur.condition?' selected':''}>${c} · ${esc(f.condition_text||'')}</option>`;}).join('')}</select></label></div>
   ${cur.arm==='sup'?`<div class="arm src"><b>Different encoder.</b> ${esc(((M.arms||{}).sup||{}).note||'trained with the true state')}. Seeds 0–4 only. Do not compare it as if it were the same model.</div>`:''}
   <p style="margin:8px 0 2px"><b>The camera-based model works on some random seeds and fails on others.</b> <span class="src">Each seed is the same recipe trained from a different random start.</span></p>
   <div class="top"><span class="src">Seed:</span><div class="seeds" role="group" aria-label="Pick a seed">${rows.map(f=>`<button data-seed="${f.seed}" data-file="${esc(f.file)}" class="${f.file===cur.file?'on':''}${f.collapsed?' bad':''}" aria-pressed="${f.file===cur.file}" title="${esc('false-safe '+pct(f.FSV)+(seedNote(f).length?' — '+seedNote(f).join('; '):''))}">${f.seed}${f.collapsed?' ✕':''}${f.tuning_seed&&f.condition==='QA'?'*':''}</button>`).join('')}</div></div>
   <div class="strip">${strip(rows,cur.file)}<div class="src leg"><span style="background:#58a6ff;border-radius:50%"></span>seed <span style="background:#d29922;border-radius:50%"></span>map folds over (fold &gt;90%) <span style="background:#f85149;border-radius:50%"></span>collapsed (certifies almost everything) · white line = median · axis is square-root scaled · ✕ = collapsed${cur.condition==='QA'?' · * = tuning seed':''}</div></div>
   <p style="margin:6px 0"><b>${esc(cur.condition)} seed ${cur.seed}</b>${cur.tuning_seed&&cur.condition==='QA'?' <span class="tune" title="The encoder settings were tuned on QA seed 0 only, so this seed looks better than a typical one.">tuning seed — not representative</span>':''} · false-safe ${pct(cur.FSV)} · fold ${pct(cur.fold)} · ρ ${cur.spearman!=null?cur.spearman.toFixed(2):'–'}</p>
   ${cur.collapsed?`<div class="col"><b>This seed collapsed.</b> The encoder degenerated, and the model now calls (almost) every state safe (it certifies ${pct(cur.cert_frac)} of states), so it misses about every real danger. This is a failure of the method, not a display bug.</div>`:cur.folded?`<div class="col" style="border-color:#d29922;background:#221c10"><b>Folded latent map.</b> The encoder maps distant real states onto the same latent points (fold ${pct(cur.fold)}), so its safety answers there are unreliable.</div>`:''}
   ${ex?`<p class="src">${ex}</p>`:''}`;}
async function loadData(id){window.FSB_WM=window.FSB_WM||[];let d=window.FSB_WM.find(x=>x.id===id);if(d)return d;
  try{await loadScript(st.base+'worldmodel/'+id+'.js');}catch(e){return null;}return window.FSB_WM.find(x=>x.id===id)||null;}
function ensure(){if(host)return host;const s=document.createElement('style');s.textContent=CSS;document.head.appendChild(s);
  host=el(`<div id="fsb-wm" role="dialog" aria-modal="true" aria-labelledby="fsb-wm-h"><div class="wrap"></div></div>`);document.body.appendChild(host);
  host.addEventListener('keydown',e=>{if(e.key==='Escape'){e.stopPropagation();close();}});host.addEventListener('click',e=>{if(e.target===host)close();});return host;}
function header(){return `<div class="top"><h2 id="fsb-wm-h">🧠 Learned from pixels (world model)</h2><span class="sp"></span><button data-a="close" aria-label="Close world-model panel">Close (Esc)</button></div>
  <div class="badge" title="These numbers come from development seeds of an exploratory experiment whose preregistered success gates were not passed.">⚠ exploratory · dev seeds · gates not passed</div>
  <p style="margin:4px 0 0">Instead of reading the true state, a <b class="fsb-term" data-term="world model">world model</b> learns from <b>32×32 images</b>: an <b class="fsb-term" data-term="encoder">encoder</b> squeezes each image into a 2-number <b class="fsb-term" data-term="latent">latent</b> state, a learned model predicts how the latent moves, and the safety value is computed <i>in that latent space</i>. Here we pull that learned safety value back onto the real states and compare it with the exact answer. This does <b>not</b> show that the world-model approach works; it shows where and why it goes wrong.</p>`;}
function emptyState(){return header()+`<div class="empty"><h3 style="margin-top:0">No world-model data in this build</h3>
  <p>This panel is optional. It appears once someone exports results from the latent world-model experiment:</p>
  <p><code>python scripts/export_worldmodel_demo.py</code> (default tags: sweep_main, sweep_sup)</p>
  <p class="src">That writes <code>demos/common/worldmodel/*.js</code> (schema <code>falsesafe.worldmodel_demo/v1</code>, see docs/WORLDMODEL_DEMO.md). Everything else in this demo works without it.</p></div>`;}
function render(){const h=ensure().firstChild;if(!st.files.length){h.innerHTML=emptyState();wire();return;}
  const d=st.cur,D=st.dec;
  if(!d){h.innerHTML=header()+controls(null)+`<p class="src">Could not load this dataset file.</p>`;wire();return;}
  const M=d.metrics||{},B=d.breakdown||{},F=M.full||{},dc=D&&D.counts;
  const tot=B.total||F.FSV||0,parts=[['encoder','#ff7b72','Encoder (pixels → latent)'],['dynamics','#d2a8ff','Latent dynamics model'],['margin','#79c0ff','Learned safety margin']];
  const mx=Math.max(1e-9,...parts.map(p=>Math.abs(B[p[0]]||0)));
  const cl=d.closed_loop&&d.closed_loop.arms||null;
  h.innerHTML=header()+controls(d)+`
  <div class="grid"><div><div class="try src"><b>Try:</b> look for the <b style="color:#ff2bd6">pink</b> traps, then step through the seeds: same recipe, very different maps. Switch condition to QB to see the hidden band kept out of training. Press <i>Encoder only</i> to see how much of the error the image-squeezing step causes on its own.</div><h3>Exact vs learned safe set, over the real states</h3>
   <div class="top" role="group" aria-label="Which learned value"><button data-m="latent" class="${st.mode==='latent'?'on':''}" aria-pressed="${st.mode==='latent'}"title="Encoder, dynamics and safety margin all learned">Full world model</button>
    <button data-m="enc" class="${st.mode==='enc'?'on':''}" aria-pressed="${st.mode==='enc'}" ${D&&D.v_enc?'':'disabled'}title="Only the image encoder is learned; dynamics and margin are exact">Encoder only</button>
    <button data-m="exact" class="${st.mode==='exact'?'on':''}" aria-pressed="${st.mode==='exact'}"title="The exact answer, for reference">Exact answer</button>
    <label style="font-size:12px"><input type="checkbox" data-a="hull" ${st.hull?'checked':''}> shade states unlike the training images</label></div>
   <canvas id="fsb-wm-map" width="${d.state?d.state.n[0]:10}" height="${d.state?d.state.n[1]:10}" tabindex="0" aria-label="Map of position (horizontal) and velocity (vertical) coloured by exact versus learned safety"></canvas>
   <div class="src">→ position p (${d.state?d.state.lo[0]:""} … ${d.state?d.state.hi[0]:""}) · ↑ velocity v (${d.state?d.state.lo[1]:""} … ${d.state?d.state.hi[1]:""}) · dashed blue = hidden-physics band</div><div class="rd src" id="fsb-wm-rd">Hover or tap the map to read values.</div>
   <div class="src leg"><span style="background:#2ea043"></span>both say safe<span style="background:#ff2bd6"></span><b>FALSE-SAFE</b>: learned says safe, exact says a crash is unavoidable<span style="background:#d29922"></span>false-unsafe (over-cautious)<span style="background:#30363d"></span>both say unsafe · <span style="background:none;border:1px dashed #58a6ff"></span>hidden-physics band</div>
   ${dc?`<p class="src">Shown: ${esc(d.display_counts?d.display_counts.grid:'')} · false-safe ${pct(dc.FSV)} of states here (full-grid Python value: <b>${pct(F.FSV)}</b>).</p>`:'<p class="src">This dataset has no grids (metrics only).</p>'}
   </div><div>
   <h3>Where does the false-safe come from?</h3>
   <div class="bar" role="img" aria-label="False-safe volume split into encoder, dynamics and margin parts">${parts.map(p=>`<div style="width:${100*Math.max(0,B[p[0]]||0)/Math.max(tot,1e-9)}%;background:${p[1]}" title="${p[2]}"></div>`).join('')}</div>
   ${parts.map(p=>`<div class="row"><span>${p[2].split(' (')[0]}</span><div class="b"><div style="width:${100*Math.abs(B[p[0]]||0)/mx}%;background:${p[1]}"></div></div><span class="n">${pct(B[p[0]])}</span></div>`).join('')}
   <p class="src">Read top to bottom: false-safe volume when only the encoder is learned (${pct(B.encoder)}), then what learning the latent dynamics adds (${pct(B.dynamics)}), then the learned margin head (${pct(B.margin)}); total ${pct(tot)} of the state box. Inside the hidden-physics band: encoder ${pct((B.gap||{}).encoder)}, dynamics ${pct((B.gap||{}).dynamics)}, margin ${pct((B.gap||{}).margin)}.</p>
   <table><tr><th></th><th>false-safe</th><th>in band</th><th>outside</th><th>false-unsafe</th></tr>
    ${[['full model',M.full],['encoder only',M.enc],['+ dynamics',M.dyn]].filter(r=>r[1]).map(r=>`<tr><td>${r[0]}</td><td class="n">${pct(r[1].FSV)}</td><td class="n">${pct(r[1].gapFSV)}</td><td class="n">${pct(r[1].outFSV)}</td><td class="n">${pct(r[1].FUV)}</td></tr>`).join('')}</table>
   <h3>Latent space (what the model "sees")</h3><canvas id="fsb-wm-lat" width="200" height="200" style="max-width:260px" aria-label="Real states placed in the 2-D latent space, coloured by class"></canvas>
   <p class="src">Each dot is a real state placed where the encoder puts it; background = sign of the latent safety value. Magenta dots sit in the learned-safe region but are truly unsafe.</p>
   ${cl?`<h3>Does it crash? (simulated safety guard, seed ${esc(d.closed_loop.seed)})</h3><p class="src">A guard that trusts each value drives many simulated runs. Lower crash rates are better, and the exact row is the best possible.</p><table><tr><th>guard trusts</th><th>crash rate</th><th>crash rate if started in a trap</th><th>take-overs per run</th></tr>
    ${Object.entries(cl).map(([k,v])=>`<tr><td>${esc(k.replace('V_star(state)','exact V* (oracle)').replace('V_full','full world model').replace('V_enc','encoder only'))}</td><td class="n">${pct(v.crash_rate)}</td><td class="n">${pct(v.crash_rate_false_safe_starts)}</td><td class="n">${v.mean_interventions!=null?v.mean_interventions.toFixed(2):'–'}</td></tr>`).join('')}</table>`:''}
   ${d.samples?`<h3>Example input images</h3><div class="imgs" id="fsb-wm-imgs"></div>`:''}
  </div></div>${gateTable()}
  <div class="warn src"><b>Honesty:</b> exploratory experiment on development seeds; the preregistered gates were <b>not</b> passed (see the table above), so treat every number here as a diagnostic, not a result. Nothing here shows that the world model works. Source: <code>${esc(d.provenance&&d.provenance.source||'')}</code>${d.provenance&&d.provenance.source_commit?' @ '+esc(d.provenance.source_commit):''}. Grids are 8-bit quantised with the sign kept exact.</div>`;
  wire();draw();}
function wire(){host.querySelectorAll('[data-a=close]').forEach(b=>b.onclick=close);
  const F=st.files||[],cur=st.cur&&F.find(f=>f.file===st.cur.id)||F[0];
  const pick=(arm,cond,seed)=>{const c=F.filter(f=>f.arm===arm&&(cond==null||f.condition===cond));const f=c.find(x=>x.seed===seed)||c.find(x=>x.condition===(cond||cur.condition)&&x.seed===seed)||c.find(x=>x.condition===(cond||cur.condition))||c[0];if(f)select(f.file);};
  host.querySelectorAll('[data-arm]').forEach(b=>b.onclick=()=>pick(b.dataset.arm,F.some(f=>f.arm===b.dataset.arm&&f.condition===cur.condition)?cur.condition:null,cur.seed));
  const cs=host.querySelector('[data-a=cond]');if(cs)cs.onchange=()=>pick(cur.arm,cs.value,cur.seed);
  host.querySelectorAll('.seeds [data-file]').forEach(b=>b.onclick=()=>select(b.dataset.file));
  host.querySelectorAll('.strip g[data-seed]').forEach(g=>g.onclick=()=>pick(cur.arm,cur.condition,+g.dataset.seed));
  host.querySelectorAll('[data-m]').forEach(b=>b.onclick=()=>{st.mode=b.dataset.m;render();});
  const hb=host.querySelector('[data-a=hull]');if(hb)hb.onchange=()=>{st.hull=hb.checked;draw();};
  const c=host.querySelector('#fsb-wm-map');if(c){const rd=e=>{const r=c.getBoundingClientRect(),p=e.touches?e.touches[0]:e;readout((p.clientX-r.left)/r.width,(p.clientY-r.top)/r.height);};
    c.addEventListener('pointermove',rd);c.addEventListener('pointerdown',rd);}
  const cb=host.querySelector('[data-a=close]');if(cb)setTimeout(()=>cb.focus(),0);}
function draw(){const d=st.cur,D=st.dec,c=host.querySelector('#fsb-wm-map');if(!c||!D||!D.vstar)return;const [n0,n1]=d.state.n,cx=c.getContext('2d'),img=cx.createImageData(n0,n1);
  const vh=st.mode==='enc'?D.v_enc:st.mode==='exact'?D.vstar:D.v_latent,cls=classify(D.vstar,vh);
  for(let i=0;i<n0;i++)for(let j=0;j<n1;j++){const k=i*n1+j,px=(n1-1-j)*n0+i,o=4*px,col=COL[cls[k]];let f=1;
    if(st.hull&&D.inhull&&!D.inhull[k])f=((i+j)&1)?0.45:0.7;img.data[o]=col[0]*f;img.data[o+1]=col[1]*f;img.data[o+2]=col[2]*f;img.data[o+3]=255;}
  cx.putImageData(img,0,0);const g=d.gap;if(g){const lo=d.state.lo[0],hi=d.state.hi[0],X=p=>(p-lo)/(hi-lo)*(n0-1)+0.5;cx.strokeStyle='#58a6ff';cx.lineWidth=0.6;cx.setLineDash([2,2]);
    cx.strokeRect(X(g.lo),0.3,X(g.hi)-X(g.lo),n1-0.6);cx.setLineDash([]);}
  const l=host.querySelector('#fsb-wm-lat');if(l&&D.zpts){const lx=l.getContext('2d'),W=l.width,L=d.latent;lx.fillStyle='#0d1117';lx.fillRect(0,0,W,W);
    if(D.V0){const [a,b]=L.V0.n,im=lx.createImageData(a,b);for(let i=0;i<a;i++)for(let j=0;j<b;j++){const o=4*((b-1-j)*a+i),s=D.V0[i*b+j]>=0;im.data[o]=s?22:40;im.data[o+1]=s?60:24;im.data[o+2]=s?34:30;im.data[o+3]=255;}
      const t=document.createElement('canvas');t.width=a;t.height=b;t.getContext('2d').putImageData(im,0,0);lx.imageSmoothingEnabled=false;lx.drawImage(t,0,0,W,W);}
    for(let k=0;k<D.zcls.length;k++){const x=(D.zpts[2*k]-L.lo[0])/(L.hi[0]-L.lo[0])*W,y=W-(D.zpts[2*k+1]-L.lo[1])/(L.hi[1]-L.lo[1])*W,c=COL[D.zcls[k]];
      lx.fillStyle=`rgb(${c[0]},${c[1]},${c[2]})`;lx.fillRect(x-1,y-1,D.zcls[k]===1?3:2,D.zcls[k]===1?3:2);}}
  const ic=host.querySelector('#fsb-wm-imgs');if(ic&&d.samples&&!ic.children.length){const S=d.samples,u=b64(S.b64);for(let k=0;k<S.n;k++){const cv=document.createElement('canvas');cv.width=S.w;cv.height=S.h;
    const x=cv.getContext('2d'),im=x.createImageData(S.w,S.h);for(let p=0;p<S.w*S.h;p++){const v=u[k*S.w*S.h+p];im.data[4*p]=im.data[4*p+1]=im.data[4*p+2]=v;im.data[4*p+3]=255;}
    x.putImageData(im,0,0);const st_=S.states[k]||[];cv.title=`p=${(+st_[0]).toFixed(2)}, v=${(+st_[1]).toFixed(2)}`;cv.setAttribute('aria-label','input image at '+cv.title);ic.appendChild(cv);}}}
function readout(fx,fy){const d=st.cur,D=st.dec,o=host.querySelector('#fsb-wm-rd');if(!D||!D.vstar||!o)return;const [n0,n1]=d.state.n;
  const i=Math.max(0,Math.min(n0-1,Math.floor(fx*n0))),j=Math.max(0,Math.min(n1-1,n1-1-Math.floor(fy*n1))),k=i*n1+j;
  const p=d.state.lo[0]+(d.state.hi[0]-d.state.lo[0])*i/(n0-1),v=d.state.lo[1]+(d.state.hi[1]-d.state.lo[1])*j/(n1-1);
  const vh=st.mode==='enc'?D.v_enc:D.v_latent,c=classify([D.vstar[k]],[vh[k]])[0];
  o.innerHTML=`p=${p.toFixed(2)}, v=${v.toFixed(2)} · exact safety ${D.vstar[k].toFixed(2)} · learned ${vh[k].toFixed(2)} (above 0 = safe) → <b style="color:rgb(${COL[c]})">${NAMES[c]}</b>${D.inhull&&!D.inhull[k]?' · unlike training images':''}`;}
async function select(id){st.cur=await loadData(id);st.dec=st.cur?decode(st.cur):null;try{localStorage.setItem('fsbWmDs',id);}catch(e){}render();}
async function open(opts){opts=opts||{};st.base=opts.base||st.base;onClose=opts.onClose||null;ensure().classList.add('show');
  host.firstChild.innerHTML=header()+'<p class="src">Loading…</p>';wire();await loadIndex();
  if(!st.files.length){render();return;}let id=opts.id;if(!id){try{id=localStorage.getItem('fsbWmDs');}catch(e){}}
  if(!st.files.some(f=>f.file===id))id=defaultId();if(!st.cur||st.cur.id!==id)await select(id);else render();}
// default = the main-arm QA seed nearest the median (never the tuning seed), so the first view is typical
function defaultId(){const F=st.files,q=F.filter(f=>f.arm==='main'&&f.condition==='QA'&&!f.tuning_seed&&f.FSV!=null);if(!q.length)return F[0].file;
  const m=median(F.filter(f=>f.arm==='main'&&f.condition==='QA').map(fsvOf));return q.reduce((b,f)=>Math.abs(f.FSV-m)<Math.abs(b.FSV-m)?f:b).file;}
function close(){if(host)host.classList.remove('show');if(onClose)onClose();}
root.FSBWorldModel=Object.assign({open,close,isOpen:()=>!!host&&host.classList.contains('show'),state:st},API);
})(typeof window!=='undefined'?window:globalThis);
