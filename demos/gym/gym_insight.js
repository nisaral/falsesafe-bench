/* FalseSafe Gym insight layer, on top of gym.js (window.__gym):
   - Timeline: V_true and V̂ along the current (or last) episode, false-safe stretches shaded,
     shield overrides ticked; drag (or [ / ]) to rewind and inspect any moment.
   - Map: hover any state for its V_true / V̂ / class, click to start an episode there, ⤢ for a bigger map.
   - Trade-off chart: FSV vs FUV of every certifier in the scenario (Python metrics from the export,
     whole map or gap only); click a dot to switch certifier.
   - Shield trigger margin slider (exploratory; in-browser shield only).
   Every value drawn is either an exported Python metric or a lookup in the exported HJ grids along
   the in-browser rollout; nothing here is a reported result. */
(function(){
'use strict';
const $=id=>document.getElementById(id);
// a timer, not requestAnimationFrame, so it also attaches in a background tab
(function boot(){const g=window.__gym;if(!g||!g.ready||!g.S||!g.ST.x){setTimeout(boot,50);return;}init(g);})();

function init(g){
const ST=g.ST,COL={vt:'#2ea043',vh:'#58a6ff',fs:'#ff2bd6',sh:'#d29922',bad:'#f85149',mut:'#8b949e',grid:'#30363d'};
const pct=v=>v==null?'n/a':(100*v).toFixed(v>0&&v<0.001?3:2)+'%';
function cls(l,vt,vh){return l<0?['hazard (crash)',COL.bad]:vt>=0&&vh>=0?['safe, and the guard knows it',COL.vt]:vt<0&&vh>=0?['FALSE-SAFE trap',COL.fs]
  :vt>=0?['false-unsafe (guard over-cautious)',COL.sh]:['truly unsafe (guard agrees)',COL.mut];}
function setPausedUI(){$('pauseBtn').classList.toggle('on',!!ST.paused);}

// ------------------------------------------------------------------ timeline
const tl=$('tl'),cv=$('tlc'),cx=cv.getContext('2d'),read=$('tlRead');
const cache={tr:null,ci:-1,n:0,vt:[],vh:[]};let hover=null,drag=false,saved=null,geo=null;
function src(){const r=ST.replay;
  if(r)return{tr:r.tr,flags:r.flags||[],cur:r.i,crashed:!!r.crashed,replay:true,ci:r.ci!=null?r.ci:(ST.last&&ST.last.tr===r.tr?ST.last.case:ST.ci)};
  return{tr:ST.trace,flags:ST.flags||[],cur:ST.trace.length-1,crashed:ST.crashT>0,replay:false,ci:ST.ci};}
function values(s){const S=g.S;if(cache.tr!==s.tr||cache.ci!==s.ci||cache.n>s.tr.length){cache.tr=s.tr;cache.ci=s.ci;cache.n=0;cache.vt=[];cache.vh=[];}
  const V=S.cases[s.ci].V;for(let i=cache.n;i<s.tr.length;i++){cache.vt.push(S.Vtrue(s.tr[i]));cache.vh.push(V(s.tr[i]));}cache.n=s.tr.length;return cache;}
function niceStep(T){return T<=4?0.5:T<=8?1:T<=20?2:5;}
function draw(){const S=g.S,r=cv.getBoundingClientRect();if(!r.width)return;const d=Math.min(2,devicePixelRatio||1),W=Math.round(r.width*d),H=Math.round(r.height*d);
  if(cv.width!==W||cv.height!==H){cv.width=W;cv.height=H;}
  // time axis grows with the episode (min 3 s, max the horizon) so short crashes are not squashed
  const s=src(),c=values(s),n=c.n,dt=S.doc.dt,Tmax=S.doc.max_steps*dt,T=Math.min(Tmax,Math.max(3,Math.ceil(n*dt*1.15))),mg=S.doc.filter_margin;
  const pL=30*d,pR=6*d,pT=4*d,pB=13*d,pw=W-pL-pR,ph=H-pT-pB;
  let m=Math.max(0.3,2.2*mg);for(let i=0;i<n;i++){const a=Math.abs(c.vt[i]),b=Math.abs(c.vh[i]);if(a>m)m=a;if(b>m)m=b;}m*=1.08;
  const X=i=>pL+pw*Math.min(1,i*dt/T),Y=v=>pT+ph*(0.5-0.5*Math.max(-1,Math.min(1,v/m)));geo={pL,pw,T,dt,n,d};
  cx.clearRect(0,0,W,H);cx.font=`${10*d}px system-ui`;cx.textBaseline='middle';
  // unsafe half tint, false-safe stretches, override ticks
  cx.fillStyle='#f8514910';cx.fillRect(pL,Y(0),pw,pT+ph-Y(0));
  const bw=Math.max(1,pw*dt/T+0.5);
  for(let i=0;i<n;i++){if(c.vh[i]>=0&&c.vt[i]<0){cx.fillStyle='#ff2bd640';cx.fillRect(X(i),pT,bw,ph);}
    const f=s.flags[i];if(f&&f[0]){cx.fillStyle=COL.sh;cx.fillRect(X(i),pT+ph-3*d,bw,3*d);}}
  // axes
  cx.strokeStyle=COL.grid;cx.lineWidth=d;cx.beginPath();cx.moveTo(pL,Y(0));cx.lineTo(pL+pw,Y(0));cx.stroke();
  cx.setLineDash([2*d,3*d]);cx.strokeStyle='#ffffffa0';cx.beginPath();cx.moveTo(pL,Y(mg));cx.lineTo(pL+pw,Y(mg));cx.stroke();cx.setLineDash([]);
  cx.fillStyle=COL.mut;cx.textAlign='right';cx.fillText('safe',pL-4*d,pT+6*d);cx.fillText('0',pL-4*d,Y(0));cx.fillText('crash',pL-4*d,pT+ph-6*d);
  cx.textAlign='center';const st=niceStep(T);for(let t=0;t<=T+1e-9;t+=st){const x=pL+pw*t/T;cx.fillText(t+'s',Math.min(W-8*d,Math.max(pL+6*d,x)),H-5*d);}
  // series
  const line=(arr,col,w)=>{if(n<1)return;cx.strokeStyle=col;cx.lineWidth=w*d;cx.beginPath();for(let i=0;i<n;i++){const x=X(i),y=Y(arr[i]);i?cx.lineTo(x,y):cx.moveTo(x,y);}cx.stroke();};
  line(c.vt,COL.vt,1.8);line(c.vh,COL.vh,1.8);
  if(s.crashed&&n){const x=X(n-1),y=Y(c.vt[n-1]),k=5*d;cx.strokeStyle=COL.bad;cx.lineWidth=2.5*d;cx.beginPath();cx.moveTo(x-k,y-k);cx.lineTo(x+k,y+k);cx.moveTo(x+k,y-k);cx.lineTo(x-k,y+k);cx.stroke();}
  // cursor (rewound) and hover
  const mark=(i,col)=>{if(i==null||i<0||i>=n)return;const x=X(i);cx.strokeStyle=col;cx.lineWidth=1.5*d;cx.beginPath();cx.moveTo(x,pT);cx.lineTo(x,pT+ph);cx.stroke();
    for(const[a,cl]of[[c.vt,COL.vt],[c.vh,COL.vh]]){cx.fillStyle=cl;cx.beginPath();cx.arc(x,Y(a[i]),3.2*d,0,7);cx.fill();}};
  if(s.replay)mark(s.cur,'#ffffff');if(hover!=null&&hover!==s.cur)mark(hover,'#ffffff70');
  const i=hover!=null?hover:s.replay?s.cur:null;
  if(i!=null&&i<n){const fs=c.vh[i]>=0&&c.vt[i]<0,f=s.flags[i];
    read.innerHTML=`t ${(i*dt).toFixed(2)}s · V_true <b style="color:${c.vt[i]>=0?COL.vt:COL.bad}">${c.vt[i].toFixed(2)}</b> · V̂ <b style="color:${COL.vh}">${c.vh[i].toFixed(2)}</b>`+(fs?` · <b style="color:${COL.fs}">false-safe</b>`:'')+(f&&f[0]?` · <b style="color:${COL.sh}">override</b>`:'');}
  else read.textContent=n>1?'drag to rewind · [ ] step':'';
  tl.classList.toggle('scrub',!!(ST.replay&&ST.paused));}
function idxAt(ev){if(!geo)return null;const r=cv.getBoundingClientRect(),px=(ev.clientX-r.left)*geo.d;
  const i=Math.round((px-geo.pL)/geo.pw*geo.T/geo.dt);return Math.max(0,Math.min(geo.n-1,i));}
function scrubTo(i){
  if(!(ST.replay&&ST.paused)){
    const live=ST.trace.length>=10||!ST.last;
    if(!ST.replay){saved={x:ST.x.slice()};
      ST.replay=live?{tr:ST.trace.slice(),flags:(ST.flags||[]).slice(),i:0,crashed:ST.crashT>0,ci:ST.ci,scrub:true}:{tr:ST.last.tr,flags:ST.last.flags,i:0,crashed:ST.last.crashed,ci:ST.last.case,scrub:true};}
    ST.paused=true;setPausedUI();}
  const r=ST.replay;r.i=Math.max(0,Math.min(r.tr.length-1,i));ST.x=r.tr[r.i].slice();ST.active=!!(r.flags[Math.min(r.i,r.flags.length-1)]||[])[0];}
function backToLive(){const S=g.S;ST.replay=null;ST.paused=false;setPausedUI();
  const x=ST.trace[ST.trace.length-1];ST.x=(x||saved&&saved.x||S.doc.start).slice();saved=null;
  if(S.l(ST.x)<0||ST.step>=S.doc.max_steps)g.reset();}
cv.addEventListener('pointerdown',ev=>{const i=idxAt(ev);if(i==null||geo.n<2)return;drag=true;cv.setPointerCapture(ev.pointerId);scrubTo(i);hover=null;});
cv.addEventListener('pointermove',ev=>{const i=idxAt(ev);if(drag){if(i!=null)scrubTo(i);}else hover=i;});
cv.addEventListener('pointerup',()=>{drag=false;});cv.addEventListener('pointercancel',()=>{drag=false;});cv.addEventListener('pointerleave',()=>{hover=null;});
$('tlResume').onclick=backToLive;
addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  if(e.code==='BracketLeft'||e.code==='BracketRight'){const s=src(),k=(e.code==='BracketLeft'?-1:1)*(e.shiftKey?10:1);scrubTo((s.replay?s.cur:s.tr.length-1)+k);e.preventDefault();}});

// ------------------------------------------------------------------ interactive map
const mc=$('map'),tip=$('mapTip'),main=$('main');
function stateAt(ev){const S=g.S,e=S.env,r=mc.getBoundingClientRect(),fx=(ev.clientX-r.left)/r.width,fy=(ev.clientY-r.top)/r.height;
  if(fx<0||fx>1||fy<0||fy>1)return null;const x=new Array(S.D).fill(0);x[0]=e.lo[0]+(e.hi[0]-e.lo[0])*fx;x[1]=e.hi[1]-(e.hi[1]-e.lo[1])*fy;if(S.D===3)x[2]=ST.x[2];return x;}
mc.addEventListener('pointermove',ev=>{const x=stateAt(ev);if(!x){tip.style.display='none';return;}
  const S=g.S,e=S.env,vt=S.Vtrue(x),vh=S.cases[ST.ci].V(x),l=S.l(x),[name,col]=cls(l,vt,vh);
  tip.innerHTML=`<div class="cls" style="color:${col}">${name}</div>${e.state_names[0]} ${x[0].toFixed(2)} · ${e.state_names[1]} ${x[1].toFixed(2)}`+(S.D===3?`<br>${e.state_names[2]} ${x[2].toFixed(2)} (current slice)`:'')+
    `<br>V_true <b style="color:${vt>=0?COL.vt:COL.bad}">${vt.toFixed(3)}</b> · V̂ <b style="color:${COL.vh}">${vh.toFixed(3)}</b>`+(S.gap(x)?'<br><span style="color:#58a6ff">inside the hidden-physics gap</span>':'')+
    `<div class="hint">${l<0?'inside the hazard':'click to start an episode here'}</div>`;
  const mr=main.getBoundingClientRect();tip.style.display='block';let tx=ev.clientX-mr.left-tip.offsetWidth-14;if(tx<4)tx=ev.clientX-mr.left+14;
  tip.style.left=tx+'px';tip.style.top=Math.min(mr.height-tip.offsetHeight-4,ev.clientY-mr.top+12)+'px';});
mc.addEventListener('pointerleave',()=>{tip.style.display='none';});
mc.addEventListener('click',ev=>{const x=stateAt(ev);if(!x)return;const S=g.S;
  if(S.l(x)<0){g.plain('That point is inside the hazard. Pick a spot outside the dark red.','crash');return;}
  ST.replay=null;saved=null;ST.paused=false;setPausedUI();g.reset(x);const vt=S.Vtrue(x),vh=S.cases[ST.ci].V(x),fs=vh>=0&&vt<0;
  g.plain('Started from your point: '+(fs?'<b>a false-safe trap</b>. The guard thinks this is fine, but a crash is already unavoidable. Watch the timeline.'
    :vt<0?'a truly unsafe state; the guard agrees, and a crash is unavoidable.':vh<0?'an over-cautious zone: actually safe, but the guard will step in.':'a safe state.'),fs?'fs':'ok');});
$('mapX').onclick=()=>{const w=$('mapwrap');w.classList.toggle('big');$('mapX').textContent=w.classList.contains('big')?'⤡':'⤢';$('mapX').title=w.classList.contains('big')?'Smaller map':'Bigger map';};

// ------------------------------------------------------------------ certifier trade-off chart
const PAL=['#58a6ff','#d2a8ff','#ffa657','#7ee787','#ff7b72','#79c0ff','#f2cc60','#a5d6ff'];let region='all',tkey='';
document.querySelectorAll('#trade .seg button').forEach(b=>b.onclick=()=>{region=b.dataset.r;document.querySelectorAll('#trade .seg button').forEach(o=>o.classList.toggle('on',o===b));trade(true);});
function esc(s){return String(s).replace(/[&<>"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[ch]);}
function trade(force){const S=g.S,key=S.doc.name+'|'+ST.ci+'|'+region;if(!force&&key===tkey)return;tkey=key;
  const pts=S.cases.map((c,k)=>{const m=c.metrics||{};return{k,label:c.label,fs:region==='gap'?m.gap_fsv:m.fsv,fu:region==='gap'?m.gap_fuv:m.fuv,oracle:/oracle/i.test(c.label)};}).filter(p=>p.fs!=null&&p.fu!=null);
  const W=290,H=150,L=40,R=10,T=10,B=26,mx=Math.max(0.002,...pts.map(p=>p.fu))*1.15,my=Math.max(0.002,...pts.map(p=>p.fs))*1.15;
  const X=v=>L+(W-L-R)*v/mx,Y=v=>H-B-(H-B-T)*v/my;let s='';
  for(const f of[0,0.5,1]){s+=`<line x1="${L}" x2="${W-R}" y1="${Y(f*my/1.15)}" y2="${Y(f*my/1.15)}" stroke="#21262d"/><text x="${L-4}" y="${Y(f*my/1.15)+3}" text-anchor="end">${pct(f*my/1.15)}</text>`;
    s+=`<line y1="${T}" y2="${H-B}" x1="${X(f*mx/1.15)}" x2="${X(f*mx/1.15)}" stroke="#21262d"/><text x="${X(f*mx/1.15)}" y="${H-B+12}" text-anchor="middle">${pct(f*mx/1.15)}</text>`;}
  s+=`<line x1="${L}" x2="${W-R}" y1="${H-B}" y2="${H-B}" stroke="#30363d"/><line x1="${L}" x2="${L}" y1="${T}" y2="${H-B}" stroke="#30363d"/>`;
  s+=`<text x="${(L+W-R)/2}" y="${H-2}" text-anchor="middle" fill="#d29922">false-unsafe (over-caution) →</text><text transform="translate(9 ${(T+H-B)/2}) rotate(-90)" text-anchor="middle" fill="#ff2bd6">false-safe (danger) →</text>`;
  s+=`<text x="${L+4}" y="${H-B-5}" fill="#7ee787">★ ideal</text>`;
  // Pareto front over the real certifiers (oracle excluded): no other point is both safer and less cautious
  const real=pts.filter(p=>!p.oracle).sort((a,b)=>a.fu-b.fu||a.fs-b.fs),front=[];let best=Infinity;for(const p of real)if(p.fs<best-1e-12){front.push(p);best=p.fs;}
  if(front.length>1)s+=`<polyline points="${front.map(p=>X(p.fu)+','+Y(p.fs)).join(' ')}" fill="none" stroke="#8b949e" stroke-dasharray="3 3"/>`;
  const seen={};pts.forEach(p=>{const cur=p.k===ST.ci,col=PAL[p.k%PAL.length],kk=Math.round(X(p.fu))+','+Math.round(Y(p.fs)),dup=(seen[kk]=(seen[kk]||0)+1)-1;
    s+=`<circle class="pt" data-k="${p.k}" cx="${X(p.fu)}" cy="${Y(p.fs)}" r="${cur?7:5}" fill="${p.oracle?'none':col}" stroke="${cur?'#fff':p.oracle?col:'#0d1117'}" stroke-width="${cur?2:p.oracle?2:1}"><title>${esc(p.label)}: FSV ${pct(p.fs)}, FUV ${pct(p.fu)}</title></circle>`+
      `<text class="${cur?'cur':''}" x="${X(p.fu)+9+dup*10}" y="${Y(p.fs)-6}">${p.k+1}</text>`;});
  const svg=$('tradeSvg');svg.innerHTML=s;svg.querySelectorAll('circle.pt').forEach(c=>c.onclick=()=>pick(+c.dataset.k));
  $('tradeList').innerHTML=pts.map(p=>`<li data-k="${p.k}" class="${p.k===ST.ci?'on':''}"><i style="background:${p.oracle?'transparent':PAL[p.k%PAL.length]};${p.oracle?`border:2px solid ${PAL[p.k%PAL.length]};color:${PAL[p.k%PAL.length]};width:12px;height:12px;line-height:12px`:''}">${p.k+1}</i>${esc(p.label)}<span class="v">${pct(p.fs)} · ${pct(p.fu)}</span></li>`).join('');
  $('tradeList').querySelectorAll('li').forEach(li=>li.onclick=()=>pick(+li.dataset.k));}
function pick(k){if(k===ST.ci)return;g.setCase(k);g.log('Certifier: '+g.S.cases[k].label,'ok');g.plain(`Rule-book switched to <b>${g.S.cases[k].label}</b>. Same scenario, new guard.`,'ok');trade(true);}

// ------------------------------------------------------------------ shield trigger margin (exploratory)
const mr=$('marg'),mv=$('margV');let mkey='';
function syncMargin(){const d=g.S.doc;if(d.margin0==null)d.margin0=d.filter_margin;mr.max=Math.max(0.6,Math.ceil(d.margin0*10*2)/10);
  mr.value=d.filter_margin;mv.textContent=d.filter_margin.toFixed(2);$('margReset').disabled=d.filter_margin===d.margin0;$('margReset').title=`Back to the exported value (${d.margin0})`;}
mr.oninput=()=>{g.setMargin(+mr.value,true);mv.textContent=(+mr.value).toFixed(2);};
mr.onchange=()=>{const d=g.S.doc;g.resetCounts();g.reset();g.plain(`Shield trigger margin set to <b>${d.filter_margin.toFixed(2)}</b>${d.filter_margin===d.margin0?' (exported default)':''}. Watch where the dotted line moves on the map. Episode and counters restarted.`,'sh');syncMargin();};
$('margReset').onclick=()=>{const d=g.S.doc;g.setMargin(d.margin0);syncMargin();};

// ------------------------------------------------------------------ loop
let lastDraw=0;
(function loop(now){requestAnimationFrame(loop);if(document.hidden||!g.S||!ST.x)return;
  const k=g.S.doc.name+'|'+g.S.doc.filter_margin;if(k!==mkey){mkey=k;syncMargin();}
  trade(false);if(now-lastDraw>33){lastDraw=now;draw();}})(0);
window.__gymInsight={draw,trade,scrubTo,backToLive,cache};
}
})();
