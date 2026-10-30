/* FalseSafe 3D viewer. Renders ONLY data exported by lcert.framework (window.FSB_DATASETS
   from data/bundle.js, or JSON files loaded by the user). No values are synthesised here. */
(function(){
const $=id=>document.getElementById(id);
// Lazy mode (site build): data/index.js lists datasets; each lives in data/ds_<name>.js and is fetched on demand.
const LAZY=!window.FSB_DATASETS&&window.FSB_DATASET_INDEX;
const DS=LAZY?window.FSB_DATASET_INDEX.map(e=>Object.assign({_stub:true},e)):(window.FSB_DATASETS||[]).slice();
function loadDS(i){const d=DS[i];if(!d||!d._stub)return Promise.resolve(d);if(d._p)return d._p;
 d._p=new Promise((ok,bad)=>{const sc=document.createElement('script');sc.src='data/ds_'+d.name+'.js';sc.onload=()=>{DS[i]=window.FSB_DS_LOADED[d.name];ok(DS[i]);};sc.onerror=bad;document.head.appendChild(sc);});return d._p;}
const S={ds:0,sel:{},slice:0,tr:0,t:0,playing:true,layers:{true:true,hat:true,floor:true,gap:true,zero:true,traj:true}};
const COL={safe:[0x2e,0xa0,0x43],fs:[0xff,0x2b,0xd6],fu:[0xd2,0x99,0x22],uns:[0x30,0x36,0x3d]};
const W=10; // world half-size of the plot box
// ---------- three setup
const cv=$('c3d'),R=new THREE.WebGLRenderer({canvas:cv,antialias:true,preserveDrawingBuffer:true});
R.setPixelRatio(Math.min(1.5,devicePixelRatio||1));
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0d1117);
const cam=new THREE.PerspectiveCamera(45,1,0.1,200);cam.position.set(14,13,17);
const ctl=new THREE.OrbitControls(cam,cv);ctl.target.set(0,0,0);ctl.enableDamping=true;
scene.add(new THREE.AmbientLight(0xffffff,0.55));const dl=new THREE.DirectionalLight(0xffffff,0.8);dl.position.set(5,20,10);scene.add(dl);
let G=new THREE.Group();scene.add(G);
const cw=$('cw'),RW=new THREE.WebGLRenderer({canvas:cw,antialias:true,preserveDrawingBuffer:true});RW.setPixelRatio(Math.min(1.5,devicePixelRatio||1));
const sw=new THREE.Scene();sw.background=new THREE.Color(0x0b0f14);const camW=new THREE.PerspectiveCamera(40,3,0.1,200);
sw.add(new THREE.AmbientLight(0xffffff,0.6));const dl2=new THREE.DirectionalLight(0xffffff,0.8);dl2.position.set(3,10,6);sw.add(dl2);
let GW=new THREE.Group();sw.add(GW);let robot=null,ghost=null,marker=null;
function resize(){const a=cv.parentElement.getBoundingClientRect();R.setSize(a.width,a.height,false);cam.aspect=a.width/a.height;cam.updateProjectionMatrix();
 const b=$('world').getBoundingClientRect();RW.setSize(b.width,b.height,false);camW.aspect=b.width/b.height;camW.updateProjectionMatrix();}
addEventListener('resize',resize);
// ---------- helpers
const ds=()=>DS[S.ds];
function curCase(){const d=ds();return d.cases.find(c=>d.sweep_keys.every(k=>JSON.stringify(c.params[k])===JSON.stringify(S.sel[k])))||d.cases[0];}
const fmt=x=>x==null?'n/a':(100*x).toFixed(2)+'%';
function mapX(d,x){const a=d.axes[0];return -W+2*W*(x-a[0])/(a[a.length-1]-a[0]);}
function mapY(d,y){const a=d.axes[1];return  W-2*W*(y-a[0])/(a[a.length-1]-a[0]);}
function vrange(c){let m=0;for(const A of [c.v_true[S.slice],c.v_hat[S.slice]])for(const r of A)for(const v of r)m=Math.max(m,Math.abs(v));return m||1;}
function surface(d,A,zs,colorFn,opts){const n0=A.length,n1=A[0].length,g=new THREE.BufferGeometry(),P=[],C=[],I=[];
 for(let i=0;i<n0;i++)for(let j=0;j<n1;j++){P.push(mapX(d,d.axes[0][i]),A[i][j]*zs,mapY(d,d.axes[1][j]));const c=colorFn(i,j);C.push(c[0],c[1],c[2]);}
 for(let i=0;i<n0-1;i++)for(let j=0;j<n1-1;j++){const a=i*n1+j,b=a+1,c=a+n1,e=c+1;I.push(a,c,b,b,c,e);}
 g.setAttribute('position',new THREE.Float32BufferAttribute(P,3));g.setAttribute('color',new THREE.Float32BufferAttribute(C,3));g.setIndex(I);g.computeVertexNormals();
 return new THREE.Mesh(g,new THREE.MeshPhongMaterial(Object.assign({vertexColors:true,side:THREE.DoubleSide,shininess:20},opts)));}
function zeroContour(d,A,y,color){const pts=[],n0=A.length,n1=A[0].length,X=d.axes[0],Y=d.axes[1];
 const lerp=(i0,j0,i1,j1)=>{const a=A[i0][j0],b=A[i1][j1],t=a/(a-b);return new THREE.Vector3(mapX(d,X[i0]+t*(X[i1]-X[i0])),y,mapY(d,Y[j0]+t*(Y[j1]-Y[j0])));};
 for(let i=0;i<n0-1;i++)for(let j=0;j<n1-1;j++){const e=[];
  if((A[i][j]>=0)!=(A[i+1][j]>=0))e.push(lerp(i,j,i+1,j));if((A[i+1][j]>=0)!=(A[i+1][j+1]>=0))e.push(lerp(i+1,j,i+1,j+1));
  if((A[i][j+1]>=0)!=(A[i+1][j+1]>=0))e.push(lerp(i,j+1,i+1,j+1));if((A[i][j]>=0)!=(A[i][j+1]>=0))e.push(lerp(i,j,i,j+1));
  if(e.length>=2){pts.push(e[0],e[1]);}if(e.length==4)pts.push(e[2],e[3]);}
 return new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(pts),new THREE.LineBasicMaterial({color}));}
function textSprite(t,color='#e6edf3',s=1){const c=document.createElement('canvas');c.width=512;c.height=64;const x=c.getContext('2d');x.font='28px system-ui';x.fillStyle=color;x.fillText(t,4,40);
 const sp=new THREE.Sprite(new THREE.SpriteMaterial({map:new THREE.CanvasTexture(c),transparent:true,depthTest:false}));sp.scale.set(8*s,1*s,1);return sp;}
function interpTraj(d,st){/* map a state to plot coords; for 3D envs show (x,y) of the trajectory */return [mapX(d,st[0]),mapY(d,st[1])];}
// ---------- build 3D scene for current case
function build(){scene.remove(G);G=new THREE.Group();scene.add(G);const d=ds(),c=curCase(),T=c.v_true[S.slice],H=c.v_hat[S.slice];
 const zs=+$('zs').value*3.6/vrange(c);$('zsv').textContent=(+$('zs').value).toFixed(1)+'x';S.zs=zs;
 const n0=T.length,n1=T[0].length;
 if(S.layers.true)G.add(surface(d,T,zs,(i,j)=>(H[i][j]>=0&&T[i][j]<0)?[1.0,0.17,0.84]:(T[i][j]>=0?[0.18,0.63,0.26]:[0.75,0.2,0.2]),{transparent:true,opacity:0.92}));
 if(S.layers.hat)G.add(surface(d,H,zs,()=>[0.35,0.65,1.0],{transparent:true,opacity:0.38,depthWrite:false}));
 if(S.layers.floor){const data=new Uint8Array(n0*n1*4);for(let i=0;i<n0;i++)for(let j=0;j<n1;j++){const st=T[i][j]>=0,sh=H[i][j]>=0;
   const k=st&&sh?COL.safe:(!st&&sh?COL.fs:(st&&!sh?COL.fu:COL.uns));const o=4*(j*n0+i);data[o]=k[0];data[o+1]=k[1];data[o+2]=k[2];data[o+3]=255;}
  const tex=new THREE.DataTexture(data,n0,n1,THREE.RGBAFormat);tex.needsUpdate=true;tex.magFilter=THREE.NearestFilter;
  const fl=new THREE.Mesh(new THREE.PlaneGeometry(2*W,2*W),new THREE.MeshBasicMaterial({map:tex,side:THREE.DoubleSide}));
  fl.rotation.x=-Math.PI/2;fl.position.y=-4.6;fl.scale.y=-1;G.add(fl);
  if(S.layers.zero){G.add(zeroContour(d,T,-4.55,0xffffff));G.add(zeroContour(d,H,-4.5,0x58a6ff));}}
 if(S.layers.zero){G.add(zeroContour(d,T,0.02,0xffffff));G.add(zeroContour(d,H,0.04,0x58a6ff));
  const p=new THREE.Mesh(new THREE.PlaneGeometry(2*W,2*W),new THREE.MeshBasicMaterial({color:0xffffff,transparent:true,opacity:0.05,side:THREE.DoubleSide,depthWrite:false}));p.rotation.x=-Math.PI/2;G.add(p);}
 if(S.layers.gap&&c.gap){const x0=Math.max(mapX(d,c.gap.lo[0]),-W),x1=Math.min(mapX(d,c.gap.hi[0]),W),y0=Math.max(Math.min(mapY(d,c.gap.lo[1]),W),-W),y1=Math.max(Math.min(mapY(d,c.gap.hi[1]),W),-W);
  const b=new THREE.Mesh(new THREE.BoxGeometry(Math.abs(x1-x0),9.2,Math.abs(y1-y0)),new THREE.MeshBasicMaterial({color:0x58a6ff,transparent:true,opacity:0.09,depthWrite:false}));
  b.position.set((x0+x1)/2,0,(y0+y1)/2);G.add(b);const e=new THREE.LineSegments(new THREE.EdgesGeometry(b.geometry),new THREE.LineBasicMaterial({color:0x58a6ff}));e.position.copy(b.position);G.add(e);
  const l=textSprite('gap region',"#58a6ff",0.8);l.position.set((x0+x1)/2+2,5.2,(y0+y1)/2);G.add(l);}
 const ax=new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(2*W,9.2,2*W)),new THREE.LineBasicMaterial({color:0x30363d}));G.add(ax);
 const sn=d.env.state_names;const a=textSprite(sn[0]+' →');a.position.set(2,-5.2,W+1.2);G.add(a);const b2=textSprite(sn[1]+' →');b2.position.set(-W-1.5,-5.2,-1);G.add(b2);
 const z=textSprite('value V (≥0 safe)');z.position.set(-W+1,5.4,-W);G.add(z);
 for(const [ti,v] of [[0,d.axes[0][0]],[1,d.axes[0][d.axes[0].length-1]]]){const s=textSprite(v.toFixed(2),'#8b949e',0.6);s.position.set(ti?W:-W,-5.2,W+0.4);G.add(s);}
 for(const [ti,v] of [[0,d.axes[1][0]],[1,d.axes[1][d.axes[1].length-1]]]){const s=textSprite(v.toFixed(2),'#8b949e',0.6);s.position.set(-W-0.8,-5.2,ti?-W:W);G.add(s);}
 // trajectory
 const tr=c.trajectories[S.tr];marker=null;
 if(tr&&S.layers.traj){const showOnSlice=d.env.slice_dim==null;const P=tr.states.map((s,k)=>{const [x,y]=interpTraj(d,s);return new THREE.Vector3(x,showOnSlice?tr.v_true[k]*zs+0.08:-4.4,y);});
  G.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(P),new THREE.LineBasicMaterial({color:tr.hit_failure?0xff5555:0xffffff})));
  const P2=tr.states.map(s=>{const [x,y]=interpTraj(d,s);return new THREE.Vector3(x,-4.4,y);});const shadow=new THREE.Line(new THREE.BufferGeometry().setFromPoints(P2),new THREE.LineDashedMaterial({color:0xaaaaaa,dashSize:0.2,gapSize:0.15}));shadow.computeLineDistances();G.add(shadow);
  marker=new THREE.Mesh(new THREE.SphereGeometry(0.28,20,14),new THREE.MeshPhongMaterial({color:0xffd33d,emissive:0x553300}));G.add(marker);marker.userData={P,tr};}
 // overlays
 const m=c.metrics;const pr=Object.entries(c.params).map(([k,v])=>k+'='+JSON.stringify(v)).join(', ')||'single case';
 $('ov3').innerHTML=`<b>${d.name}</b> — ${pr}${d.env.slice_dim!=null?` — slice ${d.env.state_names[2]}=${d.env.slice_values[S.slice].toFixed(3)}`:''}<br>`+
  `<span style="color:#7ee787">solid: V_true (HJ ground truth, true dynamics)</span> · <span style="color:#58a6ff">translucent: V_hat (${c.certifier})</span> · white/blue lines: zero level sets`;
 $('met').innerHTML=[['FSV (false-safe volume)',fmt(m.fsv)],['FUV (false-unsafe volume)',fmt(m.fuv)],['gap-FSV',fmt(m.gap_fsv)],['non-gap FSV',fmt(m.non_gap_fsv)],['IoU',m.iou.toFixed(4)],
  ['true safe vol',fmt(m.safe_true_volume)],['certified safe vol',fmt(m.safe_hat_volume)],['eval cells',m.num_eval_cells]].map(r=>`<tr><td>${r[0]}</td><td>${r[1]}</td></tr>`).join('');
 $('msrc').textContent=`Source: ${d.name}.json → cases[${d.cases.indexOf(c)}].metrics (lcert.metrics.compute_certification_metrics, full grid ${d.env.grid_shape.join('×')}, border ${d.provenance.border_cells} cells). Fractions of the evaluation domain.`;
 buildWorld();}
// ---------- world view (robot)
function buildWorld(){sw.remove(GW);GW=new THREE.Group();sw.add(GW);const d=ds(),c=curCase(),tr=c.trajectories[S.tr];robot=ghost=null;
 const a=d.axes[0],is3=d.env.slice_dim!=null,L=c.hazard_l?c.hazard_l[S.slice]:null;
 if(!is3){camW.position.set(0,6,13);camW.lookAt(0,0,0);
  const n=a.length,jm=Math.floor(d.axes[1].length/2);
  for(let i=0;i<n-1;i++){const x0=mapX(d,a[i]),x1=mapX(d,a[i+1]);const bad=L?L[i][jm]<0:false;
   const inGap=c.gap&&a[i]>=c.gap.lo[0]&&a[i]<=c.gap.hi[0];
   const t=new THREE.Mesh(new THREE.BoxGeometry(x1-x0+0.01,0.3,2.4),new THREE.MeshPhongMaterial({color:bad?0x8b1a1a:(inGap?0x1f4a7a:0x2b313a)}));t.position.set((x0+x1)/2,-0.15,0);GW.add(t);}
  const s=textSprite('track: '+d.env.state_names[0]+'  (red: hazard l<0 at v≈0, blue: gap)','#8b949e',0.9);s.position.set(-2,2.6,-2);GW.add(s);
  robot=new THREE.Group();const body=new THREE.Mesh(new THREE.BoxGeometry(0.9,0.5,1.2),new THREE.MeshPhongMaterial({color:0xffd33d}));body.position.y=0.4;robot.add(body);
  for(const z of [-0.6,0.6])for(const x of [-0.3,0.3]){const w=new THREE.Mesh(new THREE.CylinderGeometry(0.16,0.16,0.12,16),new THREE.MeshPhongMaterial({color:0x111111}));w.rotation.x=Math.PI/2;w.position.set(x,0.16,z);robot.add(w);}
  GW.add(robot);}
 else{camW.position.set(0,16,9);camW.lookAt(0,0,0);
  const n0=a.length,n1=d.axes[1].length,data=new Uint8Array(n0*n1*4);
  for(let i=0;i<n0;i++)for(let j=0;j<n1;j++){const bad=L&&L[i][j]<0,o=4*(j*n0+i);const gx=c.gap&&a[i]>=c.gap.lo[0]&&a[i]<=c.gap.hi[0]&&d.axes[1][j]>=c.gap.lo[1]&&d.axes[1][j]<=c.gap.hi[1];
   const k=bad?[140,26,26]:(gx?[31,74,122]:[43,49,58]);data[o]=k[0];data[o+1]=k[1];data[o+2]=k[2];data[o+3]=255;}
  const tex=new THREE.DataTexture(data,n0,n1,THREE.RGBAFormat);tex.needsUpdate=true;const fl=new THREE.Mesh(new THREE.PlaneGeometry(2*W,2*W),new THREE.MeshBasicMaterial({map:tex}));fl.rotation.x=-Math.PI/2;fl.scale.y=-1;GW.add(fl);
  if(tr){const P=tr.states.map(s=>new THREE.Vector3(mapX(d,s[0]),0.05,mapY(d,s[1])));GW.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(P),new THREE.LineBasicMaterial({color:tr.hit_failure?0xff5555:0xffffff})));}
  robot=new THREE.Group();const body=new THREE.Mesh(new THREE.BoxGeometry(1.0,0.4,0.6),new THREE.MeshPhongMaterial({color:0xffd33d}));body.position.y=0.3;robot.add(body);
  const nose=new THREE.Mesh(new THREE.ConeGeometry(0.25,0.5,12),new THREE.MeshPhongMaterial({color:0xff7b00}));nose.rotation.z=-Math.PI/2;nose.position.set(0.7,0.3,0);robot.add(nose);GW.add(robot);}
 $('ovw').innerHTML=tr?`World view — rollout under TRUE dynamics (exported, RK4 dt=${tr.dt}). ${tr.mode=='filtered_by_certifier'?'Task policy + safety filter driven by V_hat':'Task policy only, no filter'}. `+
  (tr.hit_failure?`<b style="color:#ff7b7b">Hit the true failure set at step ${tr.hit_failure_step}.</b>`:'<span style="color:#7ee787">Did not enter the failure set within the horizon.</span>'):'No trajectories exported for this dataset.';}
// ---------- UI
function layerButtons(){$('layers').innerHTML='';for(const k of Object.keys(S.layers)){const b=document.createElement('button');b.textContent={true:'V_true surface',hat:'V_hat surface',floor:'error floor',gap:'gap region',zero:'zero sets',traj:'trajectory'}[k];
 b.className=S.layers[k]?'on':'';b.onclick=()=>{S.layers[k]=!S.layers[k];layerButtons();build();};$('layers').appendChild(b);}}
function initDataset(){const d=ds();S.sel={};S.slice=0;S.tr=0;S.t=0;
 $('badge').innerHTML=d.illustrative?'<span class="badge illu">ILLUSTRATIVE / synthetic</span>':'<span class="badge real">solver-exported data</span>';
 $('desc').textContent=(d.env.description||'')+' Hazard: '+d.env.hazard+'.';
 $('sliders').innerHTML='';for(const k of d.sweep_keys){const vals=d.sweep_values[k];S.sel[k]=vals[Math.min(vals.length-1,vals.length>2?Math.floor(vals.length/2):vals.length-1)];
  const lab=document.createElement('label');const inp=document.createElement('input');inp.type='range';inp.min=0;inp.max=vals.length-1;inp.step=1;inp.value=vals.indexOf(S.sel[k]);inp.id='sw_'+k;
  const upd=()=>{S.sel[k]=vals[+inp.value];lab.innerHTML=`${k} = <b>${JSON.stringify(S.sel[k])}</b> <span style="color:#555">(${vals.join(', ')})</span>`;};upd();
  inp.oninput=()=>{upd();S.t=0;fillTraj();build();};$('sliders').append(lab,inp);}
 if(!d.sweep_keys.length)$('sliders').innerHTML='<div class="src">single case (no sweep)</div>';
 $('slicebox').innerHTML='';if(d.env.slice_dim!=null){const lab=document.createElement('label');lab.textContent='Slice: '+d.env.state_names[d.env.slice_dim];const s=document.createElement('select');
  d.env.slice_values.forEach((v,i)=>{const o=document.createElement('option');o.value=i;o.textContent=v.toFixed(3);s.appendChild(o);});s.onchange=()=>{S.slice=+s.value;build();};$('slicebox').append(lab,s);}
 const p=d.provenance;$('prov').innerHTML=`${p.source}<br>created ${p.created}<br>commit ${p.git_commit||'n/a'}<br>${p.display_note}`;
 fillTraj();build();}
function fillTraj(){const c=curCase();$('tr').innerHTML='';c.trajectories.forEach((t,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${t.policy} · x0=${JSON.stringify(t.x0)} · ${t.mode}${t.hit_failure?' · CRASH':''}`;$('tr').appendChild(o);});
 S.tr=Math.min(S.tr,Math.max(0,c.trajectories.length-1));$('tr').value=S.tr;}
$('tr').onchange=()=>{S.tr=+$('tr').value;S.t=0;build();};$('zs').oninput=build;
$('play').onclick=()=>{S.playing=!S.playing;$('play').textContent=S.playing?'pause':'play';$('play').className=S.playing?'on':'';};$('restart').onclick=()=>{S.t=0;};
function refreshDS(){$('ds').innerHTML='';DS.forEach((d,i)=>{const o=document.createElement('option');o.value=i;o.textContent=d.name+(d.illustrative?' (illustrative)':'');$('ds').appendChild(o);});$('ds').value=S.ds;}
function selectDS(i){S.ds=i;$('ov3').textContent='loading '+DS[i].name+' …';return loadDS(i).then(()=>{if(S.ds===i)initDataset();});}
$('ds').onchange=()=>selectDS(+$('ds').value);
function addFiles(files){for(const f of files)f.text().then(t=>{const d=JSON.parse(t);if(d.schema!=='falsesafe.framework/v1'){alert(f.name+': not a falsesafe.framework/v1 export');return;}DS.push(d);S.ds=DS.length-1;refreshDS();initDataset();});}
$('file').onchange=e=>addFiles(e.target.files);document.body.ondragover=e=>e.preventDefault();document.body.ondrop=e=>{e.preventDefault();addFiles(e.dataTransfer.files);};
// ---------- animate
let last=performance.now(),hudT=0;
function tick(now){if(document.hidden){last=now;requestAnimationFrame(tick);return;}const dt=Math.max(0,Math.min(0.1,(now-last)/1000));last=now;ctl.update();const c=DS.length&&!ds()._stub?curCase():null,tr=c&&c.trajectories[S.tr];
 if(tr){if(S.playing)S.t+=dt/tr.dt*0.5;const n=tr.states.length;if(S.t>n+20)S.t=0;S.t=Math.max(0,S.t);const k=Math.min(n-1,Math.floor(S.t)),s=tr.states[k],d=ds();
  if(marker&&marker.userData.P[k])marker.position.copy(marker.userData.P[k]);
  if(robot){if(d.env.slice_dim==null){robot.position.set(mapX(d,s[0]),0,0);}else{robot.position.set(mapX(d,s[0]),0,mapY(d,s[1]));robot.rotation.y=s[2];}}
  if(now-hudT>100){hudT=now;$('trinfo').innerHTML=`step ${k}/${n-1} · t=${(k*tr.dt).toFixed(2)}s · state=${JSON.stringify(s)}<br>V_true=${tr.v_true[k].toFixed(3)} · V_hat=${tr.v_hat[k].toFixed(3)}`+
   (k>0&&tr.filter_active[k-1]?' · <b style="color:#58a6ff">filter active</b>':'')+(tr.v_hat[k]>=0&&tr.v_true[k]<0?' · <b style="color:#ff2bd6">in FALSE-SAFE region</b>':'')+'<br><span>values interpolated from the exported solver grids</span>';}}
 R.render(scene,cam);RW.render(sw,camW);requestAnimationFrame(tick);}
layerButtons();resize();
$('views').onclick=e=>{const v=e.target.dataset.v;if(!v)return;const P={iso:[14,13,17],top:[0,36,0.01],side:[0,2,26]}[v];cam.position.set(...P);ctl.target.set(0,0,0);};
if(!DS.length){$('ov3').innerHTML='No datasets found. Run <code>PYTHONPATH=src python -m lcert.framework run configs/framework/di_nominal_sweep.yaml</code> or load a JSON.';}
else{refreshDS();selectDS(0).then(()=>{window.__fsb.ready=true;});}
window.__fsb={S,DS,curCase,build,initDataset,selectDS,loadDS,ready:!DS.length};
requestAnimationFrame(tick);
})();
