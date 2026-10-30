/* FalseSafe Gym UI: scenario menu, manual/autonomous driving, certifier shield, HUD, mini-map,
   replay and batch mode.  Physics + shield live in gym_core.js (FSBGym). */
(function(){
'use strict';
const G=window.FSBGym,$=id=>document.getElementById(id);
const order=window.FSB_GYM_FILES||[];
const DOCS=(window.FSB_GYM||[]).slice().sort((a,b)=>order.indexOf(a.name)-order.indexOf(b.name));
const ST={si:0,ci:0,pol:'manual',shield:true,paused:false,x:null,t:0,step:0,trace:[],last:null,replay:null,
  score:0,crashes:0,episodes:0,interv:0,fsTime:0,crashT:0,events:[],keys:{},touch:{},banner:null,bannerT:0};
let S=null,polFn=null;
const fmt=(v,d=3)=>v==null||!isFinite(v)?'n/a':(+v).toFixed(d);
const pct=v=>v==null?'n/a':(100*v).toFixed(2)+'%';

// ------------------------------------------------------------------ three.js scene
const cv=$('c3d'),R=new THREE.WebGLRenderer({canvas:cv,antialias:true,preserveDrawingBuffer:true});
R.setPixelRatio(Math.min(1.5,devicePixelRatio||1));
const scene=new THREE.Scene();scene.background=new THREE.Color(0x0b1018);scene.fog=new THREE.Fog(0x0b1018,40,90);
const cam=new THREE.PerspectiveCamera(45,1,0.1,300);const ctl=new THREE.OrbitControls(cam,cv);ctl.enableDamping=true;
scene.add(new THREE.HemisphereLight(0xbcd7ff,0x202a33,0.75));const sun=new THREE.DirectionalLight(0xffffff,0.8);sun.position.set(8,20,10);scene.add(sun);
let W=new THREE.Group();scene.add(W);let agent=null,ring=null,trail=null,goalM=null,spark=null,dyn=[];
function resize(){const r=$('main').getBoundingClientRect();R.setSize(r.width,r.height,false);cam.aspect=r.width/Math.max(1,r.height);cam.updateProjectionMatrix();}
addEventListener('resize',resize);
const M=(c,o)=>new THREE.MeshStandardMaterial(Object.assign({color:c,roughness:0.7,metalness:0.1},o||{}));
function box(w,h,d,c,o){return new THREE.Mesh(new THREE.BoxGeometry(w,h,d),M(c,o));}
function cyl(r,h,c,o,seg){return new THREE.Mesh(new THREE.CylinderGeometry(r,r,h,seg||32),M(c,o));}
// planar mapping: state (x, y) -> world (X, Z) with +y away from camera
let SC=1;const P=(x,y,h)=>new THREE.Vector3(x*SC,h||0,-y*SC);

function mapTexture(){const t=new THREE.CanvasTexture($('map'));t.magFilter=THREE.NearestFilter;return t;}
function buildScene(){
  scene.remove(W);W=new THREE.Group();scene.add(W);dyn=[];const e=S.env,pr=e.params,kind=S.doc.render;
  const lo=e.lo,hi=e.hi;
  if(['car','rover','boat','point'].includes(kind)&&!(kind==='point'&&false)){
    SC=16/Math.max(hi[0]-lo[0],hi[1]-lo[1]);
    const gw=(hi[0]-lo[0])*SC,gd=(hi[1]-lo[1])*SC,cx=(lo[0]+hi[0])/2,cy=(lo[1]+hi[1])/2;
    const ground=new THREE.Mesh(new THREE.PlaneGeometry(gw,gd),new THREE.MeshBasicMaterial({map:mapTexture(),transparent:true,opacity:kind==='boat'?0.55:0.9}));
    ground.rotation.x=-Math.PI/2;ground.position.copy(P(cx,cy,kind==='boat'?0.02:0.01));W.add(ground);dyn.push(()=>{ground.material.map.needsUpdate=true;});
    if(kind==='boat'){const water=new THREE.Mesh(new THREE.PlaneGeometry(gw*1.6,gd*1.6),M(0x1b4f7a,{roughness:0.3,metalness:0.2,transparent:true,opacity:0.8}));water.rotation.x=-Math.PI/2;water.position.copy(P(cx,cy,-0.02));W.add(water);
      const sand=box(gw*1.6,0.4,(pr.shore-lo[1]+1)*SC,0xc2a45a);sand.position.copy(P(cx,(pr.shore+lo[1]-1)/2,0.1));W.add(sand);
      const rock=new THREE.Mesh(new THREE.DodecahedronGeometry(pr.rock[2]*SC,1),M(0x6b6b6b,{flatShading:true}));rock.position.copy(P(pr.rock[0],pr.rock[1],0.2));W.add(rock);
      // current arrows (hidden physics) drift across the band
      const arrows=[];for(let i=0;i<24;i++){const a=new THREE.ArrowHelper(new THREE.Vector3(pr.current[0],0,-pr.current[1]).normalize(),new THREE.Vector3(),0.9,0x58a6ff,0.3,0.2);a.userData={x:lo[0]+Math.random()*(hi[0]-lo[0]),y:pr.band[0]+Math.random()*(pr.band[1]-pr.band[0])};W.add(a);arrows.push(a);}
      dyn.push(dt=>arrows.forEach(a=>{const u=a.userData;u.x+=pr.current[0]*dt;u.y+=pr.current[1]*dt;if(u.y<pr.band[0]){u.y=pr.band[1];}if(u.x>hi[0])u.x=lo[0];a.position.copy(P(u.x,u.y,0.15));}));}
    if(kind==='car'){const ob=pr.obstacle||[0,0,0.8];const c=cyl(ob[2]*SC,1.4,0xbf3333);c.position.copy(P(ob[0],ob[1],0.7));W.add(c);
      const pa=pr.patch||[0.8,2,-1.5,1.5];const ice=box((pa[1]-pa[0])*SC,0.03,(pa[3]-pa[2])*SC,0x9fd4ff,{transparent:true,opacity:0.35,metalness:0.6,roughness:0.05});ice.position.copy(P((pa[0]+pa[1])/2,(pa[2]+pa[3])/2,0.04));W.add(ice);}
    if(kind==='rover'){const drop=box(gw,6,(hi[1]-pr.cliff)*SC,0x1a0d0d);drop.position.copy(P(cx,(pr.cliff+hi[1])/2,-3.05));W.add(drop);
      const edge=box(gw,0.25,0.15,0xf85149);edge.position.copy(P(cx,pr.cliff,0.12));W.add(edge);
      const pit=cyl(pr.pit[2]*SC,0.06,0x050505,null,40);pit.position.copy(P(pr.pit[0],pr.pit[1],0.03));W.add(pit);
      const scree=box(gw,0.02,(pr.cliff-pr.slope_lo)*SC,0x8a6d3b,{transparent:true,opacity:0.3});scree.position.copy(P(cx,(pr.cliff+pr.slope_lo)/2,0.03));W.add(scree);}
    if(kind==='car'){agent=new THREE.Group();const b=box(0.9,0.35,0.5,0x58a6ff);b.position.y=0.3;agent.add(b);const cab=box(0.45,0.25,0.42,0xcfe3ff);cab.position.set(-0.05,0.58,0);agent.add(cab);
      for(const[a,c]of[[0.3,0.27],[0.3,-0.27],[-0.3,0.27],[-0.3,-0.27]]){const w=cyl(0.13,0.1,0x111111);w.rotation.x=Math.PI/2;w.position.set(a,0.13,c);agent.add(w);}}
    else if(kind==='rover'){agent=new THREE.Group();const b=box(0.8,0.3,0.6,0xd29922);b.position.y=0.35;agent.add(b);const m=cyl(0.04,0.6,0xcccccc);m.position.set(-0.2,0.8,0);agent.add(m);const h=box(0.15,0.12,0.25,0xffffff);h.position.set(-0.2,1.1,0);agent.add(h);
      for(const a of[0.3,0,-0.3])for(const c of[0.34,-0.34]){const w=cyl(0.12,0.08,0x222222);w.rotation.x=Math.PI/2;w.position.set(a,0.12,c);agent.add(w);}}
    else if(kind==='boat'){agent=new THREE.Group();const hull=new THREE.Mesh(new THREE.ConeGeometry(0.32,1.1,16),M(0xffffff));hull.rotation.z=-Math.PI/2;hull.position.y=0.2;hull.scale.set(1,1,0.6);agent.add(hull);const cab=box(0.3,0.25,0.3,0x58a6ff);cab.position.set(-0.1,0.42,0);agent.add(cab);}
    else{agent=new THREE.Mesh(new THREE.SphereGeometry(0.35,24,16),M(0x58a6ff));agent.position.y=0.35;const a=new THREE.Group();a.add(agent);agent=a;}
    cam.position.copy(P(cx,lo[1]-0.55*(hi[1]-lo[1]),13));ctl.target.copy(P(cx,cy,0));
  } else if(kind==='cart'){
    SC=16/(hi[0]-lo[0]);const track=box((hi[0]-lo[0])*SC,0.1,1.2,0x30363d);track.position.set(((lo[0]+hi[0])/2)*SC,0,0);W.add(track);
    const wall=box(0.3,3,2.4,0xbf3333);wall.position.set(-0.15,1.5,0);W.add(wall);
    const g=e.gap;if(g){const z=box((g.hi[0]-g.lo[0])*SC,0.02,1.25,0x58a6ff,{transparent:true,opacity:0.35});z.position.set((g.lo[0]+g.hi[0])/2*SC,0.07,0);W.add(z);}
    const strip=new THREE.Mesh(new THREE.PlaneGeometry((hi[0]-lo[0])*SC,6),new THREE.MeshBasicMaterial({map:mapTexture(),transparent:true,opacity:0.85}));
    strip.position.set((lo[0]+hi[0])/2*SC,3,-3);W.add(strip);dyn.push(()=>{strip.material.map.needsUpdate=true;});
    agent=new THREE.Group();const b=box(1.0,0.5,0.8,0x58a6ff);b.position.y=0.4;agent.add(b);for(const a of[0.3,-0.3])for(const c of[0.42,-0.42]){const w=cyl(0.15,0.08,0x111111);w.rotation.x=Math.PI/2;w.position.set(a,0.15,c);agent.add(w);}
    cam.position.set((lo[0]+hi[0])/2*SC,7,14);ctl.target.set((lo[0]+hi[0])/2*SC,1.5,0);
  } else if(kind==='drone'){
    SC=10/(pr.ceiling-pr.floor+0.6);const fl=box(12,0.1,8,0x3b4a2f);fl.position.y=-0.05;W.add(fl);
    const ce=box(12,0.15,8,0xbf3333,{transparent:true,opacity:0.6});ce.position.y=pr.ceiling*SC+0.075;W.add(ce);
    const wash=box(12,(pr.wash_hi-pr.wash_lo)*SC,8,0x58a6ff,{transparent:true,opacity:0.12,depthWrite:false});wash.position.y=(pr.wash_lo+pr.wash_hi)/2*SC;W.add(wash);
    const parts=[];for(let i=0;i<60;i++){const p=new THREE.Mesh(new THREE.SphereGeometry(0.03,6,4),new THREE.MeshBasicMaterial({color:0x9fd4ff}));p.userData={y:pr.wash_lo+Math.random()*(pr.wash_hi-pr.wash_lo)};p.position.set(-5+Math.random()*10,0,-3+Math.random()*6);W.add(p);parts.push(p);}
    dyn.push(dt=>parts.forEach(p=>{p.userData.y-=dt*1.2;if(p.userData.y<pr.wash_lo)p.userData.y=pr.wash_hi;p.position.y=p.userData.y*SC;}));
    const strip=new THREE.Mesh(new THREE.PlaneGeometry(6,(hi[0]-lo[0])*SC),new THREE.MeshBasicMaterial({map:mapTexture(),transparent:true,opacity:0.85}));
    strip.rotation.z=Math.PI/2;strip.position.set(4.5,(lo[0]+hi[0])/2*SC,-3.9);W.add(strip);dyn.push(()=>{strip.material.map.needsUpdate=true;});
    agent=new THREE.Group();const body=box(0.7,0.18,0.7,0x58a6ff);agent.add(body);const rot=[];
    for(const[a,c]of[[0.45,0.45],[0.45,-0.45],[-0.45,0.45],[-0.45,-0.45]]){const arm=box(0.08,0.05,0.08,0x888888);arm.position.set(a/2,0,c/2);agent.add(arm);const r=new THREE.Mesh(new THREE.CylinderGeometry(0.25,0.25,0.02,16),M(0xcccccc,{transparent:true,opacity:0.6}));r.position.set(a,0.1,c);agent.add(r);rot.push(r);}
    dyn.push(dt=>rot.forEach(r=>r.rotation.y+=dt*40));
    cam.position.set(0,pr.ceiling*SC*0.55,13);ctl.target.set(0,pr.ceiling*SC*0.45,0);
  } else if(kind==='pendulum'){
    SC=1;const base=box(2,0.4,1.2,0x30363d);base.position.y=-0.2;W.add(base);const L=4;
    const wedge=(a0,a1,c,op)=>{const g=new THREE.CircleGeometry(L+0.6,32,Math.PI/2-a1,a1-a0);const m=new THREE.Mesh(g,new THREE.MeshBasicMaterial({color:c,transparent:true,opacity:op,side:THREE.DoubleSide}));m.position.z=-0.3;W.add(m);};
    wedge(pr.theta_max,Math.PI/2,0xbf3333,0.35);wedge(-Math.PI/2,-pr.theta_max,0xbf3333,0.35);wedge(pr.heavy_lo,pr.theta_max,0x58a6ff,0.25);
    const strip=new THREE.Mesh(new THREE.PlaneGeometry(5,5),new THREE.MeshBasicMaterial({map:mapTexture(),transparent:true,opacity:0.85}));strip.position.set(-7,2.5,-0.5);W.add(strip);dyn.push(()=>{strip.material.map.needsUpdate=true;});
    agent=new THREE.Group();const rod=box(0.12,L,0.12,0xcfd8e3);rod.position.y=L/2;agent.add(rod);const bob=new THREE.Mesh(new THREE.SphereGeometry(0.35,24,16),M(0x58a6ff));bob.position.y=L;agent.add(bob);
    const pay=box(0.4,0.4,0.4,0xd29922);pay.position.set(0.35,L*0.8,0);agent.add(pay);const piv=cyl(0.18,0.3,0x888888);piv.rotation.x=Math.PI/2;agent.add(piv);
    cam.position.set(-1.5,3,12);ctl.target.set(-1.5,2.2,0);
  }
  ring=new THREE.Mesh(new THREE.RingGeometry(0.55,0.72,32),new THREE.MeshBasicMaterial({color:0x2ea043,side:THREE.DoubleSide,transparent:true,opacity:0.9}));ring.rotation.x=-Math.PI/2;
  W.add(agent);W.add(ring);
  trail=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0xffffff,transparent:true,opacity:0.6}));W.add(trail);
  goalM=null;const g=S.doc.goal;if(g&&['car','rover','boat','point'].includes(kind)){goalM=new THREE.Mesh(new THREE.TorusGeometry(0.5,0.06,8,32),M(0x2ea043,{emissive:0x2ea043}));goalM.rotation.x=Math.PI/2;goalM.position.copy(P(g[0],kind==='point'?0:g[1],0.1));W.add(goalM);}
  else if(g&&kind==='cart'){goalM=box(0.08,0.02,1.25,0x2ea043);goalM.position.set(g[0]*SC,0.08,0);W.add(goalM);}
  else if(g&&kind==='drone'){goalM=box(12,0.02,0.02,0x2ea043);goalM.position.y=g[0]*SC;W.add(goalM);}
  spark=new THREE.Points(new THREE.BufferGeometry(),new THREE.PointsMaterial({color:0xff7b54,size:0.25}));spark.userData={v:[],t:0};W.add(spark);
}
function agentPose(x){const k=S.doc.render;
  if(k==='car'||k==='rover'){agent.position.copy(P(x[0],x[1]));agent.rotation.y=x[2];}
  else if(k==='boat'){agent.position.copy(P(x[0],x[1]));const u=ST.lastU||[0,0];if(Math.hypot(u[0],u[1])>0.05)agent.rotation.y=Math.atan2(u[1],u[0]);}
  else if(k==='point'){agent.position.copy(P(x[0],0));}
  else if(k==='cart'){agent.position.set(x[0]*SC,0,0);}
  else if(k==='drone'){agent.position.set(0,x[0]*SC,0);agent.rotation.z=-0.15*(ST.lastU?ST.lastU[0]:0)*0;}
  else if(k==='pendulum'){agent.rotation.z=-x[0];}
  const p=agent.position;ring.position.set(p.x,(k==='drone'?p.y-0.3:k==='pendulum'?-0.15:0.05),p.z);if(k==='drone'||k==='pendulum')ring.rotation.x=-Math.PI/2;}
function worldPt(x){const k=S.doc.render;if(k==='car'||k==='rover'||k==='boat')return P(x[0],x[1],0.08);if(k==='point')return P(x[0],0,0.08);
  if(k==='cart')return new THREE.Vector3(x[0]*SC,0.1,0);if(k==='drone')return new THREE.Vector3(0,x[0]*SC,0);const L=4;return new THREE.Vector3(L*Math.sin(x[0]),L*Math.cos(x[0]),0);}
function burst(x){const p=worldPt(x),pos=[],v=[];for(let i=0;i<80;i++){pos.push(p.x,p.y+0.3,p.z);v.push([(Math.random()-0.5)*6,Math.random()*5,(Math.random()-0.5)*6]);}
  spark.geometry.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));spark.userData={v,t:1.2};}

// ------------------------------------------------------------------ mini-map
const mc=$('map'),mx=mc.getContext('2d'),MN=160;let mapKey=null;
function mapDims(){const D=S.D;return D===1?[0,null]:[0,1];}
function drawMap(force){const e=S.env,D=S.D,th=D===3?ST.x[2]:null,dth=D===3?e.axes[2][1]:1;
  const key=ST.ci+'|'+(D===3?Math.round(th/dth):0);if(!force&&key===mapKey)return;mapKey=key;
  const img=mx.createImageData(MN,MN),V=S.cases[ST.ci].V,x=new Array(D).fill(0);if(D===3)x[2]=th;
  for(let j=0;j<MN;j++)for(let i=0;i<MN;i++){x[0]=e.lo[0]+(e.hi[0]-e.lo[0])*(i+0.5)/MN;x[1]=e.hi[1]-(e.hi[1]-e.lo[1])*(j+0.5)/MN;
    const vt=S.Vtrue(x),vh=V(x),l=S.l(x);let c;
    if(l<0)c=[0x6e,0x1b,0x1b];else if(vt>=0&&vh>=0)c=[0x2e,0xa0,0x43];else if(vt<0&&vh>=0)c=[0xff,0x2b,0xd6];else if(vt>=0)c=[0xd2,0x99,0x22];else c=[0x30,0x36,0x3d];
    const o=4*(j*MN+i);img.data[o]=c[0];img.data[o+1]=c[1];img.data[o+2]=c[2];img.data[o+3]=255;}
  mx.putImageData(img,0,0);const g=e.gap;if(g){mx.strokeStyle='#58a6ff';mx.setLineDash([4,3]);mx.lineWidth=1.5;
    const X=v=>(v-e.lo[0])/(e.hi[0]-e.lo[0])*MN,Y=v=>(e.hi[1]-v)/(e.hi[1]-e.lo[1])*MN;mx.strokeRect(X(g.lo[0]),Y(g.hi[1]),X(g.hi[0])-X(g.lo[0]),Y(g.lo[1])-Y(g.hi[1]));mx.setLineDash([]);}
  mapBase=mx.getImageData(0,0,MN,MN);
  $('mapcap').textContent=`axes: ${e.state_names[0]} → , ${e.state_names[1]} ↑`+(D===3?` · slice at ${e.state_names[2]} = ${th.toFixed(2)}`:'');}
let mapBase=null;
function drawMapAgent(){if(!mapBase)return;mx.putImageData(mapBase,0,0);const e=S.env,X=v=>(v-e.lo[0])/(e.hi[0]-e.lo[0])*MN,Y=v=>(e.hi[1]-v)/(e.hi[1]-e.lo[1])*MN;
  mx.strokeStyle='#ffffffb0';mx.lineWidth=1;mx.beginPath();const tr=(ST.replay?ST.replay.tr:ST.trace);tr.forEach((s,i)=>{i?mx.lineTo(X(s[0]),Y(s[1])):mx.moveTo(X(s[0]),Y(s[1]));});mx.stroke();
  const x=ST.x;mx.fillStyle='#fff';mx.beginPath();mx.arc(X(x[0]),Y(x[1]),3.5,0,7);mx.fill();mx.strokeStyle='#000';mx.stroke();
  if(S.doc.goal){const g=S.doc.goal;mx.strokeStyle='#7ee787';mx.beginPath();mx.arc(X(g[0]),Y(g[1]!=null?g[1]:0),4,0,7);mx.stroke();}}

// ------------------------------------------------------------------ UI
function opt(sel,items){sel.innerHTML='';items.forEach(([v,t])=>{const o=document.createElement('option');o.value=v;o.textContent=t;sel.appendChild(o);});}
opt($('scn'),DOCS.map((d,i)=>[i,(d.user_defined?'[user] ':'')+d.title]));
const POLS=[['manual','manual (you drive: keys / gamepad / touch)'],...Object.keys(G.POLICIES).map(k=>[k,G.POLICIES[k].label])];
opt($('pol'),POLS);opt($('bPol'),Object.keys(G.POLICIES).map(k=>[k,k]));
function log(msg,cls){ST.events.unshift([msg,cls]);ST.events=ST.events.slice(0,7);$('log').innerHTML=ST.events.map(([m,c])=>`<div class="${c||''}">${m}</div>`).join('');}
function loadScenario(i){ST.si=i;S=G.buildScenario(DOCS[i]);ST.ci=0;$('scn').value=i;
  opt($('cert'),S.cases.map((c,k)=>[k,c.label]));const d=S.doc,e=S.env;
  $('scnInfo').innerHTML=`${d.blurb}<br><span class="tag ${d.user_defined?'user':'real'}">${d.user_defined?'user-defined env':'built-in env'}</span><span class="tag">${e.name}</span><span class="tag">${S.D}D · grid ${e.shape.join('×')}</span><span class="tag">${S.dyn==='js-mirror'?'JS dynamics mirror':'tabulated dynamics'}</span><br><b>Hazard:</b> ${e.hazard}<br><b>Hidden physics (gap):</b> ${e.gap?e.gap.description:'none'}<br><b>Reduction:</b> ${d.reduction||'—'}`;
  $('prov').innerHTML=`${d.provenance.source}. created ${d.provenance.created}, git ${String(d.provenance.git_commit).slice(0,10)}. HJ horizon ${e.horizon}s; shield margin ${d.filter_margin}; dt ${d.dt}s. Quantisation: int16, max error ${S.quantErr.toExponential(2)} in V.`;
  buildScene();resize();ST.x=null;setShield(true,true);setCase(0,true);reset(d.start);resetCounts();$('batchres').innerHTML='';$('batchNote').textContent='';log(`Scenario: ${d.title}`,'ok');}
function resetCounts(){ST.score=0;ST.crashes=0;ST.episodes=0;}
// Counters are per (scenario, certifier, shield) setting: switching certifier or shield restarts the
// episode and the counters, so a crash is never attributed to a certifier it did not happen under.
function tallyKey(){return `${S.doc.name}|${ST.ci}|${ST.shield?'on':'off'}`;}
function setCase(k,quiet){const ch=k!==ST.ci;ST.ci=k;$('cert').value=k;if(ch&&!quiet&&ST.x){reset();resetCounts();}const m=S.cases[k].metrics;
  $('met').innerHTML=`<tr><td>FSV (false-safe volume)</td><td><b style="color:#ff2bd6">${pct(m.fsv)}</b></td></tr><tr><td>FUV (false-unsafe)</td><td>${pct(m.fuv)}</td></tr><tr><td>gap-FSV</td><td>${pct(m.gap_fsv)}</td></tr><tr><td colspan=2 class="src">Full-grid metrics computed in Python (lcert.metrics), not in the browser.</td></tr>`;
  mapKey=null;if(ST.x)drawMap(true);}
function reset(x0){ST.replay=null;ST.x=(x0||S.doc.start||S.env.lo.map((a,i)=>(a+S.env.hi[i])/2)).slice();ST.t=0;ST.step=0;ST.trace=[ST.x.slice()];ST.flags=[];ST.interv=0;ST.fsTime=0;ST.crashT=0;
  polFn=ST.pol==='manual'?null:G.POLICIES[ST.pol].make(S,(Math.random()*1e9)|0);mapKey=null;drawMap(true);}
function randomStart(){const r=G.mulberry32((Math.random()*1e9)|0);const x=G.sampleStart(S,r,S.cases[ST.ci].V,S.doc.filter_margin);if(x){reset(x);log('Random start where the certifier claims safe','ok');}}
function setShield(on,quiet){const ch=on!==ST.shield;ST.shield=on;if(ch&&!quiet&&ST.x&&!ST.replay){reset();resetCounts();}const b=$('shieldBtn');b.classList.toggle('on',on);b.textContent=`Shield ${on?'ON':'OFF'} (space)`;if(!quiet)log(`Shield ${on?'ON':'OFF'} — episode and counters restarted`,'sh');}
function setPol(p){ST.pol=p;$('pol').value=p;polFn=p==='manual'?null:G.POLICIES[p].make(S,(Math.random()*1e9)|0);log(`Driver: ${p}`,'ok');}
$('scn').onchange=e=>loadScenario(+e.target.value);$('cert').onchange=e=>{setCase(+e.target.value);log('Certifier: '+S.cases[ST.ci].label,'ok');};
$('pol').onchange=e=>setPol(e.target.value);$('shieldBtn').onclick=()=>setShield(!ST.shield);$('resetBtn').onclick=()=>reset();$('randStart').onclick=randomStart;
$('pauseBtn').onclick=()=>{ST.paused=!ST.paused;$('pauseBtn').classList.toggle('on',ST.paused);};$('replayBtn').onclick=startReplay;
$('helpLink').onclick=e=>{e.preventDefault();$('help').classList.add('show');};$('helpClose').onclick=()=>$('help').classList.remove('show');
$('batchBtn').onclick=runBatch;
const KEYMAP={ArrowLeft:'left',KeyA:'left',ArrowRight:'right',KeyD:'right',ArrowUp:'up',KeyW:'up',ArrowDown:'down',KeyS:'down'};
addEventListener('keydown',e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT'&&!KEYMAP[e.code])return;
  if(KEYMAP[e.code]){ST.keys[KEYMAP[e.code]]=true;e.preventDefault();return;}
  if(e.repeat)return;const c=e.code;
  if(c==='Space'){setShield(!ST.shield);e.preventDefault();}else if(c==='KeyR')reset();else if(c==='KeyT')randomStart();else if(c==='KeyP')$('pauseBtn').onclick();
  else if(c==='KeyL')startReplay();else if(c==='KeyN')loadScenario((ST.si+(e.shiftKey?DOCS.length-1:1))%DOCS.length);else if(c==='KeyC'){setCase((ST.ci+1)%S.cases.length);log('Certifier: '+S.cases[ST.ci].label,'ok');}
  else if(c==='KeyM'){const ks=POLS.map(p=>p[0]);setPol(ks[(ks.indexOf(ST.pol)+1)%ks.length]);}else if(c==='KeyB')runBatch();else if(c==='KeyH'||e.key==='?')$('help').classList.toggle('show');else if(c==='Escape')$('help').classList.remove('show');});
addEventListener('keyup',e=>{if(KEYMAP[e.code])ST.keys[KEYMAP[e.code]]=false;});
document.querySelectorAll('#touch button').forEach(b=>{const k=b.dataset.k;const on=e=>{e.preventDefault();ST.touch[k]=true;},off=e=>{e.preventDefault();ST.touch[k]=false;};
  b.addEventListener('pointerdown',on);b.addEventListener('pointerup',off);b.addEventListener('pointerleave',off);b.addEventListener('pointercancel',off);});
function axes(){const k=ST.keys,t=ST.touch;let ax=(k.right||t.right?1:0)-(k.left||t.left?1:0),ay=(k.up||t.up?1:0)-(k.down||t.down?1:0);
  const gp=navigator.getGamepads?[...navigator.getGamepads()].find(g=>g):null;
  if(gp){if(Math.abs(gp.axes[0])>0.15)ax=gp.axes[0];if(Math.abs(gp.axes[1])>0.15)ay=-gp.axes[1];if(gp.buttons[14]&&gp.buttons[14].pressed)ax=-1;if(gp.buttons[15]&&gp.buttons[15].pressed)ax=1;
    if(gp.buttons[12]&&gp.buttons[12].pressed)ay=1;if(gp.buttons[13]&&gp.buttons[13].pressed)ay=-1;const a=gp.buttons[0]&&gp.buttons[0].pressed;if(a&&!ST.gpA)setShield(!ST.shield);ST.gpA=a;}
  return{x:ax,y:ay};}
function manualU(){const a=axes(),u=S.umin.map(()=>0);for(const[axn,spec]of Object.entries(S.doc.manual||{x:0})){const[d,sg]=Array.isArray(spec)?spec:[spec,1];
    const v=sg*a[axn];u[d]=v>=0?v*S.umax[d]:-v*S.umin[d];}
  if(u.length>1){const m=Math.max(...S.ctrls.map(c=>Math.hypot(...c))),n=Math.hypot(...u);if(n>m)for(let i=0;i<u.length;i++)u[i]*=m/n;}return u;}

// ------------------------------------------------------------------ sim loop
function endEpisode(crashed){ST.episodes++;ST.last={tr:ST.trace.slice(),flags:ST.flags.slice(),crashed,case:ST.ci,shield:ST.shield};
  if(crashed){ST.crashes++;ST.score-=100;}
  const tk=tallyKey(),T=(ST.tally=ST.tally||{});T[tk]=T[tk]||{episodes:0,crashes:0};T[tk].episodes++;if(crashed)T[tk].crashes++;
  if($('autoReset').checked)setTimeout(()=>{if(!ST.replay)reset();},crashed?1300:300);else ST.paused=true;}
function simStep(){const dt=S.doc.dt,V=S.cases[ST.ci].V,x=ST.x;
  let u=ST.pol==='manual'?manualU():polFn(x,ST.t,dt),active=false;
  if(ST.shield){const r=G.shield(S,V,x,u,dt,S.doc.filter_margin);if(r.active&&!ST.wasActive)log(`t=${ST.t.toFixed(2)}s shield override`,'sh');u=r.u;active=r.active;}
  ST.wasActive=active;ST.lastU=u;ST.active=active;const vt=S.Vtrue(x),vh=V(x),fs=vh>=0&&vt<0;
  if(fs&&!ST.wasFS)log(`t=${ST.t.toFixed(2)}s entered FALSE-SAFE region (V̂=${vh.toFixed(3)}, V_true=${vt.toFixed(3)})`,'fs');ST.wasFS=fs;
  if(active)ST.interv++;if(fs)ST.fsTime+=dt;
  ST.x=G.step(S,x,u,dt);ST.t+=dt;ST.step++;ST.trace.push(ST.x.slice());ST.flags.push([active,fs]);if(ST.trace.length>2000)ST.trace.shift();
  ST.score+=10*dt;if(S.doc.goal&&goalDist(ST.x)<0.25)ST.score+=20*dt;
  if(S.l(ST.x)<0){ST.crashT=1.3;burst(ST.x);log(`t=${ST.t.toFixed(2)}s CRASH (${S.env.hazard})${fs||ST.fsTime>0?' — after trusting a false-safe certificate':''}${ST.shield?' with shield ON':''}`,'crash');endEpisode(true);return false;}
  if(ST.step>=S.doc.max_steps){log(`episode survived ${ST.t.toFixed(1)}s`,'ok');endEpisode(false);return false;}
  return true;}
function goalDist(x){const g=S.doc.goal,w=S.doc.goal_weights||g.map(()=>1);let c=0,sw=0;g.forEach((gi,i)=>{if(gi==null)return;c+=w[i]*(x[i]-gi)**2;sw+=w[i];});return Math.sqrt(c/Math.max(sw,1e-9));}
function startReplay(){if(!ST.last){log('nothing to replay yet','sh');return;}ST.replay={tr:ST.last.tr,flags:ST.last.flags,i:0,crashed:ST.last.crashed};log('Replaying last episode','ok');}

function hud(){const V=S.cases[ST.ci].V,x=ST.x,vt=S.Vtrue(x),vh=V(x),inGap=S.gap(x),fs=vh>=0&&vt<0;
  const sc=Math.max(0.5,...[vt,vh].map(Math.abs));const bar=(v,c)=>`<div class="bar"><i style="background:${c};${v>=0?`left:50%;width:${Math.min(50,50*v/sc)}%`:`left:${50-Math.min(50,-50*v/sc)}%;width:${Math.min(50,-50*v/sc)}%`}"></i></div>`;
  const sh=!ST.shield?'<b style="color:#f85149">OFF</b>':ST.active?'<b style="color:#d29922">INTERVENING</b>':'<b style="color:#2ea043">armed</b>';const m=S.cases[ST.ci].metrics;
  $('hud').innerHTML=`<div class="row"><span class="k">state</span><span>${x.map(v=>v.toFixed(2)).join(', ')}</span></div>
   <div class="row"><span class="k">V_true (HJ, true physics)</span><b style="color:${vt>=0?'#2ea043':'#f85149'}">${fmt(vt)}</b></div>${bar(vt,vt>=0?'#2ea043':'#f85149')}
   <div class="row"><span class="k">V̂ (certifier)</span><b style="color:${vh>=0?'#2ea043':'#f85149'}">${fmt(vh)}</b></div>${bar(vh,vh>=0?'#58a6ff':'#f85149')}
   <div class="row"><span class="k">shield</span><span>${sh}</span></div><div class="row"><span class="k">in hidden-physics gap</span><b style="color:${inGap?'#58a6ff':'#8b949e'}">${inGap?'YES':'no'}</b></div>
   <div class="row"><span class="k">false-safe here</span><b style="color:${fs?'#ff2bd6':'#8b949e'}">${fs?'YES':'no'}</b></div>
   <div class="row"><span class="k">case FSV / gap-FSV</span><span>${pct(m.fsv)} / ${pct(m.gap_fsv)}</span></div>
   <div class="row"><span class="k">driver · t</span><span>${ST.replay?'REPLAY':ST.pol} · ${ST.t.toFixed(1)}s</span></div>`;
  $('score').innerHTML=`<tr><td>score (in-browser)</td><td><b>${ST.score.toFixed(0)}</b></td></tr><tr><td>episodes / crashes (this certifier, shield ${ST.shield?'ON':'OFF'})</td><td>${ST.episodes} / ${ST.crashes}</td></tr><tr><td>shield overrides (this ep.)</td><td>${ST.interv}</td></tr><tr><td>time in false-safe (this ep.)</td><td>${ST.fsTime.toFixed(2)} s</td></tr>`;
  const bn=$('banner');let cls='',txt='';if(ST.crashT>0){cls='crash';txt='💥 CRASH';}else if(ST.replay){cls='replay';txt='REPLAY of last episode';}else if(fs){cls='fs';txt='FALSE-SAFE: certifier says safe, truth says a crash is unavoidable';}else if(ST.active){cls='shield';txt='Shield overriding your input';}
  bn.className=cls;bn.textContent=txt;
  ring.material.color.set(ST.crashT>0?0xf85149:fs?0xff2bd6:ST.active?0xd29922:ST.shield?0x2ea043:0x8b949e);
  window.__gym.readout={v_true:vt,v_hat:vh,in_gap:inGap,false_safe:fs,shield:ST.shield,active:!!ST.active,x:x.slice(),t:ST.t,crashes:ST.crashes,episodes:ST.episodes,score:ST.score,scenario:S.doc.name,case:ST.ci};}

let last=performance.now(),acc=0;
let hudT=0;
function frame(now){requestAnimationFrame(frame);if(document.hidden){last=now;return;}let dt=Math.min(0.1,(now-last)/1000);last=now;
  if(!ST.paused&&!$('help').classList.contains('show')||ST.forceRun){
    if(ST.replay){const r=ST.replay;r.i+=Math.max(1,Math.round(dt/S.doc.dt));if(r.i>=r.tr.length){ST.replay=null;reset();}else{ST.x=r.tr[r.i].slice();const f=r.flags[Math.min(r.i,r.flags.length-1)]||[false,false];ST.active=f[0];}}
    else if(ST.crashT<=0){acc+=dt;let n=0;while(acc>=S.doc.dt&&n<10){acc-=S.doc.dt;n++;if(!simStep()){acc=0;break;}}}}
  if(ST.crashT>0)ST.crashT-=dt;
  if(spark.userData.t>0){spark.userData.t-=dt;const a=spark.geometry.attributes.position;if(a){for(let i=0;i<a.count;i++){const v=spark.userData.v[i];v[1]-=9*dt;a.setXYZ(i,a.getX(i)+v[0]*dt,Math.max(0,a.getY(i)+v[1]*dt),a.getZ(i)+v[2]*dt);}a.needsUpdate=true;}spark.visible=true;}else spark.visible=false;
  agentPose(ST.x);if(S.D===3)drawMap(false);drawMapAgent();
  const tr=(ST.replay?ST.replay.tr.slice(0,ST.replay.i+1):ST.trace).slice(-600).map(worldPt);trail.geometry.setFromPoints(tr);
  dyn.forEach(f=>f(dt));if(['car','rover','boat','point'].includes(S.doc.render)){const p=agent.position;ctl.target.lerp(new THREE.Vector3(p.x*0.6,0,p.z*0.6),0.03);}
  ctl.update();if(now-hudT>66){hudT=now;hud();}R.render(scene,cam);}

// ------------------------------------------------------------------ batch
function runBatch(){const N=Math.max(1,Math.min(1000,+$('bN').value||50)),pk=$('bPol').value,ci=ST.ci,seed=1+((Math.random()*1e6)|0);
  const pb=$('bProg');pb.style.display='block';pb.value=0;$('batchBtn').disabled=true;window.__gym.batchDone=false;
  G.batch(S,ci,pk,N,seed,(i,n)=>{pb.value=i/n;},r=>{pb.style.display='none';$('batchBtn').disabled=false;window.__gym.batch=r;window.__gym.batchDone=true;
    const row=(k,a,b)=>`<tr><td>${k}</td><td>${a}</td><td>${b}</td></tr>`;
    $('batchres').innerHTML=`<tr><td></td><td>shield OFF</td><td>shield ON</td></tr>`+row('crash rate',pct(r.off.rate),pct(r.on.rate))+row('crashes / episodes',`${r.off.crashes}/${r.off.n}`,`${r.on.crashes}/${r.on.n}`)+row('override rate (steps)','—',pct(r.on.interventions))+
      `<tr><td colspan=3>starts in false-safe region: <b>${r.startsFalseSafe}</b>; shield-ON crashes from those: <b>${r.crashOnStartFalseSafe}</b>, from truly-safe starts: <b>${r.crashOnStartTrulySafe}</b></td></tr>`;
    $('batchNote').innerHTML=`<span class="tag expl">in-browser simulation</span> ${S.doc.title} · certifier “${S.cases[ci].label}” · policy ${pk} · N=${N} seeded starts (seed ${seed}) sampled in the start box where V̂ ≥ margin, same starts for OFF/ON, ≤ ${S.doc.max_steps} steps of ${S.doc.dt}s, true dynamics via ${S.dyn==='js-mirror'?'JS mirror of the Python env (cross-checked in tests/test_gym_js_parity.py)':'grid-tabulated dynamics'}, value lookups from exported grids. Exploratory, not a preregistered result.`;
    log(`Batch done: crash rate OFF ${pct(r.off.rate)} vs ON ${pct(r.on.rate)}`,'ok');});}

window.__gym={ready:false,ST,get S(){return S;},loadScenario,setCase,setPol,setShield,reset,runBatch,DOCS};
if(!DOCS.length){document.body.innerHTML='<p style="padding:20px">No scenarios found in data/. Run <code>python -m lcert.framework.gym</code>.</p>';return;}
const qs=new URLSearchParams(location.search);const qi=DOCS.findIndex(d=>d.name===qs.get('scenario'));
loadScenario(qi>=0?qi:0);if(!qs.has('nohelp')&&!localStorage.getItem('fsbGymSeen')){$('help').classList.add('show');localStorage.setItem('fsbGymSeen','1');}
requestAnimationFrame(frame);window.__gym.ready=true;
})();
