/* FalseSafe gym core: data decoding, grid interpolation, JS mirrors of the Python
   framework dynamics, RK4 stepping, the certifier-driven safety filter, policies and
   batch episodes.  Pure JS, no DOM: runs in the browser and in node (tests/test_gym_js_parity.py).
   Every value function used here comes from the exported HJ grids (lcert.framework.gym);
   nothing is synthesised. */
(function(root){
'use strict';
function b64ToBytes(s){
  if(typeof Buffer!=='undefined'&&typeof atob==='undefined'){const b=Buffer.from(s,'base64');return new Uint8Array(b.buffer,b.byteOffset,b.length);}
  const bin=atob(s),u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return u;}
function decode(d){ // int16 LE * scale -> Float64Array (row-major, shape d.shape)
  const by=b64ToBytes(d.b64),n=by.length>>1,dv=new DataView(by.buffer,by.byteOffset,by.length),out=new Float64Array(n);
  for(let i=0;i<n;i++)out[i]=dv.getInt16(2*i,true)*d.scale;return {data:out,shape:d.shape,maxErr:d.max_abs_err};}

// ---------------------------------------------------------------- grid interpolation
// Multilinear interpolation on the hj grid; periodic dims wrap, other dims extrapolate
// linearly from the edge cell (same as scipy RegularGridInterpolator(fill_value=None)
// used by lcert.framework.policies.GridValue).
function makeInterp(env,arr,extraStride){
  const D=env.lo.length,shape=env.shape,per=env.periodic_dims||[],ax=env.axes;
  const st=new Array(D);let s=extraStride||1;for(let d=D-1;d>=0;d--){st[d]=s;s*=shape[d];}
  const data=arr.data;
  return function(x,offset){
    offset=offset||0;const i0=new Array(D),t=new Array(D),i1=new Array(D);
    for(let d=0;d<D;d++){
      let xd=x[d];const n=shape[d],x0=ax[d][0],dx=ax[d][1];
      if(per.indexOf(d)>=0){const L=env.hi[d]-env.lo[d];xd=env.lo[d]+(((xd-env.lo[d])%L)+L)%L;
        let f=(xd-x0)/dx,k=Math.floor(f);if(k>n-1)k=n-1;if(k<0)k=0;i0[d]=k;i1[d]=(k+1)%n;t[d]=f-k;}
      else{let f=(xd-x0)/dx,k=Math.floor(f);if(k>n-2)k=n-2;if(k<0)k=0;i0[d]=k;i1[d]=k+1;t[d]=f-k;}
    }
    let v=0;
    for(let c=0;c<(1<<D);c++){let w=1,idx=offset;
      for(let d=0;d<D;d++){const b=(c>>d)&1;w*=b?t[d]:1-t[d];idx+=(b?i1[d]:i0[d])*st[d];}
      v+=w*data[idx];}
    return v;};
}

// ---------------------------------------------------------------- dynamics mirrors
// Each mirrors the jnp function of the same env in src/lcert/framework/envs.py.
const MIRRORS={
  di_hidden_physics(P){const lo=P.pc-0.5*P.w,hi=P.pc+0.5*P.w;return{
    ft:(x,u)=>{const g=x[0]>=lo&&x[0]<=hi;return[x[1],g?(1-P.eta)*u[0]+P.b*P.s:u[0]];},
    fn:(x,u)=>[x[1],u[0]], l:x=>x[0]};},
  dubins_slippery(P){const ob=P.obstacle||[0,0,0.8],pa=P.patch||[0.8,2,-1.5,1.5];return{
    ft:(x,u)=>{const ins=x[0]>=pa[0]&&x[0]<=pa[1]&&x[1]>=pa[2]&&x[1]<=pa[3];return[P.speed*Math.cos(x[2]),P.speed*Math.sin(x[2]),ins?(1-P.slip)*u[0]:u[0]];},
    fn:(x,u)=>[P.speed*Math.cos(x[2]),P.speed*Math.sin(x[2]),u[0]], l:x=>Math.hypot(x[0]-ob[0],x[1]-ob[1])-ob[2]};},
  drone_vertical(P){return{
    ft:(x,u)=>{const b=x[0]>=P.wash_lo&&x[0]<=P.wash_hi;return[x[1],b?(1-P.loss)*u[0]-P.wash:u[0]];},
    fn:(x,u)=>[x[1],u[0]], l:x=>Math.min(x[0]-P.floor,P.ceiling-x[0])};},
  pendulum_payload(P){return{
    ft:(x,u)=>[x[1],P.g_over_l*Math.sin(x[0])-P.damping*x[1]+(x[0]>=P.heavy_lo?1-P.gain_loss:1)*u[0]],
    fn:(x,u)=>[x[1],P.g_over_l*Math.sin(x[0])-P.damping*x[1]+u[0]], l:x=>P.theta_max-Math.abs(x[0])};},
  boat_current(P){const r=P.rock,b=P.band,c=P.current;return{
    ft:(x,u)=>{const i=x[1]>=b[0]&&x[1]<=b[1];return[u[0]+(i?c[0]:0),u[1]+(i?c[1]:0)];},
    fn:(x,u)=>[u[0],u[1]], l:x=>Math.min(Math.hypot(x[0]-r[0],x[1]-r[1])-r[2],x[1]-P.shore)};},
  rover_cliff3d(P){const p=P.pit;return{
    ft:(x,u)=>[P.speed*Math.cos(x[2]),P.speed*Math.sin(x[2])+(x[1]>=P.slope_lo?P.drift:0),u[0]],
    fn:(x,u)=>[P.speed*Math.cos(x[2]),P.speed*Math.sin(x[2]),u[0]],
    l:x=>Math.min(P.cliff-x[1],Math.hypot(x[0]-p[0],x[1]-p[1])-p[2])};},
};

// Tabulated fallback for user-defined envs: f(x,u) interpolated on the grid; linear in u
// between candidate controls for scalar controls (exact for control-affine systems),
// nearest candidate otherwise.
function tabulated(env,tab,ctrls){
  const T=decode(tab),D=env.lo.length,N=env.shape.reduce((a,b)=>a*b,1),fs=[];
  for(let k=0;k<ctrls.length;k++)for(let d=0;d<D;d++){const I=makeInterp(env,T,D);fs.push([k,d,I]);}
  const evalK=(k,x)=>{const o=new Array(D);for(let d=0;d<D;d++)o[d]=fs[k*D+d][2](x,k*N*D+d);return o;};
  return (x,u)=>{
    if(ctrls[0].length===1){const cs=ctrls.map(c=>c[0]);let j=0;while(j<cs.length-2&&u[0]>cs[j+1])j++;
      const a=evalK(j,x),b=evalK(j+1,x),t=(u[0]-cs[j])/(cs[j+1]-cs[j]);return a.map((v,i)=>v+(b[i]-v)*t);}
    let best=0,bd=1e18;ctrls.forEach((c,k)=>{let s=0;c.forEach((v,i)=>s+=(v-u[i])**2);if(s<bd){bd=s;best=k;}});return evalK(best,x);};
}

function buildScenario(doc){
  const env=doc.env,S={doc,env,D:env.lo.length,ctrls:env.controls};
  S.Vtrue=makeInterp(env,decode(doc.v_true));
  S.cases=doc.cases.map(c=>({label:c.label,metrics:c.metrics,certifier:c.certifier,params:c.params,V:makeInterp(env,decode(c.v_hat))}));
  S.quantErr=Math.max(doc.v_true.max_abs_err,...doc.cases.map(c=>c.v_hat.max_abs_err));
  if(env.js_dynamics&&MIRRORS[env.js_dynamics]){const m=MIRRORS[env.js_dynamics](env.params);S.ft=m.ft;S.fn=m.fn;S.l=m.l;S.dyn='js-mirror';}
  else{S.ft=tabulated(env,doc.f_true_table,env.controls);S.fn=tabulated(env,doc.f_nom_table,env.controls);
    const L=makeInterp(env,decode(doc.hazard_l));S.l=x=>L(x);S.dyn='tabulated';}
  S.umin=[],S.umax=[];const ud=env.controls[0].length;
  for(let i=0;i<ud;i++){S.umin.push(Math.min(...env.controls.map(c=>c[i])));S.umax.push(Math.max(...env.controls.map(c=>c[i])));}
  S.gap=x=>{const g=env.gap;if(!g)return false;for(let d=0;d<x.length;d++)if(x[d]<g.lo[d]||x[d]>g.hi[d])return false;return true;};
  return S;
}

const add=(a,b,s)=>a.map((v,i)=>v+s*b[i]);
function wrap(S,x){const p=S.env.periodic_dims||[];for(const d of p){const L=S.env.hi[d]-S.env.lo[d];x[d]=S.env.lo[d]+(((x[d]-S.env.lo[d])%L)+L)%L;}return x;}
function rk4(f,x,u,dt){const k1=f(x,u),k2=f(add(x,k1,dt/2),u),k3=f(add(x,k2,dt/2),u),k4=f(add(x,k3,dt),u);
  return x.map((v,i)=>v+dt/6*(k1[i]+2*k2[i]+2*k3[i]+k4[i]));}
function step(S,x,u,dt){return wrap(S,rk4(S.ft,x,u,dt));}

// Safety filter == lcert.framework.policies.SafetyFilter: keep the task/user control while
// V_hat(x) >= margin, else pick the candidate control maximising V_hat(x + dt f_nom(x,u)).
function shield(S,V,x,uTask,dt,margin){
  if(V(x)>=margin)return{u:uTask,active:false};
  let best=0,bv=-Infinity;
  S.ctrls.forEach((c,k)=>{const n=add(x,S.fn(x,c),dt);const v=V(n);if(v>bv){bv=v;best=k;}});
  return{u:S.ctrls[best].slice(),active:true};}

// ---------------------------------------------------------------- policies
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function lookahead(S,x,u,H,dt){let y=x.slice();for(let i=0;i<H;i++)y=wrap(S,rk4(S.fn,y,u,dt));return y;}
function goalCost(S,y){const g=S.doc.goal,w=S.doc.goal_weights||g.map(()=>1);let c=0;
  for(let i=0;i<g.length;i++){if(g[i]==null)continue;let e=y[i]-g[i];if((S.env.periodic_dims||[]).indexOf(i)>=0)e=Math.atan2(Math.sin(e),Math.cos(e));c+=w[i]*e*e;}return c;}
const POLICIES={
  goal:{label:'go-to-goal (greedy lookahead on nominal model, ignores hazards)',make:(S)=>(x,t,dt)=>{
    let best=0,bc=Infinity;S.ctrls.forEach((c,k)=>{const y=lookahead(S,x,c,15,dt);const v=goalCost(S,y)+0.3*goalCost(S,lookahead(S,x,c,5,dt));if(v<bc){bc=v;best=k;}});return S.ctrls[best].slice();}},
  random:{label:'random (piecewise-constant, resampled every 0.3 s)',make:(S,seed)=>{const r=mulberry32(seed||1);let u=null,next=0;
    return (x,t,dt)=>{if(u===null||t>=next-1e-9){u=S.umin.map((a,i)=>a+(S.umax[i]-a)*r());next=t+0.3;}return u.slice();};}},
  adversarial:{label:'adversarial (drives toward the hazard: min l(x) after 0.3 s lookahead)',make:(S)=>(x,t,dt)=>{
    let best=0,bc=Infinity;S.ctrls.forEach((c,k)=>{const v=S.l(lookahead(S,x,c,15,dt));if(v<bc){bc=v;best=k;}});return S.ctrls[best].slice();}},
};

function sampleStart(S,rng,V,margin,tries){const b=S.doc.start_box||[S.env.lo,S.env.hi];
  for(let k=0;k<(tries||500);k++){const x=b[0].map((a,i)=>a+(b[1][i]-a)*rng());if(S.l(x)>0&&V(x)>=margin)return x;}return null;}

// One episode under the TRUE dynamics (JS mirror); returns summary + trace.
function episode(S,caseIdx,policy,x0,opts){
  const dt=S.doc.dt,steps=opts.steps||S.doc.max_steps,margin=S.doc.filter_margin,V=S.cases[caseIdx].V;
  let x=x0.slice(),crash=null,inter=0,fsSteps=0,gapSteps=0;const trace=opts.trace?[x.slice()]:null;
  for(let i=0;i<steps;i++){const t=i*dt;let u=policy(x,t,dt),act=false;
    if(opts.shield){const r=shield(S,V,x,u,dt,margin);u=r.u;act=r.active;}
    if(act)inter++;if(V(x)>=0&&S.Vtrue(x)<0)fsSteps++;if(S.gap(x))gapSteps++;
    x=step(S,x,u,dt);if(trace)trace.push(x.slice());
    if(S.l(x)<0){crash=i+1;break;}}
  return{crash,steps:crash||steps,interventions:inter,falseSafeSteps:fsSteps,gapSteps,startFalseSafe:S.Vtrue(x0)<0,trace};}

// Batch: same seeded starts for shield off and on.  Starts are sampled from start_box where
// the certifier claims safety (V_hat >= margin) -- where a deployed shield would let you start.
function batch(S,caseIdx,polKey,N,seed,onProgress,done){
  const rng=mulberry32(seed||7),V=S.cases[caseIdx].V,margin=S.doc.filter_margin,res={off:[],on:[],starts:[]};let i=0;
  function chunk(){const t0=Date.now();
    while(i<N&&Date.now()-t0<30){const x0=sampleStart(S,rng,V,margin);if(!x0){i++;continue;}const ps=seed*1000+i;
      res.starts.push(x0);
      res.off.push(episode(S,caseIdx,POLICIES[polKey].make(S,ps),x0,{shield:false}));
      res.on.push(episode(S,caseIdx,POLICIES[polKey].make(S,ps),x0,{shield:true}));i++;}
    if(onProgress)onProgress(i,N);if(i<N)setTimeout(chunk,0);else done(summarize(res));}
  chunk();}
function summarize(res){const f=a=>{const n=a.length,c=a.filter(e=>e.crash).length;return{n,crashes:c,rate:n?c/n:null,
  interventions:a.reduce((s,e)=>s+e.interventions,0)/Math.max(1,a.reduce((s,e)=>s+e.steps,0)),
  falseSafeSteps:a.reduce((s,e)=>s+e.falseSafeSteps,0)};};
  const doomed=res.on.filter(e=>e.startFalseSafe).length;
  return{off:f(res.off),on:f(res.on),startsFalseSafe:doomed,n:res.starts.length,
    crashOnStartFalseSafe:res.on.filter(e=>e.startFalseSafe&&e.crash).length,
    crashOnStartTrulySafe:res.on.filter(e=>!e.startFalseSafe&&e.crash).length};}

const API={decode,makeInterp,MIRRORS,buildScenario,rk4,step,wrap,shield,POLICIES,episode,batch,summarize,mulberry32,sampleStart};
if(typeof module!=='undefined'&&module.exports)module.exports=API;else root.FSBGym=API;
})(typeof window!=='undefined'?window:globalThis);
