// Helper for tests/test_gym_js_parity.py: reads a JSON job on stdin, runs gym_core.js, prints JSON.
const path=require('path'),fs=require('fs');
const G=require(path.join(__dirname,'..','demos','gym','gym_core.js'));
const job=JSON.parse(fs.readFileSync(0,'utf8'));global.window={};
for(const f of job.files)require(path.resolve(f));
const docs=window.FSB_GYM,out=[];
for(const r of job.runs){const doc=docs.find(d=>d.name===r.scenario),S=G.buildScenario(doc),V=S.cases[r.case].V,dt=doc.dt;
  if(r.kind==='interp'){out.push({vt:r.points.map(p=>S.Vtrue(p)),vh:r.points.map(p=>V(p)),l:r.points.map(p=>S.l(p))});continue;}
  if(r.kind==='f'){out.push({ft:r.points.map(p=>S.ft(p,r.u)),fn:r.points.map(p=>S.fn(p,r.u))});continue;}
  let x=r.x0.slice();const xs=[x.slice()],act=[];let hit=null;
  for(let i=0;i<r.us.length;i++){let u=r.us[i],a=false;if(r.shield){const s=G.shield(S,V,x,u,dt,doc.filter_margin);u=s.u;a=s.active;}
    x=G.step(S,x,u,dt);xs.push(x.slice());act.push(a);if(S.l(x)<0){hit=i+1;break;}}
  out.push({states:xs,active:act,hit,dyn:S.dyn});}
process.stdout.write(JSON.stringify(out));
