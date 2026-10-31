/* FalseSafe demos: adaptive render quality.  Watches the frame interval of frames that actually
   rendered; if the median over ~1 s is slow it lowers the WebGL pixel ratio a step (never below
   0.75, and below 1 only when very slow); when frames are fast again it steps back up to the cap
   (min(1.5, devicePixelRatio); 1.25 on low-end devices: <=4 cores or <=4 GB).  Exposes window.FSB_PERF for tests. */
(function(){
window.FSBAdaptive=function(renderers,resize){
  const low=/[?&]lowend=1/.test(location.search)||(navigator.hardwareConcurrency||8)<=4||(navigator.deviceMemory||8)<=4;// low-end: cap 1.25
  const dpr=window.devicePixelRatio||1,cap=Math.min(low?1.25:1.5,dpr);let pr=cap,buf=[],cool=0,acc=0;
  const P=window.FSB_PERF=window.FSB_PERF||{};P.pixelRatio=pr;P.cap=cap;P.lowEnd=low;P.changes=0;
  const apply=()=>{renderers.forEach(r=>r.setPixelRatio(pr));P.pixelRatio=pr;P.changes++;resize();};
  apply();P.changes=0;
  return {get pixelRatio(){return pr;},
    sample(ms){if(!(ms>0)||ms>1000)return;buf.push(ms);acc+=ms;if(buf.length<8||(acc<1000&&buf.length<60))return;acc=0;
      const s=buf.slice().sort((a,b)=>a-b),med=s[s.length>>1];buf=[];if(cool>0){cool--;return;}
      let n=pr;if(med>30&&pr>1)n=Math.max(1,pr-0.25);else if(med>50&&pr>0.75)n=0.75;
      else if(med<14&&pr<cap)n=Math.min(cap,pr+0.25);
      if(n!==pr){pr=n;apply();cool=2;}}};};
})();
