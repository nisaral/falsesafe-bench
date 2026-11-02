import asyncio
from playwright.async_api import async_playwright
OUT='/workspace/falsesafe-bench-shots/3d'
URL='file://'+__import__('os').path.abspath(__import__('os').path.join(__import__('os').path.dirname(__file__),'..','demos'))+'/viewer3d/index.html'
async def main():
  async with async_playwright() as p:
    b=await p.chromium.launch(executable_path='/usr/bin/google-chrome',args=['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
    pg=await b.new_page(viewport={'width':1600,'height':1000});errs=[]
    pg.on('console',lambda m: errs.append(m.text) if m.type=='error' else None);pg.on('pageerror',lambda x:errs.append(str(x)+' '+str(x.stack)[:400]))
    await pg.goto(URL);await pg.wait_for_function('window.__fsb&&window.__fsb.ready');await pg.wait_for_timeout(2500)
    names=await pg.evaluate('window.__fsb.DS.map(d=>d.name)');print('datasets',names)
    async def shot(n): await pg.wait_for_timeout(1800); await pg.screenshot(path=f'{OUT}/{n}.png'); print('saved',n, await pg.evaluate("document.getElementById('met').innerText.replace(/\\n/g,' | ')"))
    for i,n in enumerate(names):
      await pg.select_option('#ds',str(i));await shot(f'{i+1:02d}_{n}_default')
    i=names.index('di_nominal_sweep');await pg.select_option('#ds',str(i))
    for k,v in [('env.b',2),('certifier.k',0)]:
      await pg.evaluate(f"(()=>{{const e=document.getElementById('sw_{k}');e.value={v};e.oninput()}})()")
    await shot('05_di_b0.6_k0_falsesafe')
    await pg.evaluate("(()=>{const e=document.getElementById('sw_certifier.k');e.value=3;e.oninput()})()");await shot('06_di_b0.6_k3')
    await pg.select_option('#tr','1');await shot('07_di_filtered_traj')
    await pg.click('#views button[data-v=top]');await shot('07b_di_top_view_errorfloor');await pg.click('#views button[data-v=iso]')
    i=names.index('dubins_slippery');await pg.select_option('#ds',str(i))
    await pg.evaluate("(()=>{const e=document.getElementById('sw_env.slip');e.value=1;e.oninput();const k=document.getElementById('sw_certifier.k');k.value=0;k.oninput()})()")
    await pg.select_option('#slicebox select','1');await shot('08_dubins_slip0.6_k0_slice0')
    print('errors:',errs)
    await b.close()
asyncio.run(main())
