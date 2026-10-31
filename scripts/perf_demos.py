"""Frame-time check for the built viewer3d and gym at device pixel ratio 1 and 2 (headless Chrome,
SwiftShader = software GL, so absolute numbers are pessimistic; compare DPR 1 vs 2).

    python scripts/build_site.py && python scripts/perf_demos.py [--runs 3]
Reports mean / p95 rAF interval over several runs and the pixel ratio the page settled on.
"""
import argparse, functools, http.server, json, statistics, threading
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']
JS = """async ms=>{const t=[];let p=performance.now();await new Promise(r=>{const f=n=>{t.push(n-p);p=n;
 if(t.length*0+n-t0<ms)requestAnimationFrame(f);else r();};const t0=performance.now();requestAnimationFrame(f);});
 t.shift();t.sort((a,b)=>a-b);return {mean:t.reduce((a,b)=>a+b,0)/t.length,p95:t[Math.floor(t.length*.95)],n:t.length,
 pr:(window.FSB_PERF&&window.FSB_PERF.pixelRatio)||null};}"""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', type=int, default=3); ap.add_argument('--ms', type=int, default=4000)
    a = ap.parse_args()
    class Q(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *x): pass
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Q, directory=str(ROOT / 'site')))
    threading.Thread(target=srv.serve_forever, daemon=True).start(); base = f'http://127.0.0.1:{srv.server_address[1]}/'
    out = {}
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path='/usr/bin/google-chrome', args=ARGS)
        for page in ['viewer3d/index.html', 'gym/index.html?nohelp=1']:
            for dpr in (1, 2):
                runs = []
                for _ in range(a.runs):
                    ctx = b.new_context(viewport={'width': 1280, 'height': 800}, device_scale_factor=dpr)
                    pg = ctx.new_page(); pg.add_init_script("try{localStorage.setItem('fsb-tour-viewer3d','done');localStorage.setItem('fsb-tour-gym','done')}catch(e){}")
                    pg.goto(base + page); pg.wait_for_timeout(2500)
                    runs.append(pg.evaluate(JS, a.ms)); ctx.close()
                k = f"{page.split('/')[0]}@dpr{dpr}"
                out[k] = {'mean_ms': round(statistics.mean(r['mean'] for r in runs), 2),
                          'p95_ms': round(statistics.mean(r['p95'] for r in runs), 2),
                          'pixelRatio_final': [r['pr'] for r in runs]}
                print(k, out[k], flush=True)
        b.close()
    print(json.dumps(out))


if __name__ == '__main__':
    main()
