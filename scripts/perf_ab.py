"""A/B render-cost check for two built site/ directories, measured in the SAME session with the
two builds interleaved (A B A B ...), so machine load affects both equally.  Absolute numbers are
meaningless on a loaded box / software GL; read only the B-vs-A ratio.

    python scripts/perf_ab.py --a /tmp/site_pre --b site [--runs 3] [--ms 4000]
Per scenario reports main-thread busy time per wall second (CDP TaskDuration), and rAF p95.
"""
import argparse, functools, http.server, json, statistics, threading
from playwright.sync_api import sync_playwright

ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']
RAF = """ms=>new Promise(r=>{const t=[];let p=performance.now();const t0=p;const f=n=>{t.push(n-p);p=n;
 if(n-t0<ms)requestAnimationFrame(f);else{t.shift();t.sort((a,b)=>a-b);r({p95:t[Math.floor(t.length*.95)]||0,n:t.length});}};requestAnimationFrame(f);})"""
TOUR = "try{for(const k of ['viewer3d','gym','landing','index'])localStorage.setItem('fsb-tour-'+k,'done');localStorage.setItem('fsbGymSeen','1')}catch(e){}"
SCEN = {  # name: (path, setup js)
    'viewer3d autoplay': ('viewer3d/index.html', ''),
    'viewer3d world-model panel open': ('viewer3d/index.html', "document.getElementById('wmBtn').click()"),
    'gym 30-s demo driving': ('gym/index.html?nohelp=1', "window.__gym.startDemo&&window.__gym.startDemo()"),
    'gym paused': ('gym/index.html?nohelp=1', "document.getElementById('pauseBtn').click()"),
    'gym help open': ('gym/index.html?nohelp=1', "document.getElementById('help').classList.add('show')"),
    'gym compare open': ('gym/index.html?nohelp=1', "window.__gym.openCompare()"),
    'gym world-model panel open': ('gym/index.html?nohelp=1', "document.getElementById('wmBtn').click()"),
}


def serve(d):
    class Q(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *x): pass
    s = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Q, directory=d))
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return f'http://127.0.0.1:{s.server_address[1]}/'


def measure(b, base, path, js, ms, dpr):
    ctx = b.new_context(viewport={'width': 1280, 'height': 800}, device_scale_factor=dpr)
    pg = ctx.new_page(); pg.add_init_script(TOUR); errs = []
    pg.on('pageerror', lambda e: errs.append(str(e)))
    pg.goto(base + path); pg.wait_for_function('window.__fsb&&window.__fsb.ready||window.__gym&&window.__gym.ready', timeout=60000); pg.wait_for_timeout(1500)
    if js: pg.evaluate(js)
    pg.wait_for_timeout(800)
    cdp = ctx.new_cdp_session(pg); cdp.send('Performance.enable')
    g = lambda: {m['name']: m['value'] for m in cdp.send('Performance.getMetrics')['metrics']}
    m0 = g(); raf = pg.evaluate(RAF, ms); m1 = g()
    busy = (m1['TaskDuration'] - m0['TaskDuration']) / (m1['Timestamp'] - m0['Timestamp'])
    ctx.close()
    return busy, raf['p95'], errs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--a', required=True); ap.add_argument('--b', required=True)
    ap.add_argument('--runs', type=int, default=3); ap.add_argument('--ms', type=int, default=4000); ap.add_argument('--dpr', type=float, default=2); ap.add_argument('--only', default='')
    a = ap.parse_args(); bases = {'A': serve(a.a), 'B': serve(a.b)}; out = {}
    with sync_playwright() as p:
        br = p.chromium.launch(executable_path='/usr/bin/google-chrome', args=ARGS)
        for name, (path, js) in SCEN.items():
            if a.only and a.only not in name: continue
            r = {'A': [], 'B': []}; errs = []
            for i in range(a.runs):
                for k in (('A', 'B') if i % 2 == 0 else ('B', 'A')):
                    busy, p95, e = measure(br, bases[k], path, js, a.ms, a.dpr); r[k].append((busy, p95)); errs += [k + ':' + x for x in e]
            med = lambda k, j: statistics.median(x[j] for x in r[k])
            out[name] = {'busy_A': round(med('A', 0), 3), 'busy_B': round(med('B', 0), 3),
                         'busy_ratio_B/A': round(med('B', 0) / max(1e-6, med('A', 0)), 2),
                         'p95_A_ms': round(med('A', 1), 1), 'p95_B_ms': round(med('B', 1), 1), 'errors': errs}
            print(f"{name:34s} busy A {out[name]['busy_A']:.3f}  B {out[name]['busy_B']:.3f}  B/A {out[name]['busy_ratio_B/A']:.2f}"
                  f"   rAF p95 A {out[name]['p95_A_ms']}  B {out[name]['p95_B_ms']}  {errs or ''}", flush=True)
        br.close()
    print(json.dumps(out))


if __name__ == '__main__':
    main()
