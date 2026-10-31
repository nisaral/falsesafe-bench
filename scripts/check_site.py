"""Headless-Chrome smoke check of the BUILT site (site/): every page loads with no console errors,
no failed requests (404s), and a screenshot per page.  Serves site/ on a local port; never deploys.

    python scripts/build_site.py && python scripts/check_site.py [--out DIR] [--fps]
"""
import argparse, functools, http.server, json, os, threading
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']
PAGES = ['index.html', 'glossary.html', '404.html', 'viewer3d/index.html', 'gym/index.html?nohelp=1',
         'studio/index.html', 'scientific_viewer/index.html', 'watch_safety_go_wrong/index.html',
         'one_robot_learns_to_be_safe/index.html']


def serve(port):
    class Q(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
    h = functools.partial(Q, directory=str(ROOT / 'site'))
    s = http.server.ThreadingHTTPServer(('127.0.0.1', port), h)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='/workspace/falsesafe-bench-shots/iter2/site')
    ap.add_argument('--port', type=int, default=0); ap.add_argument('--chrome', default='/usr/bin/google-chrome')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); srv = serve(a.port); base = f'http://127.0.0.1:{srv.server_address[1]}/'
    report, bad = {}, 0
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=a.chrome, args=ARGS)
        for vp in [(1280, 800), (390, 844)]:
            for pg_ in PAGES:
                ctx = b.new_context(viewport={'width': vp[0], 'height': vp[1]}, has_touch=vp[0] < 500)
                pg = ctx.new_page(); errs, fails = [], []
                pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
                pg.on('pageerror', lambda x: errs.append(str(x)))
                pg.on('response', lambda r: fails.append(f'{r.status} {r.url}') if r.status >= 400 and not r.url.endswith('/404-probe') else None)
                pg.on('requestfailed', lambda r: fails.append(f'FAILED {r.url}'))
                pg.goto(base + pg_, wait_until='load'); pg.wait_for_timeout(2500)
                name = pg_.split('?')[0].replace('/', '_').replace('.html', '') + f'_{vp[0]}'
                pg.screenshot(path=f'{a.out}/{name}.png')
                # 404.html is fetched on purpose only when visited directly; ignore github links (offline)
                fails = [f for f in fails if '127.0.0.1' in f]
                report[name] = {'errors': errs, 'failed_requests': fails}
                bad += len(errs) + len(fails)
                print(('OK  ' if not errs and not fails else 'BAD ') + name, errs[:3], fails[:3])
                ctx.close()
        b.close()
    srv.shutdown(); Path(a.out, 'report.json').write_text(json.dumps(report, indent=1))
    raise SystemExit(1 if bad else 0)


if __name__ == '__main__':
    main()
