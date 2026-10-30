"""Headless Chrome walk-through of demos/gym with scripted keypresses.

Saves screenshots + frame sequences to /workspace/falsesafe-bench-shots/interactive/ (override
with FSB_SHOTS).  Usage: python scripts/shots_gym.py
"""
import asyncio, json, os
from playwright.async_api import async_playwright
OUT = os.environ.get('FSB_SHOTS', '/workspace/falsesafe-bench-shots/interactive')
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'demos'))
URL = 'file://' + ROOT + '/gym/index.html'
ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader']

async def main():
    os.makedirs(OUT, exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path='/usr/bin/google-chrome', args=ARGS)
        pg = await b.new_page(viewport={'width': 1500, 'height': 900}); errs = []
        pg.on('console', lambda m: errs.append(m.text) if m.type == 'error' else None)
        pg.on('pageerror', lambda x: errs.append(str(x)))
        await pg.goto(URL); await pg.wait_for_function('window.__gym&&window.__gym.ready')
        await pg.wait_for_timeout(800); await pg.screenshot(path=f'{OUT}/00_help_overlay.png')
        await pg.keyboard.press('h'); await pg.wait_for_timeout(300)
        names = await pg.evaluate('window.__gym.DOCS.map(d=>d.name)'); print('scenarios', names)
        R = 'JSON.stringify(window.__gym.readout)'
        # 1) every scenario: short drive with the default certifier
        for i, n in enumerate(names):
            await pg.evaluate(f'window.__gym.loadScenario({i})')
            await pg.keyboard.down('ArrowLeft' if n in ('cart_wall',) else 'ArrowRight')
            await pg.wait_for_timeout(1500)
            await pg.keyboard.up('ArrowLeft' if n in ('cart_wall',) else 'ArrowRight')
            await pg.screenshot(path=f'{OUT}/{i+1:02d}_{n}.png'); print(n, await pg.evaluate(R))
        # 2) cart_wall frame sequence: naive certifier, shield ON, drive into the wall through the gap
        await pg.evaluate("window.__gym.loadScenario(window.__gym.DOCS.findIndex(d=>d.name==='cart_wall'))")
        await pg.keyboard.press('r'); seq = []
        await pg.keyboard.down('ArrowLeft')
        for f in range(12):
            await pg.wait_for_timeout(250); await pg.screenshot(path=f'{OUT}/seq_cart_naive_{f:02d}.png')
            seq.append(json.loads(await pg.evaluate(R)))
        await pg.keyboard.up('ArrowLeft')
        print('cart naive seq', [(round(s['v_true'], 3), round(s['v_hat'], 3), s['false_safe'], s['active']) for s in seq])
        print('events', await pg.evaluate("document.getElementById('log').innerText"))
        # 3) same with the oracle (press C until oracle)
        k = await pg.evaluate("window.__gym.S.cases.findIndex(c=>/oracle/i.test(c.label))")
        await pg.evaluate(f'window.__gym.setCase({k})'); await pg.wait_for_timeout(1500); await pg.keyboard.press('r')
        await pg.keyboard.down('ArrowLeft')
        for f in range(10):
            await pg.wait_for_timeout(300); await pg.screenshot(path=f'{OUT}/seq_cart_oracle_{f:02d}.png')
        await pg.keyboard.up('ArrowLeft'); print('cart oracle', await pg.evaluate(R))
        # 4) replay + shield toggle + cycle keys
        await pg.keyboard.press('l'); await pg.wait_for_timeout(600); await pg.screenshot(path=f'{OUT}/20_replay.png')
        await pg.keyboard.press('Space'); await pg.keyboard.press('c'); await pg.keyboard.press('m')
        await pg.wait_for_timeout(800); await pg.screenshot(path=f'{OUT}/21_shield_off_cycled.png'); print('after keys', await pg.evaluate(R))
        # 5) adversarial driver on boat + batch
        await pg.evaluate("window.__gym.loadScenario(window.__gym.DOCS.findIndex(d=>d.name==='boat_rocks'))")
        await pg.evaluate("window.__gym.setPol('adversarial')"); await pg.wait_for_timeout(2500)
        await pg.screenshot(path=f'{OUT}/22_boat_adversarial.png')
        await pg.fill('#bN', '30'); await pg.select_option('#bPol', 'adversarial'); await pg.evaluate('document.activeElement.blur()'); await pg.keyboard.press('b')
        await pg.wait_for_function('window.__gym.batchDone', timeout=180000); await pg.wait_for_timeout(300)
        await pg.screenshot(path=f'{OUT}/23_boat_batch.png'); print('batch (in-browser)', await pg.evaluate('JSON.stringify(window.__gym.batch)'))
        # 6) rover 3D + user-defined env, then mobile layout with touch pad
        await pg.evaluate("window.__gym.loadScenario(window.__gym.DOCS.findIndex(d=>d.name==='rover_cliff'))")
        await pg.evaluate("window.__gym.setPol('goal')"); await pg.wait_for_timeout(3000); await pg.screenshot(path=f'{OUT}/24_rover_goal.png')
        await pg.close()
        m = await b.new_page(viewport={'width': 390, 'height': 844}, has_touch=True, is_mobile=True)
        await m.goto(URL + '?nohelp=1&scenario=drone_hover'); await m.wait_for_function('window.__gym&&window.__gym.ready')
        await m.tap('#touch button[data-k=up]'); await m.wait_for_timeout(1200); await m.screenshot(path=f'{OUT}/30_mobile_drone.png', full_page=True)
        print('errors:', errs); await b.close()

asyncio.run(main())
