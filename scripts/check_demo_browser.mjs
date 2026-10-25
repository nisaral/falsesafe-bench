import { spawn } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { readFile, writeFile, mkdir, access } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { setTimeout as sleep } from 'node:timers/promises';
import { Script } from 'node:vm';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const hash = contents => createHash('sha256').update(contents).digest('hex');

function within(parent, filename) {
  const relative = path.relative(parent, filename);
  return relative !== '..' && !relative.startsWith(`..${path.sep}`) && !path.isAbsolute(relative);
}

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

async function timeout(promise, label, milliseconds = 15000) {
  let timer;
  try {
    return await Promise.race([promise, new Promise((resolve, reject) => {
      timer = setTimeout(() => reject(new Error(`Timed out: ${label}`)), milliseconds);
    })]);
  } finally {
    clearTimeout(timer);
  }
}

async function connect(url) {
  const socket = new WebSocket(url);
  await timeout(new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  }), 'DevTools connection');
  const pending = new Map();
  const errors = [];
  let identifier = 0;
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.id) {
      const handlers = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) handlers?.reject(new Error(JSON.stringify(message.error)));
      else handlers?.resolve(message.result);
    } else if (message.method === 'Runtime.exceptionThrown') {
      errors.push(message.params.exceptionDetails);
    } else if (message.method === 'Runtime.consoleAPICalled' && message.params.type === 'error') {
      errors.push(message.params.args);
    }
  });
  const send = (method, params = {}) => timeout(new Promise((resolve, reject) => {
    identifier += 1;
    pending.set(identifier, { resolve, reject });
    socket.send(JSON.stringify({ id: identifier, method, params }));
  }), method);
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate', { expression, returnByValue: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  };
  return { socket, send, evaluate, errors };
}

async function main() {
  const options = {};
  const argumentsList = process.argv.slice(2);
  if (argumentsList.includes('--help')) {
    console.log('node scripts/check_demo_browser.mjs --browser <Chrome/Chromium executable> --run-dir <simulation run>');
    return;
  }
  for (let index = 0; index < argumentsList.length; index += 2) {
    requireCondition(['--browser', '--run-dir'].includes(argumentsList[index]) && argumentsList[index + 1],
      'Expected explicit --browser and --run-dir arguments');
    options[argumentsList[index]] = argumentsList[index + 1];
  }
  requireCondition(options['--browser'] && options['--run-dir'] && typeof WebSocket === 'function',
    'Use Node 22+ with an explicitly selected browser and run directory');
  const directory = path.resolve(options['--run-dir']);
  requireCondition(within(path.join(root, 'results', 'runs'), directory), 'Select a project-local simulation run');
  await access(options['--browser']);
  const record = JSON.parse(await readFile(path.join(directory, 'runlog.json'), 'utf8'));
  requireCondition(record.status === 'ok' && record.experiment === 'simulation' && !record.test_split_used
    && record.git_tree_clean && record.git_tree_clean_end && record.git_commit === record.git_commit_end,
  'Browser checks require a successful clean-source simulation diagnostic');
  const html = await readFile(path.join(directory, 'demo.html'), 'utf8');
  const replayContents = await readFile(path.join(directory, 'replay.json'));
  requireCondition(hash(html) === record.artifact_hashes['demo.html']
    && hash(replayContents) === record.artifact_hashes['replay.json'], 'Replay/HTML source hash mismatch');
  const replay = JSON.parse(replayContents);
  requireCondition(replay.metadata.run_id === record.run_id && replay.metadata.learning === false
    && replay.metadata.safety_certificate === false, 'Replay identity or diagnostic label mismatch');
  for (const match of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) {
    if (!match[0].includes('application/json')) new Script(match[1]);
  }
  const profile = path.join(root, '.cache', 'browser', randomUUID());
  await mkdir(profile, { recursive: true });
  const browser = spawn(options['--browser'], ['--headless=new', '--disable-gpu', '--no-first-run',
    '--no-default-browser-check', '--disable-background-networking', '--disable-extensions',
    '--remote-debugging-port=0', '--remote-debugging-address=127.0.0.1',
    `--user-data-dir=${profile}`, '--window-size=1440,1100', 'about:blank'],
  { windowsHide: true, stdio: ['ignore', 'ignore', 'pipe'] });
  let browserStderr = '';
  let startupError;
  browser.stderr.on('data', contents => { browserStderr += contents.toString(); });
  browser.on('error', error => { startupError = error; });
  let protocol;
  let failure;
  let result;
  try {
    const activePortPath = path.join(profile, 'DevToolsActivePort');
    let activePort;
    for (let attempt = 0; attempt < 150; attempt += 1) {
      try {
        activePort = await readFile(activePortPath, 'utf8');
        break;
      } catch (error) {
        if (error.code !== 'ENOENT') throw error;
        if (startupError) throw startupError;
        if (browser.exitCode !== null) throw new Error(`Browser exited before startup: ${browserStderr}`);
        await sleep(100);
      }
    }
    requireCondition(activePort, `Browser did not expose DevTools within its startup budget: ${browserStderr}`);
    const port = Number(activePort.split('\n')[0]);
    const response = await fetch(`http://127.0.0.1:${port}/json/new?about:blank`, { method: 'PUT' });
    requireCondition(response.ok, 'DevTools target creation failed');
    const target = await response.json();
    protocol = await connect(target.webSocketDebuggerUrl);
    await protocol.send('Runtime.enable');
    await protocol.send('Page.enable');
    await protocol.send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false });
    await protocol.send('Page.navigate', { url: pathToFileURL(path.join(directory, 'demo.html')).href });
    let ready = false;
    for (let attempt = 0; attempt < 100; attempt += 1) {
      ready = await protocol.evaluate("document.readyState === 'complete' && !!document.querySelector('.agent')");
      if (ready) break;
      await sleep(50);
    }
    requireCondition(ready, 'Replay did not initialise its Canvas and state panels');
    requireCondition(await protocol.evaluate(`document.querySelectorAll('.agent').length === ${replay.frames[0].agents.length}
      && document.getElementById('arena').width > 0 && document.querySelector('.badge').textContent.includes('not learned')`),
    'Robot cards, Canvas dimensions, or integrity badge failed');
    const midpoint = Math.floor((replay.frames.length - 1) / 2);
    await protocol.evaluate(`(() => { const timeline = document.getElementById('timeline'); timeline.value = ${midpoint};
      timeline.dispatchEvent(new Event('input', { bubbles: true })); document.getElementById('speed').value = '2';
      document.getElementById('play').click(); return true; })()`);
    await sleep(350);
    requireCondition(await protocol.evaluate(`Number(document.getElementById('timeline').value) > ${midpoint}`),
      'Playback did not advance actual recorded frames');
    await protocol.evaluate("document.getElementById('restart').click()");
    requireCondition(await protocol.evaluate("Number(document.getElementById('timeline').value) === 0 && document.getElementById('play').textContent === 'Play'"),
      'Restart did not reset the replay');
    await protocol.evaluate("window.dispatchEvent(new KeyboardEvent('keydown', { code: 'ArrowRight' }))");
    requireCondition(await protocol.evaluate("Number(document.getElementById('timeline').value) === 1"), 'Keyboard stepping failed');
    await protocol.send('Emulation.setDeviceMetricsOverride', { width: 390, height: 844, deviceScaleFactor: 1, mobile: true });
    await sleep(100);
    requireCondition(await protocol.evaluate("document.documentElement.scrollWidth <= innerWidth + 1 && document.getElementById('arena').clientWidth > 0"),
      'Mobile layout overflows or has no Canvas');
    await protocol.send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 1100, deviceScaleFactor: 1, mobile: false });
    await protocol.evaluate(`(() => { const timeline = document.getElementById('timeline'); timeline.value = ${midpoint};
      timeline.dispatchEvent(new Event('input', { bubbles: true })); return true; })()`);
    await sleep(100);
    requireCondition(protocol.errors.length === 0, `Browser runtime errors: ${JSON.stringify(protocol.errors)}`);
    const screenshot = await protocol.send('Page.captureScreenshot', { format: 'png' });
    const image = Buffer.from(screenshot.data, 'base64');
    await writeFile(path.join(directory, 'browser_preview.png'), image);
    result = { source_run_id: record.run_id, source_commit: record.git_commit,
      replay_sha256: hash(replayContents), html_sha256: hash(html), preview_sha256: hash(image),
      selected_frame_index: midpoint, frame_selection_rule: 'fixed midpoint of the recorded timeline; no outcome selection',
      screenshot: 'browser_preview.png', browser: await protocol.send('Browser.getVersion'), node: process.version,
      script_sha256: hash(await readFile(fileURLToPath(import.meta.url))),
      checks: ['script_syntax', 'runtime', 'state_panels', 'playback', 'speed', 'scrub', 'restart', 'keyboard', 'mobile_layout'],
      learning: false, safety_certificate: false, benchmark_metrics: false };
  } catch (error) {
    failure = error;
  } finally {
    if (protocol) {
      try {
        await protocol.send('Browser.close');
      } catch (error) {
        console.error('Browser cleanup failed:', error.message);
        failure ??= error;
      }
      protocol.socket.close();
    }
    await sleep(100);
    if (browser.exitCode === null) browser.kill();
  }
  if (failure) throw failure;
  await writeFile(path.join(directory, 'browser_check.json'), `${JSON.stringify(result, null, 2)}\n`);
  console.log(JSON.stringify(result, null, 2));
}

await main();
