// Stands in for `tsu`: `daemon --config <file>` serves /api/health and /api/runtime, `daemon token create` prints a new
// token and counts mints in <valid token file>.mints, and live `desktop` tokens in .live, which `daemon token revoke desktop`
// decrements one at a time, exiting 1 at zero.
const fs = require('node:fs');
const http = require('node:http');

const args = process.argv.slice(2);
const mode = process.env.FAKE_MODE || '';
const validTokenFile = process.env.FAKE_VALID_TOKEN_FILE;

const bump = (kind, by) => {
  const file = `${validTokenFile}.${kind}`;
  const n = (fs.existsSync(file) ? Number(fs.readFileSync(file, 'utf8')) : 0) + by;
  fs.writeFileSync(file, String(n));
  return n;
};
if (args[2] === 'revoke') {
  if (args[3] !== 'desktop' || bump('live', 0) === 0) process.exit(1);
  bump('live', -1);
  process.exit(0);
}
if (args[1] === 'token') {
  bump('live', 1);
  const n = bump('mints', 1);
  fs.writeFileSync(validTokenFile, `tsu_minted_${n}`);
  console.log(`tsu_minted_${n}`);
  process.exit(0);
}

const config = fs.readFileSync(args[args.indexOf('--config') + 1], 'utf8');
const port = Number(/port:\s*(\d+)/.exec(config)[1]);

if (mode === 'exit-early') {
  console.error('boom: bad config');
  process.exit(3);
}
if (mode === 'orphan') {
  const child = require('node:child_process').spawn(process.execPath, ['-e', "process.on('SIGTERM',()=>{});setInterval(()=>{},1000)"], { stdio: 'ignore' });
  fs.writeFileSync(process.env.FAKE_CHILD_PID_FILE, String(child.pid));
}
process.on('SIGTERM', () => mode !== 'ignore-term' && process.exit(0));

const server = http.createServer((req, res) => {
  if (req.url === '/api/health') {
    res.end('{"status":"ok"}');
  } else if (req.url === '/api/runtime') {
    const ok = validTokenFile && fs.existsSync(validTokenFile) && req.headers.authorization === `Bearer ${fs.readFileSync(validTokenFile, 'utf8')}`;
    res.statusCode = ok ? 200 : 401;
    res.end('{}');
  } else {
    res.statusCode = 404;
    res.end();
  }
});
console.log('fake daemon starting');
if (mode !== 'never-healthy') server.listen(port, '127.0.0.1');
else setInterval(() => {}, 1000);
