const CALLBACK_PATH = '/api/plugins/mcp/callback';

// The query string of a navigation to the loopback redirect the MCP plugin registers, else null.
function loopbackCallback(url) {
  try {
    const u = new URL(url);
    return u.protocol === 'http:' && u.hostname === '127.0.0.1' && u.pathname === CALLBACK_PATH ? u.search : null;
  } catch {
    return null;
  }
}

function watchSignin(contents, forward) {
  const check = (details) => {
    const search = loopbackCallback(details.url);
    if (search === null) return;
    details.preventDefault();
    forward(search);
  };
  contents.on('will-navigate', check);
  contents.on('will-redirect', check);
}

async function forwardCallback(origin, search, fetchFn = fetch) {
  const resp = await fetchFn(origin + CALLBACK_PATH + search);
  if (!resp.ok) throw new Error(`The daemon rejected the sign-in (${resp.status}).`);
}

module.exports = { forwardCallback, loopbackCallback, watchSignin };
