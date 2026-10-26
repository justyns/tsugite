// Serializes daemon lifecycle transitions. `connect({ adopt, superseded })` starts what the current
// mode needs and returns the target { origin, token }. Call `adopt(daemon)` right before `daemon.start()`
// with no await between them. The supervisor owns an adopted daemon until it has exited. A stale run
// cannot adopt. When connect fails, an adopted daemon gets `exitGraceMs` to report its exit, which then replaces the
// connect error.
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

class Supervisor {
  constructor({ connect, onReady, onError, exitGraceMs = 1000 }) {
    Object.assign(this, { connect, onReady, onError, exitGraceMs });
    this.target = { origin: '', token: '' };
    this.generation = 0;
    this.chain = Promise.resolve();
    this.owned = new Set();
    this.stops = new Map();
    this.quitting = false;
  }

  launch() {
    if (this.quitting) return Promise.resolve();
    const gen = ++this.generation;
    this.stopOwned().catch(() => {});
    const run = this.chain.then(() => this.run(gen));
    this.chain = run.catch(() => {});
    return run;
  }

  async quit() {
    this.quitting = true;
    this.generation++;
    try {
      await this.stopOwned();
    } catch (err) {
      this.quitting = false;
      throw err;
    }
  }

  stopOwned() {
    return Promise.all([...this.owned].map((d) => this.stop(d)));
  }

  stop(daemon) {
    if (!this.stops.has(daemon)) {
      const stopping = daemon
        .stop()
        .then(() => this.owned.delete(daemon))
        .finally(() => this.stops.delete(daemon));
      this.stops.set(daemon, stopping);
    }
    return this.stops.get(daemon);
  }

  async run(gen) {
    const stale = () => gen !== this.generation;
    try {
      await this.stopOwned();
    } catch (err) {
      if (!stale()) this.onError(`The previous daemon did not stop: ${err.message}`);
      return;
    }
    if (stale()) return;
    this.target = { origin: '', token: '' };
    let ready = false;
    let crashed = '';
    let noteExit;
    const exited = new Promise((r) => (noteExit = r));
    const adopt = (daemon) => {
      if (stale()) throw new Error('Launch superseded');
      this.owned.add(daemon);
      daemon.on('exit', (reason) => {
        crashed = reason;
        noteExit();
        if (ready && !stale()) this.onError(reason);
      });
    };
    let target;
    try {
      target = await this.connect({ adopt, superseded: stale });
    } catch (err) {
      if (!crashed && this.owned.size) await Promise.race([exited, sleep(this.exitGraceMs)]);
      if (!stale()) this.onError(crashed || err.message);
      return;
    }
    if (stale()) return;
    this.target = target;
    ready = true;
    this.onReady(target);
  }
}

module.exports = { Supervisor };
