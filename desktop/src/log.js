const fs = require('node:fs');
const path = require('node:path');

class RotatingLog {
  constructor(file, { maxBytes = 5 * 1024 * 1024, keep = 3 } = {}) {
    this.file = file;
    this.maxBytes = maxBytes;
    this.keep = keep;
    fs.mkdirSync(path.dirname(file), { recursive: true });
    this.size = fs.existsSync(file) ? fs.statSync(file).size : 0;
  }

  write(chunk) {
    if (this.size + chunk.length > this.maxBytes) this.rotate();
    fs.appendFileSync(this.file, chunk);
    this.size += chunk.length;
  }

  rotate() {
    for (let i = this.keep - 1; i >= 1; i--) {
      if (fs.existsSync(`${this.file}.${i}`)) fs.renameSync(`${this.file}.${i}`, `${this.file}.${i + 1}`);
    }
    if (fs.existsSync(this.file)) fs.renameSync(this.file, `${this.file}.1`);
    this.size = 0;
  }

  tail(lines = 40) {
    if (!fs.existsSync(this.file)) return '';
    return fs.readFileSync(this.file, 'utf8').split('\n').slice(-lines).join('\n');
  }
}

module.exports = { RotatingLog };
