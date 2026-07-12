"""tsugite-cc-driver: run interactive Claude Code sessions as verified tsugite jobs.

A job with executor="cc" is driven by an interactive `claude` process in a PTY.
HTTP Stop hooks decide whether the attempt is finished (completion marker or
continue-budget exhausted) or should be nudged onward; completion routes into the
existing job verifier/AC/retry machinery. See README.md.
"""
