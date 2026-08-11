# ADR: how HyprL is packaged for local use

**Status:** accepted (Phase 5E)
**Decision:** ship a single-origin localhost application served by the existing
Python process, opened in the user's own browser. No desktop wrapper.

## Context

After Phase 5D the application is a read-only Python API, a small React
cockpit, and a shadow trading engine driven from the command line. It runs on
one machine, for one user, against local artefacts and a local SQLite log.

The question is what "installed" should mean. The reflex answer in 2026 is
Electron or Tauri, and reflexes are how a 250 kB application acquires a
150 MB runtime.

## Options

### A. Single-origin localhost app + the user's browser  *(chosen)*

One Python process serves `/api/v1/*` and the built frontend from
`127.0.0.1:8787`. `./scripts/hyprl.sh start` builds if needed, starts it, and
opens a tab.

- **Bundle:** ~18 MB, of which 17 MB is the research corpus the app actually
  reads. The application itself is ~500 kB.
- **Memory:** ~3.5 MB RSS for the server. The browser is one the user already
  has running.
- **Security surface:** loopback bind, GET-only API, no IPC bridge, no native
  code, no privileged renderer.
- **Python sidecar:** none — Python *is* the app.
- **Updater:** replace a directory.
- **Cost:** zero. This is what already exists.

### B. Tauri wrapper

A native webview shell around the same server.

- **Bundle:** +3–10 MB, plus a Rust toolchain in the build.
- **Memory:** +40–80 MB for the webview.
- **Security surface:** an IPC bridge between web content and native code —
  a new class of vulnerability for a page that currently cannot do anything
  privileged.
- **Python sidecar:** real complexity. The Python process must be shipped,
  spawned, supervised and killed by the shell, and its lifecycle must survive
  the window closing. That is the supervisor from this phase, duplicated in
  Rust.
- **Updater:** signing keys and an update server, or it is worse than nothing.
- **Cost:** a new toolchain and a second lifecycle implementation, for a
  window frame.

### C. Electron

- **Bundle:** +120–180 MB. A full Chromium, next to the Chromium the user
  already runs.
- **Memory:** +150–300 MB.
- **Security surface:** the largest of the four, and the one with the most
  history.
- Everything wrong with B, plus the size.

### D. Native rewrite

Rewriting the cockpit in Qt, GTK or SwiftUI. Discards a working, tested,
accessible UI to remove a browser tab, and gives up cross-platform reach.
Not seriously considered.

## Decision

**Option A.**

The application is already a web app that talks to a local server. Wrapping it
in a second browser to hide the first one buys a window frame and a dock icon,
and costs a webview, an IPC bridge, a sidecar lifecycle, a signed updater and
a build toolchain. None of those makes a single number in this project more
correct.

The single-origin bundle also removes the one genuine reason a wrapper is
sometimes needed: cross-origin friction. In production there is one origin, so
there is no CORS to work around.

## What this does not preclude

Tauri would wrap *this exact architecture* — a localhost server plus static
assets — without changing it. Nothing here is a dead end. The moment a real
requirement appears (an app-store listing, an OS notification, a file
association, a tray icon), option B becomes a thin layer over unchanged code.

Until such a requirement exists, adding it would be packaging as decoration.

## Consequences

- Users open a browser tab. On a machine with no browser, there is no app.
- No auto-updater. The version is the commit in `manifest.json`; updating is
  replacing a directory. An updater able to silently replace the code that
  decides trades is a supply chain, and it needs signing before it deserves
  to exist. **No `curl | bash`.**
- The frontend must stay dependency-light and fully bundled: no CDN at
  runtime, no analytics, no remote font. The cockpit works offline for every
  view backed by local artefacts.
- The browser's own security model does the work an Electron sandbox would
  otherwise have to.
