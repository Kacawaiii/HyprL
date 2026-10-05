"""Children of the private operations controller; workloads reuse the existing runners."""
import argparse
import os
import signal
import threading

from scripts.trading_lab.ops.control import load_config, save, versions, stamp, edgar_preflight


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--service", choices=("app", "workers", "edgar"), required=True)
    parser.add_argument("--marker", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    root, name = config["runtime_root"], args.service
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    ready = {"pid": os.getpid(), "state": "READY", "at": stamp(), "versions": versions()}
    if name == "app":
        from scripts.trading_lab.app_api.server import make_server
        server = make_server("data/crypto", port=config["port"], ops_root=root,
                             fomc_store=config.get("fomc_store"), edgar_store=config.get("edgar_archive"),
                             research_root=config.get("research_root"))
        server.timeout = .2
        try:
            save(root / (name + ".ready.json"), ready)
            while not stop.is_set():
                server.handle_request()
        finally:
            server.server_close()
    elif name == "workers":
        from scripts.trading_lab.platform.jobs import JobRunner
        with JobRunner(root / "lab"):
            save(root / (name + ".ready.json"), ready)
            stop.wait()
    else:
        from scripts.trading_lab.edgar.service import run
        edgar_preflight(config, allow_capture=True)
        # Actual service still owns the durable store, budget and dispatch gates.
        def log(_):
            if not (root / (name + ".ready.json")).exists():
                save(root / (name + ".ready.json"), ready)
        run(root / "edgar", config["edgar_authorization"], stop=stop, log=log)
    save(root / (name + ".ready.json"), dict(ready, state="STOPPED", at=stamp()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
