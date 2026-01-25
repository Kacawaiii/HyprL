from __future__ import annotations

import asyncio
import inspect
import warnings

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message="scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.",
)
def pytest_pyfunc_call(pyfuncitem):  # type: ignore[override]
    marker = pyfuncitem.get_closest_marker("asyncio")
    if marker is None:
        return None
    func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(func):
        return None
    argnames = getattr(pyfuncitem._fixtureinfo, "argnames", ())
    testargs = {name: pyfuncitem.funcargs[name] for name in argnames if name in pyfuncitem.funcargs}
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(func(**testargs))
    finally:
        loop.close()
    return True
