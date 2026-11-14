"""Hold kill signals back while audio is being saved."""

import contextlib
import os
import signal
import threading

EXIT_SIGNAL_NAMES = ("SIGHUP", "SIGTERM")  # a closed terminal; a polite kill


@contextlib.contextmanager
def deferred_exit_signals():
    """Record SIGHUP/SIGTERM instead of dying, for as long as the block runs.

    Yields the list of signals received so far, so a loop can notice one and wind down.
    On leaving, the previous handlers are restored and the first signal received is sent
    again, so the process still ends the way the sender asked, only after the save.
    Does nothing off the main thread or where a signal does not exist (Windows has no SIGHUP).
    """
    received: list[int] = []
    previous: dict[int, object] = {}
    if threading.current_thread() is threading.main_thread():
        for name in EXIT_SIGNAL_NAMES:
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                previous[sig] = signal.signal(sig, lambda signum, _frame: received.append(signum))
            except (ValueError, OSError):
                continue
    try:
        yield received
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, signal.SIG_DFL if handler is None else handler)
        if received:
            os.kill(os.getpid(), received[0])
