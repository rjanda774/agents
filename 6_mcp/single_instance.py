"""
Cross-platform single-instance lock for trading_floor.py.

Reproduced live: two trading_floor.py processes running concurrently against
the same 6_mcp/ checkout caused hard-to-diagnose Schwab flakiness -- both
independently read/refreshed/wrote the same schwab_token.json (schwab-py's
own token writer has no file locking of its own, see schwab_client.py), so
one process's write could land mid-read/mid-write of the other, corrupting
what the other saw and making a perfectly valid token look broken. Worse:
two live instances also means Cathie's account state (accounts.db) gets
driven by two independent trading cycles at once, risking duplicated
trading activity, not just a Schwab data-source problem.

Uses a real OS-level file lock (msvcrt on Windows, fcntl on POSIX) held for
the whole lifetime of the process, rather than a manually-managed PID file --
the OS releases the lock automatically when the process exits for any reason
(clean exit, crash, or being killed), so there's no stale-lock cleanup logic
to get wrong, unlike a PID file that could be left behind by a crash and
wrongly block a legitimate restart.
"""
import os
import sys

_LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".trading_floor.lock")

# Kept as a module global, not a local variable, so the file object (and the
# OS lock tied to its handle) stays alive for the process's whole lifetime --
# closing it, even implicitly via garbage collection, would release the lock.
_lock_file = None


class AlreadyRunningError(RuntimeError):
    """Another trading_floor.py process already holds the single-instance lock."""


def acquire_single_instance_lock():
    """Acquire the lock, or raise AlreadyRunningError if another process
    already holds it. Call this once, as early as possible in the process
    (before anything touches accounts.db or schwab_token.json). The lock is
    released automatically when this process exits, however it exits."""
    global _lock_file

    _lock_file = open(_LOCK_PATH, "a+")
    if os.path.getsize(_LOCK_PATH) == 0:
        # msvcrt.locking needs at least one byte to lock a region over.
        _lock_file.write("0")
        _lock_file.flush()

    try:
        _lock_file.seek(0)
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(_lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(_lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        _lock_file.close()
        _lock_file = None
        raise AlreadyRunningError(
            f"Another trading_floor.py process already appears to be running "
            f"(lock held on {_LOCK_PATH}). Two instances running at once will race "
            "on the same Schwab token file and drive Cathie's account state "
            "concurrently -- find and stop the other one first (on Windows: "
            "`Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | "
            "Select-Object ProcessId, CommandLine` to find it, then "
            "`Stop-Process -Id <PID>`)."
        ) from e

    # Lock acquired -- record our PID for diagnostics. Purely informational;
    # the OS-level lock above is what actually enforces single-instance, not
    # this content, so a benign race on this write from a process that's
    # simultaneously failing to acquire the lock is harmless.
    _lock_file.seek(0)
    _lock_file.truncate()
    _lock_file.write(str(os.getpid()))
    _lock_file.flush()
