"""Bounded retries for filesystem failures; semantic errors remain fatal."""
import errno
from functools import wraps
import json
import sys
import time

TRANSIENT = {errno.ESTALE, errno.EIO, errno.ETIMEDOUT, errno.EINTR, errno.EAGAIN, errno.ENOENT}
DELAYS = (0.1, 0.5, 1.5)

def transient(error):
    return isinstance(error, OSError) and error.errno in TRANSIENT

def retry(operation, *args, **kwargs):
    for attempt in range(len(DELAYS) + 1):
        try:
            return operation(*args, **kwargs)
        except OSError as error:
            if not transient(error) or attempt == len(DELAYS):
                raise
            time.sleep(DELAYS[attempt])

def strict_io(operation):
    @wraps(operation)
    def wrapped(*args, **kwargs):
        return retry(operation, *args, **kwargs)
    return wrapped

def optional_monitoring(operation):
    """Defer only exhausted filesystem failures; never discard bad metadata."""
    @wraps(operation)
    def wrapped(*args, **kwargs):
        try:
            return retry(operation, *args, **kwargs)
        except OSError as error:
            if not transient(error):
                raise
            print(json.dumps(dict(event='monitoring_io_deferred', operation=operation.__name__,
                                  errno=error.errno, message=str(error), at_unix=time.time())),
                  file=sys.stderr, flush=True)
            return None
    return wrapped
