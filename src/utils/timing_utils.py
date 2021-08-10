import logging
import time

import contextlib
from datetime import timedelta

logger = logging.getLogger(__name__)

# Misc logger setup so a debug log statement gets printed on stdout.
logger.setLevel("DEBUG")
logger.propagate = False
handler = logging.StreamHandler()
log_format = "%(asctime)s %(levelname)s -- %(message)s"
formatter = logging.Formatter(log_format)
handler.setFormatter(formatter)
logger.addHandler(handler)

@contextlib.contextmanager
def time_measure(ident, _logger=logger, show_started=True):
    if show_started:
        _logger.info("%s Started" % ident)
    start_time = time.time()
    yield
    elapsed_time = str(timedelta(seconds=time.time() - start_time))
    _logger.info("%s Finished in %s " % (ident, elapsed_time))
