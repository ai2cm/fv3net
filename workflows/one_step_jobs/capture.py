import logging
import contextlib
import tempfile
import sys
import io
import os
import sys
import threading
import time
import logging



@contextlib.contextmanager
def capture_stream(stream):

    # create a pipe to communicate with the fork
    r, w = os.pipe()

    # fork the process
    pid = os.fork()

    if pid > 0:
        # parent process:
        # close the reading end, we won't need this
        os.close(r)
        try:
            orig_file_handle = os.dup(stream.fileno())
            # overwrite the streams fileno with a the pipe to be read by the forked 
            # process below
            os.dup2(w, stream.fileno())
            yield
        finally:
            # restore the original file handle
            os.dup2(orig_file_handle, stream.fileno())

            # close the pipe
            os.close(w)
    else:
        # child process:
        # close the writing end, we won't need this
        os.close(w)
        with os.fdopen(r) as f:
            for line in f:
                logging.debug(line.strip())
        sys.exit(0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    with capture_stream(sys.stdout):
        print("should appear")
        print("should appear")
        print("should appear")