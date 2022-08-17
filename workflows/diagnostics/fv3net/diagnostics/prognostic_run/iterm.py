"""
From https://github.com/wookayin/python-imgcat

License:

The MIT License

Copyright (c) 2018 Jongwook Choi

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""
import base64
import os

TMUX_WRAP_ST = b"\033Ptmux;"
TMUX_WRAP_ED = b"\033\\"

OSC = b"\033]"
CSI = b"\033["
ST = b"\a"  # \a = ^G (bell)


def write_image(buf, fp, filename, width=None, height=None, preserve_aspect_ratio=None):
    # need to detect tmux
    is_tmux = "TMUX" in os.environ and "tmux" in os.environ["TMUX"]

    # tmux: print some margin and the DCS escape sequence for passthrough
    # In tmux mode, we need to first determine the number of actual lines
    if is_tmux:
        fp.write(b"\n" * height)
        # move the cursers back
        fp.write(CSI + b"?25l")
        fp.write(CSI + str(height).encode() + b"F")  # PEP-461
        fp.write(TMUX_WRAP_ST + b"\033")

    # now starts the iTerm2 file transfer protocol.
    fp.write(OSC)
    fp.write(b"1337;File=inline=1")
    fp.write(b";size=" + str(len(buf)).encode())
    if filename:
        if isinstance(filename, bytes):
            filename_bytes = filename
        else:
            filename_bytes = filename.encode()
        fp.write(b";name=" + base64.b64encode(filename_bytes))
    if height is not None:
        fp.write(b";height=" + str(height).encode())
    if width is not None:
        fp.write(b";width=" + str(width).encode())
    if not preserve_aspect_ratio:
        fp.write(b";preserveAspectRatio=0")

    fp.write(b":")
    fp.flush()

    buf_base64 = base64.b64encode(buf)
    fp.write(buf_base64)

    fp.write(ST)

    if is_tmux:
        # terminate DCS passthrough mode
        fp.write(TMUX_WRAP_ED)
        # move back the cursor lines down
        fp.write(CSI + str(height).encode() + b"E")
        fp.write(CSI + b"?25h")
    else:
        fp.write(b"\n")

    # flush is needed so that the cursor control sequence can take effect
    fp.flush()
