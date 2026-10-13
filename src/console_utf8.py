# -*- coding: utf-8 -*-
"""控制台编码：把 stdout/stderr 切成 UTF-8。

单独拆成一个模块，是为了让 scripts/ 下的验证脚本也能用上它，又不必为了这
十来行去 import asr_onnx_decode（那会把 onnxruntime / soundfile 整套拖进来，
而 probe_mel_iterate.py 本来只依赖 numpy）。

用法：
    from console_utf8 import setup_stdout_utf8
    setup_stdout_utf8()
"""

import sys


def setup_stdout_utf8():
    """把 stdout/stderr 切成 UTF-8。

    识别结果是中文，而 Windows 控制台默认代码页常是 gbk，不切的话满屏乱码。
    reconfigure 是 3.7+ 才有的，失败就静默跳过（不影响解码，只是显示难看）。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
