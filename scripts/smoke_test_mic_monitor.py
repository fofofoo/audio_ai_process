# -*- coding: utf-8 -*-
"""realtime_mic_monitor.py 的 offscreen 冒烟测试：真开麦克风跑几秒。

验证三件事：
    1. 三块图 + 底部识别标签能搭起来
    2. 「回调线程 -> 主线程 -> 识别线程」三条线程的交接正常，不报错
    3. 退出时能补完尾帧并给出最终文本与统计

不弹窗：QT_QPA_PLATFORM=offscreen。
会真开麦克风采样几秒环境音 —— 这是唯一能验证回调链路的方式，没有假数据可替代。

用法（放项目根目录）：
    python -I scripts/smoke_test_mic_monitor.py [秒数]      # 默认 8 秒
"""
import os
import sys

os.environ["QT_QPA_PLATFORM"] = "offscreen"   # 必须在 import pyqtgraph 之前

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 核心库现在在 src/ 下：根目录用来 import 入口脚本，src/ 用来 import 核心库
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

import realtime_mic_monitor as mic


def main():
    mic.setup_stdout_utf8()   # 先转好编码，否则下面这句中文在 GBK 控制台上是乱码
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
    mic.LIST_DEVICES_ON_START = False   # 设备表已经确认过，冒烟时不刷屏

    app = pg.mkQApp()
    # main() 内部会再调一次 mkQApp，拿到的是同一个实例，所以这个定时器有效
    QtCore.QTimer.singleShot(int(seconds * 1000), app.quit)
    print(f"[冒烟] 麦克风将真开 {seconds:.0f} 秒，随便说点什么或保持安静...")

    mic.main()
    print("[冒烟] 已退出，三条线程的交接没崩")
    return 0


if __name__ == "__main__":
    sys.exit(main())
