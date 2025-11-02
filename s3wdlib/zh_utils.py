# -*- coding: utf-8 -*-
"""
中文可视化工具：
- set_chinese_font()：优先设置宋体（SimSun），若缺失则回退到 NSimSun / Noto Sans CJK SC / Microsoft YaHei
- fix_minus()：解决坐标轴负号显示为方块的问题
"""
from matplotlib import rcParams, font_manager
import matplotlib.pyplot as plt

def set_chinese_font(prefer=("SimSun","NSimSun","Noto Sans CJK SC","Microsoft YaHei")):
    # 查找系统中可用字体
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in prefer:
        if name in available:
            rcParams["font.family"] = [name]
            rcParams["font.sans-serif"] = [name]
            break
    else:
        # 若都找不到，仍设置默认 sans-serif，避免报错
        rcParams["font.family"] = ["sans-serif"]
    # 负号正常显示
    rcParams["axes.unicode_minus"] = False

def fix_minus():
    rcParams["axes.unicode_minus"] = False

def demo_plot():
    set_chinese_font()
    fix_minus()
    plt.figure()
    plt.title("中文标题（宋体优先）")
    plt.plot([0,1,2], [0,-1,4])
    plt.xlabel("横轴")
    plt.ylabel("纵轴")
    plt.tight_layout()
    plt.show()
