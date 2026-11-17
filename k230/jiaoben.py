# -*- coding: utf-8 -*-
"""
jiaoben1.py -- 在 K230 上采识别数据集（红蓝球 / 锥桶 / 方块）
存 /sdcard/<CLASS>/，编号接着目录里已有的往下排，不会覆盖旧图。

用法
    1. 用 CanMV IDE 把本文件传到板子（文件名别叫 main.py）
    2. 在 IDE 里点运行
    3. 屏上出实时画面。摆好目标，点一下屏幕拍一张
    4. 触摸不可用时自动退回定时模式，每 INTERVAL_MS 一张
    5. 停下来看终端，会一行行打 已存 /sdcard/xxx/000.jpg

两个和上一版的关键区别
    显示用 Display.ST7701，不用 Display.VIRT
        VIRT 是"板子没屏、把画面推给 IDE"的虚拟显示。这块板子是
        ST7701 800x480 触摸屏，用 ST7701 画面直接出在屏上，站在板子
        前面就能看见，不依赖电脑也不依赖 IDE 的预览通道。

    保存显式给 JPEG_QUALITY
        默认 JPEG 质量是 50，边缘和角点会糊，拍出来的图拿去训练会吃亏。
        这里给 90。
"""

import os
import time
import gc

from media.sensor import *
from media.display import *
from media.media import *

# ---------------- 每次拍之前改这里 ----------------
CLASS = "box"  # ball/ cone/ block
CAP_N = 2000        # 这次拍多少张
INTERVAL_MS = 5000  # 没触摸时：每几毫秒自动拍一张
CAM_W, CAM_H = 800, 480  # 和屏一致
LCD_TYPE = "ST7701"
JPEG_QUALITY = 90
SAVE_ROOT = "/sdcard"
# --------------------------------------------------


def ensure_dir(path):
    """建目录。已经存在不算错，真出错会打出来。"""
    try:
        os.mkdir(path)
        print("mkdir:", path)
    except OSError as e:
        if e.args[0] != 17:  # 17 = EEXIST
            print("mkdir 失败:", e)


def init_display():
    try:
        Display.init(getattr(Display, LCD_TYPE), width=CAM_W, height=CAM_H, to_ide=True)
    except Exception as e:
        print("显示(带尺寸)失败:", e)
        Display.init(getattr(Display, LCD_TYPE), to_ide=True)


def make_trigger():
    """触摸优先，不行退回定时。"""
    try:
        from machine import TOUCH

        tp = TOUCH(0)
        tp.read()
        print("触发方式: 触摸（点屏幕拍一张）")
        return "touch", tp
    except Exception as e:
        print("触摸不可用:", e)
    print("触发方式: 定时，每 %d ms 一张" % INTERVAL_MS)
    return "timer", None


def next_index(folder):
    """接着目录里最大的编号往后排。"""
    n = 0
    try:
        for name in os.listdir(folder):
            if name.endswith(".jpg"):
                head = name[:-4]
                if head.isdigit():
                    v = int(head) + 1
                    if v > n:
                        n = v
    except Exception as e:
        print("扫目录失败，从 0 开始:", e)
    return n


def read_touch(dev):
    try:
        p = dev.read()
        return p is not None and len(p) > 0
    except Exception:
        return False


def main():
    sensor = None
    dev = None
    mode = "timer"
    saved = 0
    folder = SAVE_ROOT + "/" + CLASS
    ensure_dir(folder)

    try:
        sensor = Sensor(width=CAM_W, height=CAM_H)
        sensor.reset()
        sensor.set_framesize(width=CAM_W, height=CAM_H)
        sensor.set_pixformat(Sensor.RGB565)
        init_display()
        MediaManager.init()
        sensor.run()

        mode, dev = make_trigger()

        idx = next_index(folder)
        start_idx = idx
        print("目录:", folder, " 起始编号:", idx)

        # 让自动曝光稳住，这 2 秒别晃板子
        for _ in range(20):
            sensor.snapshot()
            time.sleep_ms(100)

        was = False
        last = time.ticks_ms()
        flash_until = 0

        while saved < CAP_N:
            os.exitpoint()
            img = sensor.snapshot()
            now = time.ticks_ms()

            img.draw_string_advanced(4, 4, 34, "#%d/%d" % (saved, CAP_N), color=(255, 255, 0))
            img.draw_string_advanced(4, 74, 18, CLASS, color=(255, 255, 0))
            if mode == "timer":
                left = INTERVAL_MS - (now - last)
                img.draw_string_advanced(4, 46, 24, "%.1fs" % (max(0, left) / 1000.0), color=(255, 160, 0))
            else:
                img.draw_string_advanced(4, 46, 22, "摆好后点屏幕拍", color=(255, 160, 0))
            if now < flash_until:
                img.draw_string_advanced(CAM_W // 2 - 60, CAM_H // 2 - 26, 34, "已存", color=(0, 255, 255))

            Display.show_image(img)

            if mode == "timer":
                fire = (now - last) >= INTERVAL_MS
            else:
                cur = read_touch(dev)
                fire = cur and not was  # 按下的那一刻，只拍一张
                was = cur

            if fire:
                path = "%s/%03d.jpg" % (folder, idx)
                try:
                    img.save(path, quality=JPEG_QUALITY)
                except TypeError:
                    img.save(path)  # 老固件不认 quality 参数
                saved += 1
                print("已存 %s (%d/%d)" % (path, saved, CAP_N))
                idx += 1
                last = time.ticks_ms()
                flash_until = last + 300
                gc.collect()
                time.sleep_ms(200)  # 防一次点击连拍两张

        print("拍完 %d 张，编号 %d ~ %d，目录 %s" % (saved, start_idx, idx - 1, folder))

    except KeyboardInterrupt:
        print("用户停止，已存 %d 张" % saved)
    except BaseException as e:
        print("ERROR:", e)
    finally:
        try:
            if isinstance(sensor, Sensor):
                sensor.stop()
        except Exception as e:
            print("sensor.stop 失败:", e)
        try:
            Display.deinit()
        except Exception as e:
            print("display.deinit 失败:", e)
        try:
            if dev is not None and mode == "touch":
                dev.deinit()
        except Exception:
            pass
        os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        time.sleep_ms(100)
        try:
            MediaManager.deinit()
        except Exception as e:
            print("MediaManager.deinit 失败:", e)
        print("cleanup done")


if __name__ == "__main__":
    main()
