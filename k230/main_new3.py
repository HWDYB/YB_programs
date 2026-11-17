# k230/main.py -- 视觉主程序：YOLO 找物体 -> 算「爪夹中心到物体中心」的实时偏移 -> 发串口
#
# 基于队友的 dp15.py 改的（PipeLine + libs.YOLO.YOLOv8，模型 480 档，UART1 IO9/IO10，
# 8 字节帧）。帧结构 0xAA 0x55 dx_hi dx_lo dy_hi dy_lo kind cs，
# kind: 0=没识别到/本帧不可用，1=ball 2=cone 3=block（2026-11 按电控要求，原来的 flag 换成种类）。
#
# 改了三处：
#   1) 偏移量基准从"画面中心 (320,240)"改成 GRASP_PX（爪夹中心对应的像素）。
#      因为相机和爪刚性固定，爪中心在画面里永远是这个像素，所以每帧算出来的
#      (物体中心 − GRASP_PX) 就是**实时剩余偏移**，不需要识别爪、也不需要记账。
#   2) 过滤 box 类（队友文档里点出的坑：box 不是目标，有框就 Flag=1 会指错）
#   3) 两级兜底：YOLO 没框时先用颜色掩膜（COLOR，约 1ms，误差 3~12px），
#      颜色也没结果才沿用上一帧的框（HOLD，最多 0.6 秒）。
#      注意 find_blobs 用不了：它只认 RGB565，而 PipeLine 给的是 rgb888p ndarray，
#      所以颜色检测直接用 ulab 在平面帧上算。
#   4) 偏移量范围检查 + 屏上大字（现场标定和调试不用盯电脑）
#
# ★ 跑之前先 Ctrl-D 软复位一次，否则 pl.create() 报 sensor(2) is already inited
#   （这个异常被 PipeLine 吞掉，表现成"打完 sensor 那行就再没声音"）
#
# ★ GRASP_PX 标定方法（一次，装车后相机锁死再做）：
#     车停好 -> 把球放台面 -> 遥控让爪张开开到球正上方、慢慢合拢夹住（球居中）
#     此时物理上"爪中心 = 球中心"，读终端打印的 cx,cy，填进下面 GRASP_PX
#     重复 3 次取平均（球对称，中心最不含糊）

import os
import sys
import time

sys.path.append("/sdcard")
import machine
import ulab.numpy as np
from libs.PipeLine import PipeLine
from libs.YOLO import YOLOv8

# ==================== 标定 / 阈值 ====================
GRASP_PX = (339, 354)     # ← 标定后改这里：(cx, cy)。未标定时就是画面中心
MIN_SCORE = 0.45          # 新模型（int16激活+uint8权重）板上真目标实测 0.8x，取 0.45 留余量；
MAX_OFF_PX = 250          # 偏移量超过它当误检（kind 发 0，电控别动）
BOX_CLASS = -1            # 新模型三类，没有 box；-1 表示不过滤任何类
LABELS = ("ball", "cone", "block")
# kind 字节 = LABELS 下标 + 1 -> 1=ball 2=cone 3=block（电控侧语义未变）
HOLD_PX = 40              # 迟滞：候选离上一帧目标小于它就算同一个，继续跟它，避免两个物块之间跳

TEST_FRAMES = 0           # >0 时跑满这么多帧自己退出（联调用）；正式跑保持 0
PRINT_EVERY = 10          # 每 N 帧打一行

HOLD_FRAMES = 10          # YOLO 和颜色都没结果时，沿用上一帧的框，最多这么多帧（约 0.6 秒）

# ---- 颜色兜底：YOLO 没框时直接在 rgb888p 平面帧上算红/蓝掩膜（约 1ms，不掉帧）----
# 实测和 YOLO 框心差 3~12px（约 3~4mm），比"保持旧框"更可信，所以排在 HOLD 之前
COLOR_FALLBACK = True
FB_STEP = 8               # 子采样步长 -> 80x60 个采样点
FB_THR = 1.35             # R 比 G、B 大这么多倍算红
FB_MIN_HITS = 8           # 命中采样点少于这个数就当没找到
FB_MAX_HITS = 1500        # 命中超过这个数（约占画面 30%）当背景/地垫，丢掉
FB_WIN = 130              # 只在上一帧目标 ±130px 的窗口里找（防止抓到别处的红色）

os.exitpoint(os.EXITPOINT_ENABLE)

pl = PipeLine(rgb888p_size=[640, 480], display_size=[640, 480],
              display_mode="lcd", osd_layer_num=4)
pl.create()

yolo = YOLOv8(task_type="detect", mode="video",
              kmodel_path="/sdcard/i16u8.kmodel",
              labels=["ball", "cone", "block"],
              rgb888p_size=[640, 480], model_input_size=[480, 480],
              display_size=[640, 480],
              conf_thresh=MIN_SCORE, nms_thresh=0.45, max_boxes_num=30,
              debug_mode=0)
yolo.config_preprocess()

# ---- 串口：UART1 IO9(TXD)/IO10(RXD)，115200。8 字节帧格式没变
# 2026-11 改为 IO9/IO10（原 dp15 的 UART3 IO32/IO33 已弃用，电控按 IO9/IO10 接线）
fpioa = machine.FPIOA()
fpioa.set_function(9, fpioa.UART1_TXD)
fpioa.set_function(10, fpioa.UART1_RXD)
uart = machine.UART(machine.UART.UART1, baudrate=115200)


def send_frame(dx, dy, kind):
    f = [0xAA, 0x55, (dx >> 8) & 0xFF, dx & 0xFF,
         (dy >> 8) & 0xFF, dy & 0xFF, kind]
    cs = 0
    for b in f:
        cs = (cs + b) & 0xFF
    f.append(cs)
    uart.write(bytes(f))


def pick_target(res, prev):
    """挑一个非 box 目标。返回 (x, y, w, h, cls, score) 或 None。

    多个候选（两个物块同时在画面里）时按这个顺序决定，避免抓夹在两者之间发疯：
      1. 先看上一帧跟的那个还在不在：离 prev 小于 HOLD_PX 就当还是它，继续跟；
      2. 上一帧的目标没了 -> 取离爪中心 GRASP_PX 最近的那个。
    """
    if not res or len(res) < 3 or len(res[0]) == 0:
        return None
    boxes, classes, scores = res[0], res[1], res[2]
    cand = []
    for i in range(len(boxes)):
        if classes[i] == BOX_CLASS:
            continue
        if scores[i] < MIN_SCORE:
            continue
        x, y, w, h = boxes[i]
        cx = x + w / 2
        cy = y + h / 2
        cand.append((cx, cy, x, y, w, h, classes[i], scores[i]))
    if not cand:
        return None

    if prev is not None:
        best = None
        bd = HOLD_PX * HOLD_PX
        for c in cand:
            dx = c[0] - prev[0]
            dy = c[1] - prev[1]
            d = dx * dx + dy * dy
            if d < bd:
                bd = d
                best = c
        if best is not None:
            return best[2:]

    best = cand[0]
    bdx = best[0] - GRASP_PX[0]
    bdy = best[1] - GRASP_PX[1]
    bd = bdx * bdx + bdy * bdy
    for c in cand[1:]:
        dx = c[0] - GRASP_PX[0]
        dy = c[1] - GRASP_PX[1]
        d = dx * dx + dy * dy
        if d < bd:
            bd = d
            best = c
    return best[2:]


def color_target(frame, ref):
    """颜色兜底：在 (3,480,640) 的平面 RGB 帧上做子采样红/蓝掩膜，算窗口内质心。

    find_blobs 用不了（只认 RGB565，PipeLine 给的是 rgb888p ndarray），所以直接用 ulab。
    实测 1ms、和 YOLO 框心差 3~12px。

    只在上一次目标 ±FB_WIN 的窗口里找：既不会抓到画面别处的红色背景，
    也符合兜底的实际用途（YOLO 短暂丢帧时接着跟住同一个物体）。
    ref 为 None（还没有过任何目标）时直接放弃，不瞎猜。返回 (cx, cy) 或 None。
    """
    if ref is None:
        return None
    sub = frame[:, ::FB_STEP, ::FB_STEP]
    RH = sub.shape[1]
    CW = sub.shape[2]
    r0 = max(0, int((ref[1] - FB_WIN) / FB_STEP))
    r1 = min(RH, int((ref[1] + FB_WIN) / FB_STEP) + 1)
    c0 = max(0, int((ref[0] - FB_WIN) / FB_STEP))
    c1 = min(CW, int((ref[0] + FB_WIN) / FB_STEP) + 1)
    if r1 <= r0 or c1 <= c0:
        return None
    R = sub[0][r0:r1, c0:c1]
    G = sub[1][r0:r1, c0:c1]
    B = sub[2][r0:r1, c0:c1]
    red = (R > 60) * (R > G * FB_THR) * (R > B * FB_THR)
    blue = (B > 60) * (B > R * FB_THR) * (B > G * FB_THR)
    n_red = int(np.sum(red))
    n_blue = int(np.sum(blue))
    if n_red >= n_blue:
        mask, hits = red, n_red
    else:
        mask, hits = blue, n_blue
    if hits < FB_MIN_HITS or hits > FB_MAX_HITS:
        return None
    ys = np.arange(mask.shape[0])
    xs = np.arange(mask.shape[1])
    cy = (float(np.sum(np.sum(mask, axis=1) * ys)) / hits + r0) * FB_STEP
    cx = (float(np.sum(np.sum(mask, axis=0) * xs)) / hits + c0) * FB_STEP
    return (int(cx), int(cy))


n = 0
agree = 0
last = None
hold_box = None
hold = 0
t0 = time.ticks_ms()

try:
    while True:
        os.exitpoint()
        frame = pl.get_frame()
        res = yolo.run(frame)
        yolo.draw_result(res, pl.osd_img)

        # 在画面上标出基准点，方便现场看"物体有没有对准"
        pl.osd_img.draw_cross(GRASP_PX[0], GRASP_PX[1], color=(0, 255, 0), size=12)
        pl.osd_img.draw_circle(GRASP_PX[0], GRASP_PX[1], 8, color=(0, 255, 0), thickness=2)
        if GRASP_PX == (320, 240):
            pl.osd_img.draw_string_advanced(8, 440, 16, "GRASP_PX 未标定", color=(255, 0, 0))

        tgt = pick_target(res, (hold_box[0], hold_box[1]) if hold_box else None)
        cx = cy = -1
        w = h = 0
        cls = -1
        score = 0.0
        if tgt is not None:
            x, y, w, h, cls, score = tgt
            cx, cy = int(x + w / 2), int(y + h / 2)
            hold = 0
            hold_box = (cx, cy, w, h, cls, score)
            src = "YOLO"
        else:
            ct = color_target(frame, (hold_box[0], hold_box[1]) if hold_box else None) if COLOR_FALLBACK else None
            if ct is not None:
                cx, cy = ct[0], ct[1]
                if hold_box is not None:
                    cls = hold_box[4]      # 颜色兜底不分类，沿用上一帧的类别
                src = "COLOR"
            elif hold_box is not None and hold < HOLD_FRAMES:
                hold += 1
                cx, cy, w, h, cls, score = hold_box
                src = "HOLD"
            else:
                src = None
        n += 1

        if src is None:
            agree = 0
            last = None
            send_frame(0, 0, 0)
            pl.osd_img.draw_string_advanced(8, 8, 32, "NO TARGET", color=(255, 0, 0))
            if n % PRINT_EVERY == 0:
                print("NOBOX n=%d K=0" % n)
        else:
            dx = cx - GRASP_PX[0]
            dy = cy - GRASP_PX[1]
            if src != "HOLD":
                hold_box = (cx, cy, w, h, cls, score)
                hold = 0
            if src == "HOLD" and w > 0:
                pl.osd_img.draw_rectangle(cx - w // 2, cy - h // 2, w, h,
                                          color=(255, 140, 0), thickness=2)
            elif src == "COLOR":
                pl.osd_img.draw_circle(cx, cy, 24, color=(255, 140, 0), thickness=2)

            # 稳定帧数：连续几帧中心都在 30px 内算一致（给现场看，不参与发帧决策）
            if last is not None and abs(cx - last[0]) < 30 and abs(cy - last[1]) < 30:
                agree += 1
            else:
                agree = 1
            last = (cx, cy)

            # 误检保护：偏得离谱就不给电控（kind=0）
            ok = abs(dx) <= MAX_OFF_PX and abs(dy) <= MAX_OFF_PX
            kind = (cls + 1) if ok else 0    # 1=ball 2=cone 3=block，0=本帧别用
            send_frame(dx, dy, kind)

            # 屏上大字：现场标定和调试都靠它（不用盯电脑）
            pl.osd_img.draw_string_advanced(8, 8, 32,
                                            "dx=%+d dy=%+d" % (dx, dy),
                                            color=(255, 255, 0))
            name = "color" if src == "COLOR" else LABELS[cls]
            pl.osd_img.draw_string_advanced(8, 44, 20,
                                            "cx=%d cy=%d %-4s %s k=%d"
                                            % (cx, cy, src, name, kind),
                                            color=(0, 255, 255))
            pl.osd_img.draw_cross(cx, cy, color=(255, 0, 255), size=10)

            if n % PRINT_EVERY == 0:
                fps = n * 1000.0 / max(1, time.ticks_ms() - t0)
                print("BOX n=%d %-5s %-5s %.3f cx=%d cy=%d dx=%+d dy=%+d K=%d n=%d %.1ffps%s"
                      % (n, src, name, score, cx, cy, dx, dy,
                         kind, agree, fps,
                         "" if ok else "  [超范围丢]"))

        pl.show_image()          # ★ 必须每帧调，否则 OSD（十字/大字）不会合成到屏上
        if TEST_FRAMES and n >= TEST_FRAMES:
            print("[main] 测试模式跑满 %d 帧，正常退出" % TEST_FRAMES)
            break
finally:
    yolo.deinit()
    pl.destroy()