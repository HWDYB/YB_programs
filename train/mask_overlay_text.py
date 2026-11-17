"""
批量抹掉采集脚本烧进画面的黄色提示文字（左上角 #N/600 / 摆好后点屏幕拍 / 类别名）

用法（在 VSCode 里按运行键即可，或用下面这条命令）：
    D:\\PythonProjects\\python.exe D:\\yolo_course\\xiaosai_arena_202610\\mask_overlay_text.py

改路径就改下面 JOBS 里那几行：
    左边是原图目录，右边是输出目录（不存在会自动建）
原图一个字节都不动；输出目录里是改过的副本，文件名不变。

做法不是盖一块黑矩形，而是：
1. 找出黄色文字像素（只在上方 110 行里找）
2. 把文字像素外扩 3 像素当掩膜
3. 用周围像素修补（OpenCV inpaint）
这样贴着画面上边缘的盒子不会被黑块盖掉。

末尾会打印每个目录处理了几张、剩多少黄色残留、跳过几张。
"""

import os
import numpy as np
import cv2

JOBS = [
    (r"D:\yolo_course\xiaosai_arena_202610\arena_box\row\box", r"D:\yolo_course\xiaosai_arena_202610\arena_box\row_clean\box"),
    (r"D:\yolo_course\xiaosai_arena_202610\arena_box\row\blanket", r"D:\yolo_course\xiaosai_arena_202610\arena_box\row_clean\blanket"),
]

TOP_ROWS = 110  # 只在画面最上面这么多行里找文字（黄字都在前 92 行内）
YELLOW = dict(r_min=150, g_min=110, b_max=90, min_pixels=40)
DILATE = 3
INPAINT_RADIUS = 3
EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def imread_u(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_u(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1].lower() or ".jpg", img)
    if ok:
        buf.tofile(path)
    return ok


def yellow_mask(top):
    b = top[:, :, 0].astype(np.int16)
    g = top[:, :, 1].astype(np.int16)
    r = top[:, :, 2].astype(np.int16)
    m = ((r > YELLOW["r_min"]) & (g > YELLOW["g_min"]) & (b < YELLOW["b_max"])).astype(np.uint8) * 255
    if m.sum() // 255 < YELLOW["min_pixels"]:
        return None
    k = np.ones((DILATE, DILATE), np.uint8)
    return cv2.dilate(m, k, iterations=2)


def main():
    for src, dst in JOBS:
        if not os.path.isdir(src):
            print("源目录不存在，跳过：", src)
            continue
        os.makedirs(dst, exist_ok=True)
        files = [f for f in sorted(os.listdir(src)) if f.lower().endswith(EXTS)]
        done = skipped = 0
        residual = []
        for f in files:
            img = imread_u(os.path.join(src, f))
            if img is None:
                print("读不了，跳过：", f)
                skipped += 1
                continue
            h, w = img.shape[:2]
            top = img[0 : min(TOP_ROWS, h), :]
            mask = yellow_mask(top)
            if mask is None:
                imwrite_u(os.path.join(dst, f), img)
                residual.append(0)
                done += 1
                continue
            full = np.zeros((h, w), np.uint8)
            full[0 : mask.shape[0], 0 : mask.shape[1]] = mask
            img = cv2.inpaint(img, full, INPAINT_RADIUS, cv2.INPAINT_TELEA)
            imwrite_u(os.path.join(dst, f), img)
            left = int((yellow_mask(img[0 : min(TOP_ROWS, h), :]) is not None))
            residual.append(left)
            done += 1
        print("%s -> %s   处理 %d 张，跳过 %d 张，黄色残留还剩 %d 张" % (os.path.basename(src), os.path.basename(dst), done, skipped, sum(residual)))


if __name__ == "__main__":
    main()
