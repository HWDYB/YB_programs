"""
给负样本配 0 字节标签（有图无框 = 这张图里没有 box）

背景：YOLO 认负样本的方式是「这张图有对应的 txt 文件，但文件里一行都没有」。
      没有 txt 文件的图会被直接跳过，等于没加进数据集。

用法（VSCode 里打开按运行键，或）：
    D:\\PythonProjects\\python.exe D:\\yolo_course\\xiaosai_arena_202610\\make_empty_labels.py

规则：
- 只看 JOBS 里列出的目录
- 每张 jpg 配一个同名 .txt
- 已经存在、且里面有内容的 txt 不覆盖（防止误清掉真标注）
- 末尾打印：新建几个、已是空文件几个、有内容跳过几个
"""
import os

JOBS = [
    r"D:\yolo_course\xiaosai_arena_202610\arena_box\row_clean\blanket",
]

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    for d in JOBS:
        if not os.path.isdir(d):
            print("目录不存在，跳过：", d)
            continue
        made = already = kept = 0
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(EXTS):
                continue
            txt = os.path.splitext(f)[0] + ".txt"
            p = os.path.join(d, txt)
            if not os.path.exists(p):
                open(p, "wb").close()
                made += 1
            elif os.path.getsize(p) == 0:
                already += 1
            else:
                kept += 1
        print("%s  新建 %d 个空 txt，原本就是空 %d 个，有内容没动 %d 个"
              % (d, made, already, kept))


if __name__ == "__main__":
    main()
