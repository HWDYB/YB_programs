"""
把 arena_box 整理成 YOLO 能直接用的数据集（只搬文件，不写 data.yaml）

输入：
    arena_box\\label\\box_label.zip / box_label1.zip   他标的 402 个框（class 0）
    arena_box\\row_clean\\box\\*.jpg                    抹掉黄字的 402 张正样本
    arena_box\\row_clean\\blanket\\*.jpg                69 张负样本（配 0 字节同名 txt）

输出：
    arena_box_ds\\images\\{train,val,test}
    arena_box_ds\\labels\\{train,val,test}

划分规则：
    正样本按序号分段：000-299 train / 300-349 val / 350-401 test
    负样本按排序后取模散开：i%9==0 -> test，i%9==4 -> val，其余 train
    全程只用复制，不动源文件；输出目录若已存在会先清空（跑之前会打印要删的路径）

用法：
    D:\\PythonProjects\\python.exe D:\\yolo_course\\xiaosai_arena_202610\\build_box_dataset.py
"""
import os
import shutil
import zipfile

BASE = r"D:\yolo_course\xiaosai_arena_202610\arena_box"
SRC_BOX_IMG = os.path.join(BASE, "row_clean", "box")
SRC_NEG_IMG = os.path.join(BASE, "row_clean", "blanket")
ZIPS = [os.path.join(BASE, "label", "box_label.zip"),
        os.path.join(BASE, "label", "box_label1.zip")]
LBL_TMP = os.path.join(BASE, "labels_box")
OUT = r"D:\yolo_course\xiaosai_arena_202610\arena_box_ds"
EXTS = (".jpg", ".jpeg", ".png", ".bmp")

BOX_SPLIT = (("train", 0, 299), ("val", 300, 349), ("test", 350, 401))


def clean_dir(p):
    if os.path.isdir(p):
        print("清空已有输出目录：", p)
        shutil.rmtree(p)


def main():
    # 1. 解压标签
    os.makedirs(LBL_TMP, exist_ok=True)
    for z in ZIPS:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(LBL_TMP)
    labels = {os.path.splitext(f)[0]: os.path.join(LBL_TMP, f)
              for f in os.listdir(LBL_TMP) if f.endswith(".txt")}
    print("标签文件 %d 个" % len(labels))

    imgs = {os.path.splitext(f)[0]: f for f in os.listdir(SRC_BOX_IMG)
            if f.lower().endswith(EXTS)}
    print("正样本图 %d 张" % len(imgs))
    miss_img = sorted(set(labels) - set(imgs))
    miss_lbl = sorted(set(imgs) - set(labels))
    print("有标签没图：%d  %s" % (len(miss_img), miss_img[:5]))
    print("有图没标签：%d  %s" % (len(miss_lbl), miss_lbl[:5]))

    negs = sorted(f for f in os.listdir(SRC_NEG_IMG) if f.lower().endswith(EXTS))
    print("负样本图 %d 张" % len(negs))
    neg_missing_txt = [f for f in negs
                       if not os.path.exists(os.path.join(SRC_NEG_IMG, os.path.splitext(f)[0] + ".txt"))]
    print("负样本缺空 txt：%d  %s" % (len(neg_missing_txt), neg_missing_txt[:5]))

    clean_dir(OUT)
    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(OUT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", split), exist_ok=True)

    def put(split, img_path, lbl_path):
        name = os.path.basename(img_path)
        dst_img = os.path.join(OUT, "images", split, name)
        dst_lbl = os.path.join(OUT, "labels", split, os.path.splitext(name)[0] + ".txt")
        if os.path.exists(dst_img):
            print("重名，跳过：", name)
            return False
        shutil.copy2(img_path, dst_img)
        shutil.copy2(lbl_path, dst_lbl)
        return True

    # 2. 正样本按序号分段
    for split, lo, hi in BOX_SPLIT:
        n = 0
        for stem in sorted(imgs, key=lambda s: int(s) if s.isdigit() else 10 ** 9):
            if not stem.isdigit():
                continue
            idx = int(stem)
            if lo <= idx <= hi:
                n += put(split, os.path.join(SRC_BOX_IMG, imgs[stem]), labels[stem])
        print("%s 正样本 %d 张" % (split, n))

    # 3. 负样本取模散开
    for i, f in enumerate(negs):
        split = "test" if i % 9 == 0 else ("val" if i % 9 == 4 else "train")
        put(split, os.path.join(SRC_NEG_IMG, f),
            os.path.join(SRC_NEG_IMG, os.path.splitext(f)[0] + ".txt"))

    # 4. 自检
    print("---- 自检 ----")
    total_img = total_lbl = 0
    for split in ("train", "val", "test"):
        di = os.path.join(OUT, "images", split)
        dl = os.path.join(OUT, "labels", split)
        ii = sorted(f for f in os.listdir(di) if f.lower().endswith(EXTS))
        ll = sorted(f for f in os.listdir(dl) if f.endswith(".txt"))
        stem_i = {os.path.splitext(f)[0] for f in ii}
        stem_l = {os.path.splitext(f)[0] for f in ll}
        empty = sum(1 for f in ll if os.path.getsize(os.path.join(dl, f)) == 0)
        onebox = sum(1 for f in ll if os.path.getsize(os.path.join(dl, f)) > 0)
        print("%-5s 图 %3d  标签 %3d  配对差集 %d  空标签(负样本) %3d  有框 %3d"
              % (split, len(ii), len(ll), len(stem_i ^ stem_l), empty, onebox))
        total_img += len(ii); total_lbl += len(ll)
    print("合计 图 %d  标签 %d" % (total_img, total_lbl))


if __name__ == "__main__":
    main()
