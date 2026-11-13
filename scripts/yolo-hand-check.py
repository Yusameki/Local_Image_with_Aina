# -*- coding: utf-8 -*-
"""
yolo-hand-check.py — 出图后用手部检测做一轮「防崩」筛查。

AI 出图最常见的结构性瑕疵是手：多指、断指、融合。人眼在缩略图上根本看不出来，
所以先让 YOLO 把「有几只手、分别在哪」框出来，再按框数裁局部图，逐张放大核对。

用法：
    python yolo-hand-check.py \
        --images  /path/to/ComfyUI/output/some_batch \
        --model   /path/to/models/ultralytics/bbox/hand_yolov8s.pt \
        --out     ./hand-report.json \
        --crops   ./hand-crops

输出：
  * JSON 报告：每张图的文件名、检出框数、每个框的 xyxy 与置信度
  * （可选）裁切图：文件名_hand<idx>.png，便于放大看

判据（经验值，不是硬规则）：
  * hands == 0  → 大概率手被藏起来了（构图合格里最常见的「安全解」），不用管
  * hands == 1  → 正常
  * hands >= 3  → 高度可疑：要么真的画了多只手，要么是「手 + 伞柄/头发」误检
  * 检出框数只是线索，**必须裁图肉眼确认**——实测过一批 5 张「可疑」全是误判。
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="图片目录")
    ap.add_argument("--model", required=True, help="手部检测权重 (.pt)")
    ap.add_argument("--out", default="hand-report.json")
    ap.add_argument("--crops", default=None, help="给定时裁切每个手部框")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--pad", type=float, default=0.15, help="裁切时向外扩的比例")
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(args.model)
    files = sorted(f for f in os.listdir(args.images) if f.lower().endswith(IMG_EXT))
    if args.crops:
        os.makedirs(args.crops, exist_ok=True)

    rows = []
    for name in files:
        path = os.path.join(args.images, name)
        result = model.predict(path, conf=args.conf, verbose=False)[0]
        boxes = []
        for b in result.boxes:
            xyxy = [round(float(v), 1) for v in b.xyxy[0].tolist()]
            boxes.append({"xyxy": xyxy, "conf": round(float(b.conf[0]), 3)})
        rows.append({"file": name, "hands": len(boxes), "boxes": boxes})
        print("%s  hands=%d" % (name, len(boxes)), flush=True)

        if args.crops and boxes:
            im = Image.open(path).convert("RGB")
            w, h = im.size
            for idx, box in enumerate(boxes):
                x0, y0, x1, y1 = box["xyxy"]
                bw, bh = x1 - x0, y1 - y0
                x0 = max(0, int(x0 - bw * args.pad))
                y0 = max(0, int(y0 - bh * args.pad))
                x1 = min(w, int(x1 + bw * args.pad))
                y1 = min(h, int(y1 + bh * args.pad))
                stem = os.path.splitext(name)[0]
                im.crop((x0, y0, x1, y1)).save(
                    os.path.join(args.crops, "%s_hand%d.png" % (stem, idx)))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    suspicious = [r["file"] for r in rows if r["hands"] >= 3]
    print("\n%d images, %d suspicious (hands>=3)" % (len(rows), len(suspicious)))
    if suspicious:
        print("需肉眼复核:", suspicious)


if __name__ == "__main__":
    main()
