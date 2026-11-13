# -*- coding: utf-8 -*-
"""
moss-matrix-batch.py — 批量跑 LoRA checkpoint × 强度 × 场景 × 种子矩阵，并拼对照图。

用途：训练出多个步数的 LoRA checkpoint 后，用同一套提示词 + 同一批种子横向对比，
让「哪一档步数最好」这件事变成可复现的批量实验，而不是靠记忆抽卡。

用法：
    python moss-matrix-batch.py \
        --host http://127.0.0.1:8188 \
        --ckpt-dir  /path/to/runs/moss/aina/checkpoints \
        --lora-dir  /path/to/ComfyUI/models/loras \
        --out-dir   ./out \
        --steps 100 200 300 400 500 600 700 800 900 1000 \
        --weights 0.6 0.8 1.0 \
        --seeds 4401 4402 4403

前置：
  ComfyUI 已启动，且 models/ 下有以下三个文件（Comfy-Org/MOSS-Image 发布）：
    diffusion_models/moss_image_base_int8_convrot.safetensors
    text_encoders/qwen3.5_9b_w4a8_gptq.safetensors
    vae/flux2-vae_small.safetensors
  LoRA 使用 ComfyUI 单文件格式（KohakuMossTrainer 的 export.py 产出即是）。

注意：
  * 单张 832x1808 / 50 步 / CFG 3 在 12GB 卡上约 50-60 秒，180 张约 2.5-3 小时。
  * 训练与出图不要并行：训练占 3-5GB + 出图峰值 11GB，12GB 卡会直接 OOM。
  * 每 4 秒轮询一次 /history；OOM 时调 /free 清一次缓存后重试，仍失败则跳过。
"""
import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

NEG = ("nsfw, worst quality, bad anatomy, watermark, jpeg artifact, signature, text, "
       "fused fingers, extra digits, oversaturated, blurry")

# 场景提示词由命令行传入，仓库里不放具体 Prompt。
# 建议两个场景做「极端对照」：一个高亮大面积白（测过曝），一个几乎全黑（测人物是否没入黑暗）。
SCENES = []  # [(scene_key, positive_prompt), ...]，由 main() 填充


def build_graph(pos, neg, lora, strength, seed, width, height, prefix):
    """原生 MOSS-Image 节点链：UNETLoader → LoraLoader → KSampler。"""
    return {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "moss_image_base_int8_convrot.safetensors",
                         "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": "qwen3.5_9b_w4a8_gptq.safetensors", "type": "moss_image"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": "flux2-vae_small.safetensors"}},
        "4": {"class_type": "LoraLoader",
              "inputs": {"model": ["1", 0], "clip": ["2", 0], "lora_name": lora,
                         "strength_model": strength, "strength_clip": 0.0}},
        "4b": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["4", 1]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["4", 1]}},
        "6": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler",
              "inputs": {"model": ["4", 0], "seed": seed, "steps": 50, "cfg": 3.0,
                         "sampler_name": "euler", "scheduler": "sgm_uniform",
                         "positive": ["4b", 0], "negative": ["5", 0],
                         "latent_image": ["6", 0], "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["8", 0]}},
    }


class Client:
    def __init__(self, host):
        self.host = host.rstrip("/")

    def post(self, path, payload):
        req = urllib.request.Request(
            self.host + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def get(self, path):
        with urllib.request.urlopen(self.host + path, timeout=60) as r:
            return json.loads(r.read().decode())


def run_one(cli, tag, graph, timeout=420, log=print):
    """提交一张图；OOM 时清一次缓存重试。返回 (ok, filename)。"""
    for attempt in (1, 2):
        try:
            pid = cli.post("/prompt", {"prompt": graph, "client_id": "matrix-batch"})["prompt_id"]
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:600]
            log("[%s] submit failed HTTP %s: %s" % (tag, e.code, body))
            if attempt == 1 and ("out of memory" in body.lower() or e.code == 500):
                cli.post("/free", {"unload_models": False, "free_memory": True})
                time.sleep(4)
                continue
            return False, None

        t0 = time.time()
        while True:
            time.sleep(4)
            elapsed = int(time.time() - t0)
            try:
                hist = cli.get("/history/" + pid)
            except Exception:
                continue
            if pid not in hist:
                if elapsed > timeout:
                    log("[%s] timeout %ds, skip" % (tag, elapsed))
                    return False, None
                continue
            status = hist[pid].get("status", {})
            ok = status.get("status_str") == "success"
            fname = None
            for _, out in hist[pid].get("outputs", {}).items():
                for img in out.get("images", []):
                    fname = img.get("filename")
            if ok:
                log("[%s] OK %ds -> %s" % (tag, elapsed, fname))
                return True, fname
            msgs = " ".join(str(m) for m in status.get("messages", []))
            log("[%s] FAIL %ds: %s" % (tag, elapsed, msgs[:400]))
            if attempt == 1 and ("out of memory" in msgs.lower() or "allocation" in msgs.lower()):
                cli.post("/free", {"unload_models": False, "free_memory": True})
                time.sleep(4)
                break
            return False, None
    return False, None


def stage_loras(ckpt_dir, lora_dir, steps, log=print):
    """把每个 step_XXXX/export/aina-comfy.safetensors 拷成 lora-<step>.safetensors。"""
    ok = True
    for s in steps:
        src = os.path.join(ckpt_dir, "step_%08d" % s, "export", "aina-comfy.safetensors")
        dst = os.path.join(lora_dir, "lora-%04d.safetensors" % s)
        if not os.path.exists(src):
            log("!! missing checkpoint: %s" % src)
            ok = False
            continue
        size = os.path.getsize(src)
        if not os.path.exists(dst) or os.path.getsize(dst) != size:
            shutil.copy2(src, dst)
            log("staged %s (%.0f MB)" % (dst, size / 1048576))
    return ok


def build_sheets(results, out_dir, image_dir, steps, weights, seeds, size, log=print):
    """每个场景一张总表：行=步数，列=强度×种子。"""
    from PIL import Image, ImageDraw
    w, h = size
    for scene_key, _ in SCENES:
        tw = 240
        th = int(tw * h / w)
        pad, lab_w, head_h = 4, 96, 56
        cols = len(weights) * len(seeds)
        sheet = Image.new("RGB", (lab_w + cols * (tw + pad) + pad,
                                  head_h + len(steps) * (th + pad) + pad), (26, 26, 30))
        d = ImageDraw.Draw(sheet)
        d.text((10, 18), "%s  (%dx%d)" % (scene_key.upper(), w, h), fill=(240, 240, 240))
        col = 0
        for wt in weights:
            for sd in seeds:
                d.text((lab_w + col * (tw + pad) + 4, 36), "w%.1f s%d" % (wt, sd), fill=(200, 200, 200))
                col += 1
        for ri, step in enumerate(steps):
            y = head_h + ri * (th + pad)
            d.text((10, y + th // 2), "%4d" % step, fill=(255, 220, 140))
            col = 0
            for wt in weights:
                for sd in seeds:
                    fn = results.get((scene_key, step, wt, sd))
                    x = lab_w + col * (tw + pad)
                    if fn and os.path.exists(os.path.join(image_dir, fn)):
                        im = Image.open(os.path.join(image_dir, fn)).convert("RGB")
                        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y))
                    else:
                        d.rectangle([x, y, x + tw, y + th], outline=(90, 40, 40))
                        d.text((x + 8, y + th // 2), "MISS", fill=(200, 90, 90))
                    col += 1
        out = os.path.join(out_dir, "matrix_%s.png" % scene_key)
        sheet.save(out, optimize=True)
        log("sheet -> %s" % out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8188")
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--lora-dir", required=True)
    ap.add_argument("--image-dir", required=True, help="ComfyUI output 目录")
    ap.add_argument("--out-dir", default="./out")
    ap.add_argument("--steps", nargs="+", type=int,
                    default=[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    ap.add_argument("--weights", nargs="+", type=float, default=[0.6, 0.8, 1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[4401, 4402, 4403])
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=1808)
    ap.add_argument("--skip-stage", action="store_true")
    ap.add_argument("--scene-a", default="bright", help="场景A 名称")
    ap.add_argument("--scene-b", default="dark", help="场景B 名称")
    ap.add_argument("--prompt-a", required=True, help="场景A 正向提示词")
    ap.add_argument("--prompt-b", required=True, help="场景B 正向提示词")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    SCENES.clear()
    SCENES.extend([(args.scene_a, args.prompt_a), (args.scene_b, args.prompt_b)])
    cli = Client(args.host)
    if not args.skip_stage and not stage_loras(args.ckpt_dir, args.lora_dir, args.steps):
        print("!! LoRA 文件不全，中止")
        return

    total = len(args.steps) * len(args.weights) * len(SCENES) * len(args.seeds)
    print("planned %d images" % total)
    results, done, failed = {}, 0, []
    t0 = time.time()

    for scene_key, pos in SCENES:
        for step in args.steps:
            lora = "lora-%04d.safetensors" % step
            for wt in args.weights:
                for sd in args.seeds:
                    tag = "%s/%04d/w%.1f/s%d" % (scene_key, step, wt, sd)
                    prefix = "mx_%s_%04d_w%s_s%d" % (scene_key, step, str(wt).replace(".", ""), sd)
                    graph = build_graph(pos, NEG, lora, wt, sd, args.width, args.height, prefix)
                    ok, fname = run_one(cli, tag, graph)
                    if ok:
                        results[(scene_key, step, wt, sd)] = fname
                    else:
                        failed.append(tag)
                    done += 1
                    print("  %d/%d  (%.1f min)" % (done, total, (time.time() - t0) / 60), flush=True)

    build_sheets(results, args.out_dir, args.image_dir, args.steps, args.weights, args.seeds,
                 (args.width, args.height))
    print("done: %d ok, %d failed" % (len(results), len(failed)))
    if failed:
        print("failed:", failed)


if __name__ == "__main__":
    main()
