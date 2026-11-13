<div align="center">

# 🎨 Local_Image_with_Aina

**SDXL 参数基线 · MOSS-Image 本地部署 · LoRA 炼丹实录**

一台 12 GB 显存的消费级主机，把 5B DiT + 9B 文本编码器跑起来，
再用自家出图训出一支能用的画风 LoRA。

![status](https://img.shields.io/badge/status-实测可复现-2ea44f)
![vram](https://img.shields.io/badge/VRAM-12GB-8957e5)
![comfyui](https://img.shields.io/badge/ComfyUI-v0.45.2-blue)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

</div>

---

## 📌 这是什么

一份「不是教程，是记录」的实战笔记。

网上关于 MOSS-Image 的资料，大多停在「装好插件、点一下」。
但真到自己动手，会遇到一串没人写的东西：插件其实要求新内核、
`--lowvram` 在默认状态下根本是空操作、量化文件的格式名和文件名对不上……

这个仓库把从零跑通的全过程摊开写：
**踩了什么坑、怎么验的、最后哪一档参数最好、为什么。**

> **Project_Aina 的第一步。**
> 这套「本地化工作流」接下来会继续往 **动画、音乐、游戏** 方向长——
> 先把图这一层彻底摸透、摸到能复现，再往下一个模态走。

---

## 🏆 两个核心结论

<table>
<tr>
<td width="50%" valign="top">

### ① 12 GB 跑得动 5B DiT

| 项 | 实测值 |
|---|---|
| 分辨率 | `832 × 1808` |
| 采样 | 50 步 / CFG 3 / euler + sgm_uniform |
| 单张耗时 | **52–60 秒** |
| 显存峰值 | **11.1 GB** / 12.3 GB |

关键是**分时**而不是**降级**：
int8 DiT 与 w4a8 文本编码器不同时驻留，编码器跑完立刻释放。

</td>
<td width="50%" valign="top">

### ② LoRA 的甜点区是 300 步

10 个 checkpoint × 3 档强度 × 2 场景 × 3 种子 = **180 张**对比图，
按 18 条长廊逐组盲选：

```
300 步  ██████████████  14/18  ← 甜点
1000 步 ██               2/18
200 步  █                 1/18
700 步  █                 1/18
```

**300 步 + 亮场景 0.8 / 暗场景 1.0** 是最终配方。

</td>
</tr>
</table>

---

## 🖼 效果

<div align="center">

| SDXL 基线（768×1664） | MOSS + 300 步 LoRA · 亮场景 |
|---|---|
| ![sdxl](assets/preview-01-sdxl.jpg) | ![seaside](assets/preview-02-seaside.jpg) |

| 纯 base 水彩（无 LoRA） | MOSS + 300 步 LoRA · 暗场景 |
|---|---|
| ![watercolor](assets/preview-03-watercolor.jpg) | ![dark](assets/preview-04-dark.jpg) |

</div>

四张图分别对应链路里的四个位置：
SDXL 基线 → MOSS 亮场景 → 纯 base 水彩线 → MOSS 暗场景。

---

## 🧭 整体链路

```
        ┌─────────────────────────────────────────────┐
        │            SDXL 栈（画风 & 数据来源）         │
        │  底模 + 画风 LoRA + 人物质感 LoRA             │
        │  CFG 5.0 · euler_ancestral · 28 步 · 768×1664│
        └──────────────────┬──────────────────────────┘
                           │  70 张精选出图
                           │  （Synthetic Data）
                           ▼
        ┌─────────────────────────────────────────────┐
        │         打标：只写内容，不写风格              │
        │  [画风触发词] + [主体/人数/景别/场景描述]      │
        └──────────────────┬──────────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────────┐
        │   KohakuMossTrainer · vram_12g 档            │
        │   1000 步 · 3.4 s/step · 58 min · 峰值 4.7G  │
        └──────────────────┬──────────────────────────┘
                           │  10 个 checkpoint
                           ▼
        ┌─────────────────────────────────────────────┐
        │      180 张矩阵 · 种子统一 · 18 组盲选        │
        │      → 300 步胜出（14/18）                    │
        └──────────────────┬──────────────────────────┘
                           │
                           ▼
        ┌─────────────────────────────────────────────┐
        │      MOSS 栈（出图）                         │
        │  int8 DiT + w4a8 TE + flux2-vae-small        │
        │  50 步 · CFG 3 · shift 5 · 832×1808          │
        └─────────────────────────────────────────────┘
```

> 这条链路里，**具体提示词与工作流 JSON 不在本仓库范围内**。
> 这里给的是结构与方法：什么参数决定什么、什么该交给 LoRA、什么该写进 caption。

---

## 📂 仓库结构

```
.
├── README.md                      # 你在这里
├── docs/
│   ├── 01_sdxl-params.md          # SDXL 参数基线（只有参数与逻辑）
│   ├── 02_moss-deployment.md      # MOSS 本地部署踩坑记
│   └── 03_moss-lora-training.md   # LoRA 炼丹实录（重头戏）
├── scripts/
│   ├── bench-llamacpp.mjs         # OpenAI 兼容端点流式测速
│   ├── moss-matrix-batch.py       # 批量跑参数矩阵 + 拼对照表
│   └── yolo-hand-check.py         # 手部检测防崩筛查
└── assets/                        # 效果图
```

---

## 🚀 快速开始

### 跑 MOSS-Image

```bash
# 1. 拿一份新内核（≥ v0.42.4，实测 v0.45.2）
git clone https://github.com/Comfy-Org/ComfyUI.git ComfyUI-MOSS
cd ComfyUI-MOSS && git checkout v0.45.2

# 2. 复用现有 torch，建一个不污染原环境的 venv
python -m venv --system-site-packages .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# 3. 下三个文件（约 12 GB），丢进对应目录
#    diffusion_models/moss_image_base_int8_convrot.safetensors
#    text_encoders/qwen3.5_9b_w4a8_gptq.safetensors
#    vae/flux2-vae_small.safetensors

# 4. 启动（不需要任何显存开关，它们都是空操作）
python main.py --port 8189 --use-sage-attention --cuda-malloc
```

节点链：`UNETLoader → CLIPLoader(type=moss_image) → VAELoader → KSampler`。
详见 [`docs/02_moss-deployment.md`](docs/02_moss-deployment.md)。

### 跑参数矩阵

```bash
python scripts/moss-matrix-batch.py \
  --ckpt-dir /path/to/runs/moss/mylora/checkpoints \
  --lora-dir /path/to/ComfyUI/models/loras \
  --image-dir /path/to/ComfyUI/output \
  --steps 100 200 300 400 500 600 700 800 900 1000 \
  --weights 0.6 0.8 1.0 \
  --seeds 4401 4402 4403
```

---

## 🕳 踩坑速查

<details>
<summary><b>展开：7 个花掉最多时间的坑</b></summary>

<br>

| 坑 | 真相 |
|---|---|
| 社区 MOSS 节点 | `manaigc/comfyui-moss-Image` **仓库不存在**；其余全是附加功能，没有一个是内核兼容层 |
| 官方插件 | 它自己 `import comfy.storage`，**老内核上直接 ImportError** → 装插件 ≈ 升级内核 |
| `--lowvram` | 翻开源码才发现：`Doesn't do anything if dynamic vram is enabled`，而 N 卡默认开着 |
| `--normalvram` | 这个参数**根本不存在**，报 unrecognized arguments |
| 量化文件名 | 名字写 `gptq`，文件里记的却是 `asym_w4a8_int8`，要读 `.comfy_quant` 才知道 |
| `KEEP_LAST` | 默认 3，跑 1000 步只留最后 3 个 checkpoint，中间全被删掉 |
| `warmup >= STEPS` | 跑 100 步验证时会直接被拒，要整体覆盖 SCHEDULE |

</details>

---

## 📚 三份文档怎么读

| 文档 | 适合谁 | 核心内容 |
|---|---|---|
| [SDXL 参数基线](docs/01_sdxl-params.md) | 想复现基线参数 | 参数表、为什么是这几个数、CFG 不能跨架构套用 |
| [MOSS 部署](docs/02_moss-deployment.md) | 想在自己机器上跑 | 版本门槛、量化校验、显存分时 |
| [LoRA 炼丹](docs/03_moss-lora-training.md) | 想训自己的画风 | Synthetic Data、打标哲学、矩阵盲选、三个反直觉发现 |

---

## 💡 最后一句

> **参数是死的，眼睛是活的。**

这一轮最贵的教训，不是哪个参数没调对，
而是两次「指标说得通、但图不好看」被眼睛纠正。
指标用来筛异常，判断交给眼睛。

<div align="center">

---

Made with too many late nights and 180 张对比图 (´-ω-`)ﾉ

</div>
