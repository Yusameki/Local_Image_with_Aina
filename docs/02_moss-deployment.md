# MOSS-Image 本地部署踩坑记

> 目标：在一张 **12 GB 显存**的消费级卡上，把 MOSS-Image（5B DiT + 9B 文本编码器）完整跑起来。
> 结论：能跑，`832 × 1808` 单张 50 步约 **52–60 秒**，显存峰值 **11.1 GB**。

---

## 0. 先说清楚它是什么

| 项 | 值 |
|---|---|
| 出方 | OpenMOSS（复旦）/ 模思智能，Apache-2.0 |
| 主干 | **5B 单流 DiT**，flow-matching，Muon 训练 |
| 文本编码器 | **Qwen3.5-9B**（中文/文字排版强的原因） |
| VAE | `flux2-vae-small` |
| 推荐采样 | **50 步 / CFG 3 / shift 5 / Euler + sgm_uniform** |
| 分辨率 | 宽高**必须是 16 的正整数倍**，推荐总像素 1M / 1.5M / 2M |

它跟 SDXL 是**两套体系**：SDXL 的 LoRA、CFG 5.0、v-pred 参数在这里全部作废。
唯一能带过来的是**画师 tag**（写法 `@[画师名]`），但那是语料印象，不是 LoRA 那种钉死的笔触。

---

## 1. 坑一：没有「装个插件就能用」这条路

社区里流传的几条路，实测全是死路：

| 路径 | 结果 |
|---|---|
| `manaigc/comfyui-moss-Image` 社区节点 | **仓库 404，不存在** |
| GitHub 上其他 MOSS 节点 | 全是「附加功能」（布局编辑器 / 漫画分格 / Apple MLX），**没有一个是让旧内核支持 MOSS 的兼容层** |
| 官方 `comfyui/moss-image/` 插件 | **它自己就要求新内核** |

第三条最反直觉。官方插件的 `loaders.py` 里有这么一行：

```python
from comfy import (conds, latent_formats, model_base, model_detection,
                   model_management, model_patcher, sd, storage, utils)
```

而 `comfy/storage.py` 是**新内核才有的文件**。
在老版本（例如 v0.30.2）上 import 它会直接 `ImportError`。

> **结论：装插件 ≈ 升级内核。既然都要升级，那不如走官方标注「推荐」的原生路径。**

---

## 2. 坑二：版本到底要升到哪一版

MOSS 支持是通过 PR #17095（commit `e087d8f6`，2026-10-16）合并进主干的，
但 **release 版 v0.41.3 里还没有**。

实测各 tag 是否包含三个关键文件：

```bash
git ls-tree -r v0.42.4 --name-only | grep -i moss
# comfy/ldm/moss_image/model.py
# comfy/text_encoders/moss_image.py
# comfy_extras/nodes_moss_image.py
```

| 版本 | 含 MOSS | 备注 |
|---|---|---|
| v0.41.3 | ❌ | 官方 README 明确说未包含 |
| **v0.42.4** | ✅ | 最低可用线，且已带 `comfy/storage.py` |
| v0.45.2 | ✅ | 本文实测版本 |

**推荐做法：clone 一份全新的 ComfyUI，checkout 到目标 tag，与日常用的整合包环境并存。**
不要在生产环境里原地升级——出问题时你会分不清是模型的问题还是环境的问题。

```bash
git clone https://github.com/Comfy-Org/ComfyUI.git /path/to/ComfyUI-MOSS
cd /path/to/ComfyUI-MOSS
git checkout v0.45.2
```

---

## 3. 下载：清单、镜像与校验

官方发布在 HuggingFace `Comfy-Org/MOSS-Image`，文件按 ComfyUI 的目录结构组织。

### 3.1 12 GB 显存的最优三件套

| 文件 | 大小 | 放到 |
|---|---|---|
| `moss_image_base_int8_convrot.safetensors` | 5.04 GiB | `models/diffusion_models/` |
| `qwen3.5_9b_w4a8_gptq.safetensors` | 6.84 GiB | `models/text_encoders/` |
| `flux2-vae_small.safetensors` | 0.23 GiB | `models/vae/` |

> ⚠️ 注意：**官方没有 W4A8 的 DiT**。
> `w4a8` 只有文本编码器那一支；DiT 只有 `bf16`（9.47 GiB）与 `int8_convrot`（5.04 GiB）两种。
> 网上说「W4A8 全家桶」的，多半是把两者混着说了。

### 3.2 国内镜像

直连 HF 慢或者要走代理时，把域名换掉即可，路径完全一致：

```
https://hf-mirror.com/Comfy-Org/MOSS-Image/resolve/main/diffusion_models/moss_image_base_int8_convrot.safetensors
https://hf-mirror.com/Comfy-Org/MOSS-Image/resolve/main/text_encoders/qwen3.5_9b_w4a8_gptq.safetensors
https://hf-mirror.com/Comfy-Org/MOSS-Image/resolve/main/vae/flux2-vae_small.safetensors
```

用 IDM 之类的多线程下载器时留意：**后缀必须还是 `.safetensors`**。
有些下载器会把文件名改成 `.bin` 或者一串乱码，那样子 ComfyUI 是认不出来的，要手动改回来。

### 3.3 三步校验（很重要）

**第一步：字节数。** 对照上表，差一个字节都不行。

**第二步：safetensors 头部。**

```python
import json, struct
with open(path, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
print(len(header) - 1, "tensors")   # 减去 __metadata__
```

**第三步：量化格式（这一步能救命）。**

新内核把量化信息写在一个叫 `.comfy_quant` 的 U8 张量里（内容是 JSON 字节）：

```python
raw = tensors[".comfy_quant"]          # 某层的
meta = json.loads(bytes(raw).decode())
# TE:  {"format": "asym_w4a8_int8", "group_size": 16, "convrot": True, ...}
# DiT: {"format": "int8_tensorwise", "convrot": True, ...}
```

拿这个 `format` 去对照内核 `comfy/quant_ops.py` 里的 `QUANT_ALGOS`：

```python
# float8_e4m3fn / float8_e5m2 / nvfp4 / mxfp8 /
# int8_tensorwise / convrot_w4a4 / asym_w4a8_int8 / w6a8_int8
```

**在列表里 = 这批文件能被加载。** 不在，就是文件下坏了或者版本不匹配。

> 顺带解释一个容易误解的点：文件名里写着 `gptq`，但文件内部记录的格式名其实是
> `asym_w4a8_int8`。`gptq` 只是**量化算法的出处**，不是运行时识别的键。
> 我们一开始也按文件名猜测，直到把元数据读出来才确认。

---

## 4. 坑三：`--lowvram` 根本不起作用

这是最浪费时间的一个坑。翻内核源码：

```python
# comfy/cli_args.py
--lowvram: Doesn't do anything if dynamic vram is enabled.
           If dynamic vram isn't being used this option makes the text
           encoders run on the CPU.
```

而 Nvidia 卡**默认开启 dynamic vram**：

```python
def enables_dynamic_vram():
    return (not args.disable_dynamic_vram and not args.highvram
            and not args.gpu_only and not args.novram and not args.cpu)
```

所以 `--lowvram` 在默认状态下是**空操作**。
另外，`--normalvram` 这个参数在新版里**根本不存在**（翻遍 vram_group 只有
`--gpu-only / --highvram / --lowvram / --novram / --cpu`）。

**那 12 GB 怎么跑得动？靠的是「分时」而不是「降级」：**

1. 用 int8 DiT + w4a8 文本编码器（两者不同时驻留）；
2. 文本编码器跑完就释放，把显存让给 DiT；
3. 分辨率控制在 1.5M 像素以内。

实测启动参数就这三个，一个显存开关都没加：

```bash
python main.py --port 8189 --use-sage-attention --cuda-malloc
```

显存曲线（832×1808 / 50 步）：

```
提交前   2045 MiB
峰值    11140 MiB  (90.7% of 12288)
回落    10723 MiB
```

余量只有 800 MB 左右。所以：**不要训练和出图同时跑**，训练占 3–5 GB，加起来必爆。

---

## 5. 环境隔离：复用 torch，但别污染原环境

torch 这类大包重装一次要几个 GB、十几分钟，没必要。用 `--system-site-packages`
建一个「共享依赖」的 venv：

```bash
/path/to/base/python.exe -m venv --system-site-packages /path/to/ComfyUI-MOSS/.venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

这样 torch / torchvision / sageattention 直接复用基础环境，
只有新版前端包（`comfyui-frontend-package` 等）装进 venv，**不会覆盖原环境**。

> 国内装包时如果卡在某个包「找不到版本」，先换镜像源再排查。
> 实测清华源缺 `comfyui-workflow-templates-media-assets-02`，换阿里云源一次过。

---

## 6. 模型目录：一份模型，两套环境共用

不复制模型，用 `extra_model_paths.yaml` 挂载：

```yaml
comfyui_models:
  base_path: /path/to/models/
  is_default: true
  checkpoints: checkpoints/
  diffusion_models: diffusion_models/
  text_encoders: text_encoders/
  vae: vae/
  loras: loras/
  # ... 其余子目录照抄
```

`is_default: true` 的意思是「用这份配置**替换**内置默认路径」，而不是追加。
好处是 SDXL 那套模型和 MOSS 这套模型放在同一个目录树里，两边环境都能看见。

---

## 7. 原生节点链

```
UNETLoader (moss_image_base_int8_convrot, weight_dtype=default)
      │
CLIPLoader (qwen3.5_9b_w4a8_gptq, type=moss_image)
      │
VAELoader (flux2-vae_small)
      │
CLIPTextEncode ×2 (positive / negative)
      │
EmptyLatentImage (832 × 1808)
      │
KSampler (euler / sgm_uniform / 50 steps / cfg 3 / shift 5 默认)
      │
VAEDecode → SaveImage
```

几个容易踩的点：

- `CLIPLoader` 的 `type` 必须选 **`moss_image`**（下拉框里在 `qwen_image` 附近）；
- **不要**沿用插件工作流里的 `ModelSamplingSD3`——那是旧文档；
- 想调 shift 用 `ModelSamplingAuraFlow(multiplier=1)`，不是改 KSampler；
- 原生的调度器**不含训练时的网格吸附**，所以「同参数同种子」不保证和官方 demo 出同一张图。

---

## 8. 踩坑速查表

| 症状 | 原因 | 修法 |
|---|---|---|
| `ModuleNotFoundError: comfy.storage` | 内核版本太老 | 升到 ≥ v0.42.4 |
| 加载模型报量化格式不支持 | 文件没下全 / 版本不认 | 读 `.comfy_quant` 里的 `format` 去对 `QUANT_ALGOS` |
| 加了 `--lowvram` 没变化 | dynamic vram 默认开启，参数被忽略 | 别加，改用分时 + 降分辨率 |
| `unrecognized arguments: --normalvram` | 这个参数根本不存在 | 删掉 |
| 第一张就 OOM | 训练进程还占着显存 | 先关训练再出图 |
| 出图和官方 demo 构图不一样 | 原生 scheduler 无网格吸附 | 正常现象，不是 bug |
| 单张要 300 秒 | 文本编码器没走 GPU / 走了 fp32 | 检查 `type=moss_image` 与 `weight_dtype` |

---

## 9. 一句话总结

> **MOSS 的门槛不在显存，在版本和量化格式。**
> 版本对上（v0.42.4+）、量化格式认得（`asym_w4a8_int8`）、显存分时用，
> 一张 12 GB 的卡就足够把它跑起来。
