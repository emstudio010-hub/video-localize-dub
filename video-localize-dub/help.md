# video-localize-dub 使用说明

把一条（中文）短视频做成目标语言（默认英文）版本：
- 去掉画面里原来的字幕/标题（逐块选：修复 inpaint / 模糊 / 色块遮挡 / 保留）
- 用小米 MiMo TTS 重新配音：按说话人自动换音色，贴着原片说话时间，不重叠，语速均匀
- 用 Remotion 渲染新字幕：位置、角度、大小、动画尽量和原字幕一致，可做卡拉OK高亮
- 输出分辨率、帧率、帧数与原片一致，可选水印版

开始前 AI 会先问你：配音音色、风格、背景音乐处理、字幕字体/大小/颜色/位置/动画、哪几块字幕替换、
每块原字幕怎么去掉（修复/模糊/色块+颜色）、输出位置。你回答后它才会动手。

## 1. 安装（AI 自动完成）

你什么都不用装。第一次使用时 AI 会运行 `py -3.12 scripts/setup.py`，缺什么装什么：
FFmpeg、Node.js 18+（winget / brew / apt）、Python 包、LaMa 去字（并自动修复 Pillow）、demucs、
Remotion（约 300 MB）、NVIDIA 显卡的 CUDA 版 torch（约 2.5 GB），以及 whisper / LaMa / demucs 模型
（连不上 huggingface.co 时自动改用 hf-mirror.com 镜像）。新电脑第一次大约 5–30 分钟，主要是下载。
唯一需要你做的是把 MiMo API Key 发给 AI 一次，AI 会把它存进凭据管理器（见第 2 节）。

只需要你自己准备 Python 3.10+（推荐 3.12，https://www.python.org/ ）。

自动安装失败时的手动方式（备用）：

| 东西 | 必需？ | 手动安装 |
|---|---|---|
| FFmpeg（含 ffprobe，加进 PATH） | 必需 | https://ffmpeg.org/ 或 `winget install Gyan.FFmpeg` |
| Python 包 | 必需 | `py -3.12 scripts/env_check.py --install-required` |
| Node.js 18+ | 加字幕时必需 | https://nodejs.org/ |
| Remotion | 加字幕时必需 | 渲染时自动装到 `%LOCALAPPDATA%ideo-localize-dub
emotion-runtime` |
| demucs | 可选 | `py -3.12 -m pip install demucs`：保留原片背景音乐、去掉原人声 |
| LaMa 去字 | 可选，强烈推荐 | `py -3.12 -m pip install simple-lama-inpainting`，装完**必须**再执行 `py -3.12 -m pip install --force-reinstall --no-deps "pillow>=10"`（它会把 Pillow 降到 9.x，导致字幕测量时 Python 崩溃） |
| npm 下载慢（国内） | - | `npm config set registry https://registry.npmmirror.com` |

检查环境：`py -3.12 scripts/env_check.py`（最后一行 `RESULT: ready` 就行）。

### GPU 加速（自动，换电脑不用改配置）

`py -3.12 scripts/hw.py` 显示这台电脑会用什么。程序自动选：有 NVIDIA 显卡用 CUDA，Mac 用 MPS，Windows 上
AMD/Intel 显卡可试 DirectML（实验性），都没有就用 CPU 全部核心。GPU 出错会自动退回 CPU，不会让任务失败。

| 显卡 | 要额外装的（NVIDIA 由 setup.py 自动装；下表是手动备用） | 效果 |
|---|---|---|
| NVIDIA | CUDA 版 torch：`py -3.12 -m pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu124`（约 2.5 GB，装完再修一次 Pillow） | LaMa 约快 10–20 倍，demucs 也走 GPU |
| NVIDIA | `py -3.12 -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*` | 语音识别走 GPU |
| AMD / Intel（Windows） | `py -3.12 -m pip install torch-directml` | LaMa 可能变快，不保证，失败自动用 CPU |
| Mac（Apple 芯片） | 不用装 | LaMa / demucs 自动用 MPS |

想手动指定，在 job.json 的 `hardware` 里改：`device`（auto/cpu/cuda/mps/directml）、`encoder`
（默认 libx264 画质最好；`auto` 用显卡编码，更快但文件稍大）、`threads`、`remotion_concurrency`。
字体也会自动在 Windows / Mac / Linux 的字体目录里找。

OCR 用的是本地 RapidOCR，离线、免费，无需任何账号。
语音识别用本地 faster-whisper。国内下载模型慢/失败时先设置镜像：
`set HF_ENDPOINT=https://hf-mirror.com`（PowerShell：`$env:HF_ENDPOINT="https://hf-mirror.com"`）。

## 2. MiMo API Key（配音必需，AI 存进凭据管理器）

1. 打开 https://platform.xiaomimimo.com/ ，用小米账号登录，进入 API Keys 页面，点创建，复制 key（只显示一次）。
2. 把 key 发给 AI。AI 通过标准输入把它存进系统凭据库（Windows 凭据管理器，普通凭据 `Codex/XiaomiMiMoAPI`；
   Mac 钥匙串 / Linux 密钥环），再实际请求一次确认能用。AI 不会把 key 写进任何文件、命令参数或回复里。
3. 以后换 key：再发一次新的给 AI；删除：`py -3.12 scripts/mimo_key.py --delete`。
   检查：`py -3.12 scripts/mimo_key.py --check`（只显示存在哪），`--test` 试请求一次。
4. 不想经过 AI 的话，可以在自己的终端里运行 `py -3.12 scripts/mimo_key.py --store --console`（输入不显示），
   或设置环境变量 `XIAOMI_MIMO_API_KEY`（setup.py 会把它转存进凭据管理器）。

查找顺序：`XIAOMI_MIMO_API_KEY` → `MIMO_API_KEY` → 凭据管理器 / 钥匙串 `Codex/XiaomiMiMoAPI`。

费用：编写本说明时 MiMo 的 TTS（mimo-v2.5-tts）和 ASR（mimo-v2.5-asr）为限时免费，
**具体以官网为准**（https://platform.xiaomimimo.com/ ）。

不要把 key 写进脚本或 job.json。

## 3. 怎么用

在 Claude Code（或其它支持 skill 的 agent）里直接说，例如：

> 用 video-localize-dub 把 D:\videos\a.mp4 做成英文配音版，输出到 D:\out

AI 会依次：部署环境（setup.py）→ 分析视频（找出所有字幕块，生成 `analysis/ocr_blocks.png` 给你看；识别说话人，生成每个人的
试听片段）→ 问你配置 → 写英文台词 → 配音并对齐时间 → 去原字幕（先给前后对比图）→ 渲染新字幕 → 自检
（生成 `qc/` 检查图）→ 报告结果和仍然存在的问题。

所有设置都在任务文件夹的 `job.json` 里，想改哪里告诉 AI 即可，只会重跑受影响的步骤。

手动跑的顺序（脚本都在 `scripts/`，除 setup.py 外都需要 `--job <任务文件夹>`）：
```
setup.py
init_job.py --source <视频> --job <任务文件夹>
ocr_scan.py → analyze_audio.py → (separate_bgm.py)
编辑 job.json，写 analysis/script.json → check_script.py
tts.py → schedule_audio.py
remove_captions.py --preview → remove_captions.py
build_captions.py → render.py → verify.py
```
试听音色：`py -3.12 scripts/mimo_client.py --sample "Hello there" --voices Mia,Chloe,Milo,Dean --out 试听`

## 4. 选项速查

- 音色：英文 Mia（女，活泼）、Chloe（女，甜美）、Milo（男，阳光）、Dean（男，沉稳）；中文 冰糖、茉莉、苏打、白桦。
- 背景：`separate` 保留原背景音乐（需 demucs）/ `none` 纯配音 / `file` 用你给的音乐。
- 去字：`inpaint`（engine `lama` 最干净，`opencv` 快但复杂背景会糊）/ `blur` / `color`（色块，可选颜色、透明度、圆角）/ `keep`。
- 字幕位置：`match_block` 和原字幕同位置同角度 / `bottom` / `custom`。
- 字幕动画：`pop` / `fade` / `slide` / `none`；`highlight` 开启逐词高亮。

## 5. 常见问题

- **配音说太快/放不下**：schedule_audio 会列出超时的句子和超多少秒。缩短这些台词，或把风格改成更紧凑，再跑
  tts.py（只重做改过的句子）。速度只在 0.95–1.10 倍内调整，保证听起来均匀。
- **说话人识别错了**：男女声分得开，同性别的两个人可能被当成一个人。告诉 AI 实际人数（`--speakers N`）或直接指定
  哪句是谁。
- **去字后还有痕迹**：看 `preview/` 对比图。填补区里出现彩色细线或色块，多半是边缘残留了一圈描边/光晕，调大该块 `grow_px`（如 12–16）；淡色底板/光晕残留可调低该块 `diff_threshold`（如 20）；动画中漏掉的用
  `"mask": "box"`；装 LaMa 效果最好。去字是“重建”而不是还原，大字压在复杂运动背景上可能留下柔化区域，这时可改用
  模糊或色块。
- **LaMa 很慢**：耗时估算公式
  `分钟 ≈ (有字幕的帧数 × 每帧秒数 + 总帧数 × 0.01) ÷ 60`，每帧秒数：LaMa 约 2.9（CPU）、opencv 约 0.1、模糊/色块约 0.02。
  例：20 秒 30fps 全程有字 → (600 × 2.9 + 6) ÷ 60 ≈ 29 分钟（实测）；60 秒视频约 1.5 小时。
  程序开始时会打印 `Estimate`，运行中显示剩余时间。比较吃内存，内存不足时可能被系统中止，关掉其它大程序后再跑。
  中断了不用从头来：再执行同一条命令会从上次完成的 150 帧段接着做（`--restart` 从头开始）。有 NVIDIA 显卡
  装上 CUDA 版 torch 后每帧约 0.15–0.3 秒。
- **Python 在测量字幕时直接退出**：Pillow 被降级了，执行上面安装表里 Pillow 那条命令。
- **whisper 模型下载失败**：设置 HF 镜像（见第 1 节）；程序也会自动改用电脑里已缓存的模型。
- **找不到 Remotion**：render.py 会自动安装；`py -3.12 scripts/render.py --job <任务> --find` 看用的是哪个；
  在 job.json 的 `render.remotion_node_modules` 指定已有项目的 node_modules 也可以。

## 6. 限制

- 目标语言默认英文。MiMo 预置音色只有英文和中文；其它语言能读但可能有口音，不支持声音克隆。
- 没有口型同步。
- 说话人识别是声学聚类，同性别多人不可靠。
- OCR 对手写/极花哨字体或很小的字可能漏检，以 `ocr_blocks.png` 为准。
- 只处理你有权使用的素材。本工具做的是本地化，不提供任何“规避平台查重/伪原创”的功能。
