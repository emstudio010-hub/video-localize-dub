# video-localize-dub

**一个把中文短视频转换成英文版（英文配音 + 英文字幕）的 AI Agent Skill。**
作者是一名初三学生。

> 不会用？直接去 [Releases](https://github.com/emstudio010-hub/video-localize-dub/releases) 下载 `video-localize-dub.zip`，
> 解压后把文件夹交给你的 AI，跟它说「用 video-localize-dub 把我的视频做成英文版」就行，剩下的都由 AI 完成。

---

## 目录

- [它能做什么](#它能做什么)
- [3 分钟上手](#3-分钟上手)
- [完整流程](#完整流程)
- [API Key 配置](#api-key-配置)
- [可以调的选项](#可以调的选项)
- [硬件与耗时](#硬件与耗时)
- [文件结构](#文件结构)
- [常见问题](#常见问题)
- [限制](#限制)
- [开发背景](#开发背景)
- [共创者](#共创者)

---

## 它能做什么

输入一条中文短视频，输出同样分辨率、帧率、帧数的英文版视频：

| 步骤 | 做了什么 | 用到的技术 |
|---|---|---|
| 字幕识别 | 找出画面里所有字幕、标题、贴纸文字，不管在什么位置、横竖还是倾斜 | RapidOCR（本地、离线、免费） |
| 语音识别 | 识别中文语音和每个词的时间，区分不同说话人，生成每个人的试听片段 | faster-whisper（本地），可选 MiMo ASR 校对 |
| 翻译台词 | AI 按每句话的时长写自然的英文口语，并自动检查字数，保证念得完 | 你的 AI 本身 |
| 英文配音 | 按说话人分配音色，贴着原片说话时间放置，不重叠，语速只在 0.95–1.10 倍内微调，听起来均匀 | 小米 MiMo `mimo-v2.5-tts` |
| 背景音乐 | 去掉原人声、保留原背景音乐，配音时自动压低音乐（ducking） | demucs（本地） |
| 去除原字幕 | 每个字幕块单独选：修复 / 模糊 / 色块遮挡 / 保留；先出前后对比图给你确认 | LaMa 或 OpenCV inpaint |
| 英文字幕 | 位置、角度、大小、动画尽量和原字幕一致，支持卡拉 OK 式逐词高亮 | Remotion |
| 自检 | 检查尺寸/帧数/音频长度/配音重叠/字幕重叠/残留文字，生成检查图 | verify.py |

另外：

- **先确认再动手**：音色、配音风格、字幕字体/颜色/大小/动画、每块字幕怎么去，AI 都会先问你。
- **硬件自适应**：自动用 GPU 加速（NVIDIA / Apple 芯片 / AMD、Intel 实验性支持），没有 GPU 就用 CPU，换电脑不用改配置。
- **可以断点续跑**：去字中途被打断，再运行一次会从上次的位置接着做。

本项目只做视频本地化，**不包含去重、二创、规避平台查重之类的功能**。一方面这类需求部分模型会拒绝执行，另一方面也希望这个 skill 保持干净。

## 3 分钟上手

**你需要准备：**

- 一台 Windows 电脑（目前只在 Windows 上完整测试过）
- Python 3.10 或更高（推荐 3.12，下载：<https://www.python.org/>，安装时勾选 *Add to PATH*）
- 一个支持 skill 的 AI Agent，例如 Claude Code、Antigravity
- 一个 MiMo API Key（免费申请，见下文）

**步骤：**

1. 从 [Releases](https://github.com/emstudio010-hub/video-localize-dub/releases) 下载 `video-localize-dub.zip` 并解压。
2. 把解压出来的 `video-localize-dub` 文件夹放进 agent 的 skills 目录：
   - Claude Code：`C:\Users\<你的用户名>\.claude\skills\video-localize-dub`
   - 其它 agent：放到它读取 skill 的目录，或者直接告诉 AI 这个文件夹的位置
3. 对 AI 说：
   > 用 video-localize-dub 把 D:\videos\a.mp4 做成英文配音版，输出到 D:\out
4. AI 第一次会运行 `scripts/setup.py` 自动装环境，新电脑大约需要 5–30 分钟，主要是下载。
5. AI 向你要 MiMo API Key 时，把 key 发给它，它会存进 Windows 凭据管理器。
6. 回答 AI 的几个问题（音色、字幕样式等），然后等成品。

## 完整流程

```
setup（装环境）
  → init_job（建任务文件夹）
  → ocr_scan（找字幕块，生成总览图）+ analyze_audio（识别语音和说话人）
  → 询问你的选择（必做）
  → 写英文台词 + check_script（检查字数）
  → tts（配音）→ schedule_audio（对齐时间、混音）
  → remove_captions（先预览，再全片去字）
  → build_captions + render（渲染英文字幕、合成音轨）
  → verify（自检）→ 汇报结果和剩余问题
```

每一步的结果都保存在任务文件夹里，所有选择都写在 `job.json` 里。改了哪个选项，只重跑受影响的步骤。

## API Key 配置

1. 打开小米 MiMo 开放平台 <https://platform.xiaomimimo.com/>，用小米账号登录。
2. 进入 **API Keys** 页面，点 **创建**，复制 key（只显示一次）。
3. 把 key 交给 AI。AI 会把它存进 **Windows 凭据管理器**（普通凭据 `Codex/XiaomiMiMoAPI`），然后实际请求一次确认能用。key 不会写进任何文件或命令参数。

不想经过 AI 的话，可以在自己的终端里运行：

```
py -3.12 video-localize-dub/scripts/mimo_key.py --store --console   # 输入时不显示
py -3.12 video-localize-dub/scripts/mimo_key.py --check             # 查看 key 存在哪（不显示 key）
py -3.12 video-localize-dub/scripts/mimo_key.py --test              # 实际请求一次
py -3.12 video-localize-dub/scripts/mimo_key.py --delete            # 删除
```

也可以设置环境变量 `XIAOMI_MIMO_API_KEY`（或 `MIMO_API_KEY`），它的优先级高于凭据管理器。

MiMo 的 `mimo-v2.5-tts` 和 `mimo-v2.5-asr` 在编写时免费，具体以官网为准。

**想换成自己的模型？** TTS / ASR 的调用都集中在 `video-localize-dub/scripts/mimo_client.py`，让你的 AI 改这个文件接入别的接口即可。

## 可以调的选项

| 类别 | 选项 |
|---|---|
| 音色 | 英文 Mia（女，活泼）、Chloe（女，甜美）、Milo（男，阳光）、Dean（男，沉稳）；中文 冰糖、茉莉、苏打、白桦。每个说话人可以单独选 |
| 配音风格 | 一句话描述，例如「有活力的短视频口播」「平静的讲解」 |
| 背景声 | `separate` 保留原背景音乐 / `none` 纯配音 / `file` 用你给的音乐 |
| 去字方式 | `inpaint` 修复（`lama` 最干净，`opencv` 快）/ `blur` 模糊 / `color` 色块（可选颜色、透明度、圆角）/ `keep` 保留 |
| 字幕位置 | `match_block` 与原字幕同位置同角度 / `bottom` 底部 / `custom` 自定义 |
| 字幕样式 | 字体（默认 Impact）、大小（可匹配原字幕）、两行颜色、描边、阴影、底板 |
| 字幕动画 | `pop` / `fade` / `slide` / `none`，可开启逐词高亮 |
| 输出 | 输出目录、文件名、是否加水印（文字、角度、透明度） |

所有选项的说明都在 `config.template.json` 的 `_doc` 里。

## 硬件与耗时

`setup.py` 和各个脚本会自动检测硬件：

| 显卡 | 自动使用 |
|---|---|
| NVIDIA | CUDA 版 torch（LaMa、demucs），whisper 和 OCR 也可以走 GPU，setup.py 会自动安装 |
| Apple 芯片 | MPS |
| AMD / Intel（Windows） | DirectML（实验性，需要手动安装 torch-directml）；视频编码可用 AMF / QSV |
| 没有独立显卡 | CPU 全部核心 |

GPU 出错会自动退回 CPU，不会让任务失败。

**去字耗时估算：**

```
分钟 ≈ (有字幕的帧数 × 每帧秒数 + 总帧数 × 0.01) ÷ 60
每帧秒数：LaMa 约 2.9（CPU）/ 约 0.15–0.3（NVIDIA GPU）；opencv 约 0.1；模糊/色块约 0.02
```

例：20 秒、30fps、全程有字幕，用 LaMa 在 CPU 上约 29 分钟（实测）。60 秒的视频约 1.5 小时。时间太长可以改用 opencv、模糊或色块。

## 文件结构

```
video-localize-dub/
├── SKILL.md               # AI 执行的完整流程和规则
├── help.md                # 给人看的说明和常见问题
├── config.template.json   # 任务配置模板（每个选项都有说明）
├── remotion-template/     # 字幕渲染用的 Remotion 项目模板
└── scripts/
    ├── setup.py           # 一键部署：缺什么装什么
    ├── env_check.py       # 环境检查
    ├── hw.py              # 硬件检测与自动选择
    ├── mimo_key.py        # API Key 存取（凭据管理器）
    ├── mimo_client.py     # MiMo TTS / ASR 接口（换模型改这里）
    ├── init_job.py        # 新建任务
    ├── ocr_scan.py        # 字幕块识别
    ├── analyze_audio.py   # 语音识别 + 说话人
    ├── separate_bgm.py    # 人声 / 背景音乐分离
    ├── check_script.py    # 台词字数检查
    ├── tts.py             # 配音
    ├── schedule_audio.py  # 配音对齐与混音
    ├── remove_captions.py # 去除原字幕
    ├── build_captions.py  # 生成字幕卡片
    ├── render.py          # Remotion 渲染与合成
    └── verify.py          # 自检
```

## 常见问题

- **配音说太快、放不下？** schedule_audio 会列出超时的句子，让 AI 把这些台词改短再重新配音（只重做改过的句子）。
- **说话人识别错了？** 告诉 AI 实际有几个人，或者直接指定哪句是谁说的。
- **去字后有痕迹？** 看 `preview/` 里的对比图。可以调大该块的 `grow_px`，或改用 LaMa、模糊、色块。
- **国内下载模型失败？** setup.py 会自动切换到 hf-mirror.com 镜像；npm 慢可以设置 `npm config set registry https://registry.npmmirror.com`。
- **Python 在测量字幕时直接退出？** Pillow 被 LaMa 降级了，运行 `py -3.12 -m pip install --force-reinstall --no-deps "pillow>=10"`（setup.py 会自动处理）。

更多问题见 [`help.md`](video-localize-dub/help.md)。

## 限制

- 目前只在 Windows 上完整测试过。macOS 和 Linux 的代码已经写好，但没有实测。
- 去字是"重建"不是还原：大字压在复杂运动背景上，可能留下柔化区域。
- 说话人识别能可靠区分男女声，同性别的多人可能分不开。
- MiMo 预置音色只有英文和中文，不支持声音克隆。
- 没有口型同步。
- 请只处理你有权使用的素材。

## 推荐的 Agent

建议选反应快、擅长前端的模型。推荐 Antigravity、DeepSeek Harness、Claude Code（Opus 5.5）。

## 开发背景

这个项目起源于一次接单经历。我在一个群里接到一个需求：把中文视频转成英文配音版。报价是提供思路 299 元、做成 skill 500 元、集成成 Agent 智能体 799 元。对我来说这是很高的价格，在业内也算合理。

对方先发了一个视频让我们做 Demo。早期是我和 S. 一起想出了整体思路，然后分工：我负责音频部分（识别、翻译、配音），S. 负责视频部分（去字、字幕）。Demo 效果不错，语音、画面、字幕都对得很准。但对方看完说效果不达标，又换了一个视频，还发来他手里的成品让我们对照。我花了一天时间做出来，发现确实比不上那个成品：我做出来的去字蒙版面积很大，不好看。

后来我才发现两个视频根本不一样：他给我的原视频有两层字幕，他的成品对应的原视频只有一层。整个过程中我没有收过任何定金，想要一点 token 补贴成本，对方也不再回复。最后我决定不接这个单了，但已经做出来的东西不想浪费。后来我对整个项目做了大规模重构，并由我自己把完整流程从头到尾实践、打磨了一遍，整理完善后开源出来。

开发过程中我自己用掉了三个 Plus 的周额度，又在中转站花了将近两百元；S. 在做 Demo 时也用掉了 20 个 Plus 的周额度。

## 共创者

- **我**（[@emstudio010-hub](https://github.com/emstudio010-hub)）：与 S. 共同提出思路；早期负责音频部分（识别、翻译、配音）；后期主导整体重构，完成并实践了完整流程
- **S.**：与我共同提出思路；早期负责视频部分（去字、字幕）
- **Claude**（Anthropic，Opus 5.5）：skill 主体开发、脚本与文档
- **ChatGPT**（OpenAI，GPT6 Astra / 5.6 Sol）：开发协助
- **Antigravity**（Google，Gemini 3.8 Flash）：开发协助与方案改进

作为初三学生，做这个项目不容易。如果有需要改进的地方，欢迎提 [Issue](https://github.com/emstudio010-hub/video-localize-dub/issues)；如果它帮到了你，欢迎点个 Star 支持一下。

## License

MIT
