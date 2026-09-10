Manga-Translation-MCP

基于 MCP（Model Context Protocol）的自动化漫画翻译服务器。通过组合 YOLO 文字区域检测与 GLM-OCR 光学字符识别，自动完成漫画中文字区域的定位、文字提取、翻译请求编排，并将译文回填到漫画图像中，（注意！后端使用的模型必须支持多模态推理，否则无法识别图像。）

传统的自动漫画翻译方案通常是一条单向流水线：文字检测 → OCR → 逐句翻译 → 回填。信息在这条链路上每经过一个环节就衰减一次——检测框偏一点，OCR 错一个字，翻译时又丢失了画面语境，最终译文与原始漫画的语义、语气、排版意图渐行渐远。更致命的是，传统方案把 OCR 文本当作翻译的唯一输入，而 OCR 本身就是一个有损过程：拟声词、手写体、特殊字体、气泡内的视觉语气，在变成纯文本的那一刻就已经被磨平了。

本项目的核心哲学截然不同：OCR 只用于定位和匹配，翻译交给能同时看到整页图像的 LLM。每一页漫画的完整视觉信息——人物表情、字体大小、拟声词形态、背景氛围、对话气泡的视觉语气——都会以 Base64 原图形式直接送入多模态大模型，由它在完整上下文中完成翻译。这意味着：没有信息磨损。从原图到译文，中间不经过任何有损压缩的“文字中转站”。LLM 看到的就是读者看到的，它翻译的也是读者感受到的。


⚠️ 重要提示：本项目采用 MCP 自我调用架构——yolo_server.py 在运行过程中会通过 other_server 的 sendRequest 向 LLM 持续发送翻译请求。由于 MCP 协议原生不支持在服务端维护完整的对话历史，本 MCP 必须搭配 open-MCPclient 使用，由后者提供多轮对话历史管理能力。请务必阅读下文「与 open-MCPclient 配合使用」章节。

核心特性
两阶段文字处理流程：先用 RT-DETR 检测漫画中的文字区域（对话气泡内文字 + 自由文字），再对每个区域裁剪后送入 GLM-OCR 进行文字识别。

完整的翻译编排：自动将每页图像 Base64 编码后逐页发送给 LLM，遵循预设的翻译规则、核查流程和 JSON 格式规范。

译文回填：根据匹配结果将中文译文以自适应字号、竖排/横排布局绘制回漫画图像，对话气泡内文字自动填充白色背景，自由文字使用半透明遮罩。

显存释放机制：文字检测完成后主动释放模型并清理 CUDA 缓存，避免 OCR 阶段显存不足。

MCP 工具接口：对外暴露 comic_translate_process 工具，通过 MCP 协议接收漫画目录和输出目录两个参数即可启动全流程。

环境要求
Python 3.10+

PyTorch（建议 CUDA 11.8+）

transformers ≥ 4.45

Pillow、pydantic

mcp 包（pip install mcp）

bash
pip install torch transformers pillow pydantic mcp
模型准备（必须完成）
本项目使用两个 HuggingFace 模型，需要手动下载并放置到指定路径，否则 yolo_server.py 将无法加载模型。

模型 1：GLM-OCR（OCR 识别模型）
项目	说明
HuggingFace 仓库	zai-org/GLM-OCR
架构	GLM-V encoder–decoder，约 0.9B 参数
用途	对裁剪后的漫画文字区域进行文字识别
代码中的期望路径	C:\Comic\GLM-OCR
下载方式：

bash
# 方式一：huggingface-cli
huggingface-cli download zai-org/GLM-OCR --local-dir C:\Comic\GLM-OCR

# 方式二：git clone（需安装 git-lfs）
git lfs install
git clone https://huggingface.co/zai-org/GLM-OCR C:\Comic\GLM-OCR
路径修改：如果不想放在 C:\Comic\GLM-OCR，请同步修改 yolo_server.py 中的 ocr_model_path 变量。

模型 2：comic-text-and-bubble-detector（文字区域检测模型）
项目	说明
HuggingFace 仓库	ogkalu/comic-text-and-bubble-detector（推荐）或 Sundowner123/comic-text-and-bubble-detector
架构	RT-DETR-v2 r50vd，约 42.9M 参数
检测类别	text_bubble（气泡内文字）、text_free（自由文字）、bubble（空对话框，代码中已跳过）
代码中的期望路径	C:\Comic\comic-text-and-bubble-detector
bash
# 推荐使用 ogkalu 原始版本（更完整）
huggingface-cli download ogkalu/comic-text-and-bubble-detector --local-dir C:\Comic\comic-text-and-bubble-detector

# 或使用 Sundowner123 的 ONNX 精简版（注意：代码使用的是 transformers 加载方式，建议优先选 ogkalu 版本）
同样，如需修改路径，请调整 yolo_server.py 中的 text_recognition_model_path 变量。

模型放置检查清单
□ C:\Comic\GLM-OCR 目录下存在 config.json、model.safetensors（或分片文件）及 tokenizer 相关文件
□ C:\Comic\comic-text-and-bubble-detector 目录下存在 config.json、model.safetensors 及 preprocessor_config.json
□ 两个目录均不是空目录或只有 .git 文件夹（若使用 git clone 后未拉取 LFS 文件，请执行 git lfs pull）
与 open-MCPclient 配合使用
为什么需要 open-MCPclient？
本项目的核心工作流依赖 MCP 服务端的自我调用：yolo_server.py 中的 start_translate_comic 函数会多次调用 other_server.sendRequest()，向 LLM 发送翻译请求、核查指令和 JSON 格式规范。然而，标准 MCP 协议在服务端并不维护对话历史，每次 sendRequest 都是无状态的。

open-MCPclient 是一个多 LLM 提供商的 MCP 客户端框架，内置了对话历史管理能力（默认保留最近 20 条消息）。通过它作为中间层，可以确保 sendRequest 发出的请求能够携带完整的历史上下文，使 LLM 理解当前翻译进度、上一页内容以及已建立的翻译规则。

配置步骤
第一步：将服务文件放入 open-MCPclient

将本仓库的 yolo_server.py 复制到 open-MCPclient 的项目根目录（或 open-MCPclient 的 MCP 服务器扫描目录中）：

text
Open-MCP-Client/
├── ChatMCP.py
├── servers_config.json
├── yolo_server.py          ← 放入此处
├── other_server.py         ← 确保依赖文件也在同目录
└── ...
第二步：将资源文件放入正确位置

本项目的 start_translate_comic 函数会读取多个规则文件，这些文件必须放在 C:\mcp_server\resources\ 目录下：

文件	用途
comic_translation_rules.md	漫画翻译规则（翻译风格、术语表等）
json格式1.md	翻译结果的 JSON 输出格式定义
核查流程.md	翻译结果正确性核查规则
匹配流程.md	将翻译文本与检测框进行匹配的规则
json格式2.md	匹配后 Lettering.json 的格式定义
请创建目录并放入对应文件：

bash
mkdir C:\mcp_server\resources
# 将上述 5 个 .md 文件复制到该目录
如果你希望将这些规则文件放在其他位置，请同步修改 yolo_server.py 中所有 open(r"C:\mcp_server\resources\...") 的硬编码路径。

第三步：在 open-MCPclient 中注册本 MCP 服务器

编辑 open-MCPclient 的 servers_config.json，在 mcpServers 中添加：

json
{
  "mcpServers": {
    "manga-translation": {
      "command": "python",
      "args": ["yolo_server.py"],
      "env": {}
    }
  }
}
第四步：确保 other_server 可用

yolo_server.py 依赖 other_server.py 中的 sendRequest 和 clear_history。请确认该文件存在于同一 Python 路径下，且 sendRequest 函数会实际调用 open-MCPclient 提供的 LLM 接口（而非直接调用某个模型的本地 API）。

项目结构
text
server/
├── yolo_server.py              # MCP 服务主入口
├── other_server.py             # LLM 请求发送与历史管理适配层
resources/
│   ├── comic_translation_rules.md
│   ├── json格式1.md
│   ├── 核查流程.md
│   ├── 匹配流程.md
│   └── json格式2.md
└── README.md
使用方式
在 open-MCPclient 的对话界面中，调用本 MCP 提供的工具：

text
comic_translate_process(
    comic_dir="D:/comics/my_manga_chapter_01",
    output_dir="D:/comics/translated_output"
)
调用后，服务端会启动一个后台线程执行完整流程，MCP 工具立即返回提示信息。翻译完成后，output_dir 中将生成：

boxes.json — 文字区域检测结果（每页的 box 坐标和识别文本）

translation.json — LLM 翻译结果

Lettering.json — 翻译文本与检测框匹配后的结果

与原始漫画同名的已翻译图像文件

翻译流程详解
文字区域检测 + OCR → 输出 boxes.json

释放检测模型 → 清理显存，为 OCR 模型腾出空间

逐页提交翻译 → 将漫画图像 Base64 编码后逐页发送给 LLM

合并与核查 → 按 json格式1.md 合并翻译结果，按 核查流程.md 核查

匹配 → 读取 translation.json 和 boxes.json，按 匹配流程.md 将翻译文本绑定到具体检测框

Lettering → 按 json格式2.md 输出 Lettering.json，调用 apply_comic_lettering 将译文绘制回图像