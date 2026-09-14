# Manga-Translation-MCP
基于 MCP（Model Context Protocol）的自动化漫画翻译服务器。通过组合 RT-DETR 文字区域检测与 GLM-OCR 光学字符识别，自动完成漫画中文字区域的定位、文字提取、翻译请求编排，并将译文回填到漫画图像中。

⚠️ 重要提示：

1.后端使用的模型必须支持多模态推理，否则无法识别图像。

2.yolo_server.py 在运行过程中会通过 mcp_rpc_utils 的 sendRequest 向 LLM 持续发送翻译请求。由于 市面大多数agent 原生不支持流程化的自我调用，本 MCP 必须搭配 open-MCPclient 使用，由后者提供严格的对话流程管理。



## 开始

### 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
venv\Scripts\activate
# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```
### 安装依赖

安装 PyTorch 2.13.0+cu132

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

*这里使用的pytorch版本较高，如有需要可自行降级。*

```bash
pip install -r requirements.txt
```

### 下载模型文件

从huggingface或其它途径下载运行服务时所需的模型文件：
ogkalu/comic-text-and-bubble-detector
zai-org/GLM-OCR

将其放入项目中对应的文件夹。

### 配置mcp服务示例

在agent的配置文件中：

```json

{
  "mcpServers": {
    "yoloServer": {
      "type": "stdio",
      "command": "C:/你的mcp服务所在目录/.venv/Scripts/python.exe",
      "args": ["C:/你的目录mcp服务所在/yolo_server.py"]
    }
  }
}

```









