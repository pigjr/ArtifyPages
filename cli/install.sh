#!/bin/bash

echo "=== 文本转Epub工具安装脚本 ==="

# 检查Python版本
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "错误: 需要Python 3.8或更高版本，当前版本: $python_version"
    exit 1
fi

echo "Python版本检查通过: $python_version"

# 安装依赖
echo "正在安装依赖包..."
pip3 install -r requirements.txt

# 检查OpenVINO
echo "检查OpenVINO..."
python3 -c "import openvino_genai" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ OpenVINO GenAI 已安装"
else
    echo "❌ OpenVINO GenAI 安装失败"
    echo "请参考: https://github.com/openvinotoolkit/openvino.genai"
fi

# 检查HuggingFace Hub
echo "检查HuggingFace Hub..."
python3 -c "import huggingface_hub" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ HuggingFace Hub 已安装"
else
    echo "❌ HuggingFace Hub 安装失败"
fi

# 检查Epub库
echo "检查Epub库..."
python3 -c "import ebooklib" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Ebooklib 已安装"
else
    echo "❌ Ebooklib 安装失败"
fi

echo ""
echo "=== 安装完成 ==="
echo "使用方法:"
echo "  python3 main.py your_text_file.txt"
echo ""
echo "首次运行会自动下载AI模型（约6GB），请确保网络连接正常"
