# 文本转Epub工具使用说明

## 功能
- 读取txt文件，自动分章节处理
- 使用本地AI模型生成章节摘要
- 生成章节配图
- 输出带插图和摘要的Epub文件
- 支持断点续传

## 安装依赖
```bash
pip install openvino-genai huggingface_hub ebooklib
```

## 使用方法

### 基本用法
```bash
python3 main.py input.txt
```

### 指定每章段落数
```bash
python3 main.py input.txt --paragraphs 15
```

### 断点续传
```bash
python3 main.py input.txt --resume
```

## 模型说明
- **文本模型**: OpenVINO/Qwen2.5-7B-Instruct-int4-ov (约4GB)
- **图像模型**: OpenVINO/LCM_Dreamshaper_v7-fp16-ov (约2GB)

首次运行会自动下载模型到 `models/` 目录。

## 输出
- 生成的Epub文件: `input_generated.epub`
- 章节图片: `chapter_X_image.png`
- 进度文件: `input.txt.progress`

## 注意事项
- 确保有足够的磁盘空间（模型+输出文件约10GB）
- 首次运行需要下载模型，时间较长
- 可以随时按Ctrl+C中断，下次使用--resume继续
