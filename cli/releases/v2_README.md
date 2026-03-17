# TextToEpub v2.0 功能说明

## 🎯 项目概述
TextToEpub v2.0是基于本地大模型的智能文本处理工具，在v1.0基础上增加了角色分析和场景插图功能。能够读取txt文件，分章节处理，生成摘要、角色卡、场景描述和配图，最终输出为丰富的epub格式。

## ✨ 新增功能 (v2.0)

### 👥 智能角色分析
- **角色识别**: 自动识别每章出现的主要角色
- **角色卡生成**: 为每个角色生成详细描述（100-150词）
- **角色插图**: 为角色卡生成专属肖像图片
- **角色追踪**: 记录角色首次出现和所有出现章节
- **描述更新**: 随着剧情发展自动更新角色描述

### 🎭 场景插图生成
- **场景识别**: 自动识别每章的主要场景和设置
- **场景描述**: 生成场景的详细描述
- **场景插图**: 为每个场景生成专属插图
- **位置标记**: 记录场景出现的章节位置

### 📚 角色数据管理
- **持久化存储**: 角色数据独立保存，支持中断恢复
- **跨章节关联**: 角色在不同章节间保持一致性
- **智能更新**: 基于新内容自动更新角色信息
- **图片更新**: 角色描述显著变化时自动更新图片

## 🔄 完整处理流程

### 1. 文本预处理
```
📖 读取txt文件
📝 按段落数分割章节
🏷️ 提取章节标题
```

### 2. 章节智能分析
```
📝 生成章节摘要
👥 分析主要角色
  🆕 新角色 → 生成角色卡 + 插图
  🔄 已有角色 → 更新描述 + 可能更新插图
🎭 分析主要场景
  🖼️ 为每个场景生成插图
🎨 生成章节配图
```

### 3. 数据管理
```
💾 保存章节进度
📚 保存角色数据
🔄 支持中断恢复
```

### 4. EPUB生成
```
📖 章节内容
  🖼️ 章节配图
  👥 角色介绍 + 插图
  🎭 场景描述 + 插图
  📝 原文内容
📚 生成完整EPUB
```

## 📋 输出示例

### 角色分析输出
```
👥 开始分析角色...
  🆕 发现新角色: 汪淼
    📝 汪淼: 纳米材料科学家，性格内向但专业能力强...
    🖼️  汪淼 角色图片生成成功
  🆕 发现新角色: 史强
    📝 史强: 经验丰富的刑警，性格粗犷但观察力敏锐...
    🖼️  史强 角色图片生成成功
⏱️  角色分析完成，耗时: 15.23 秒
```

### 场景分析输出
```
🎭 开始分析场景...
  📝 场景 1: 现代化的纳米材料实验室，充满高科技设备...
    🖼️  场景 1 图片生成成功
  📝 场景 2: 繁华的北京街头，夜晚霓虹闪烁...
    🖼️  场景 2 图片生成成功
⏱️  场景分析完成，耗时: 8.67 秒
```

### EPUB结构
```html
<h1>第一章</h1>
<p><img src="chapter_1_image.png" alt="第一章"/></p>

<h3>角色介绍</h3>
<div class="character">
  <h4>汪淼</h4>
  <img src="character_汪淼_image.png" alt="汪淼"/>
  <p>纳米材料科学家，性格内向但专业能力强...</p>
</div>

<h3>场景描述</h3>
<div class="scene">
  <img src="chapter_1_scene_1.png" alt="场景 1"/>
  <p>现代化的纳米材料实验室，充满高科技设备...</p>
</div>

<div>
  <p>原文段落1...</p>
  <p>原文段落2...</p>
</div>
```

## 🛠️ 技术特性

### 角色分析算法
```python
def extract_characters(text: str) -> List[str]:
    """使用LLM提取主要角色"""
    prompt = "分析文本，识别主要角色，按出现顺序返回"
    
def generate_character_description(name: str, text: str) -> str:
    """生成角色详细描述"""
    prompt = "基于文本生成角色描述（100-150词）"
    
def update_character_description(name: str, existing: str, new_text: str) -> str:
    """更新角色描述"""
    prompt = "基于新内容更新角色描述，保持一致性"
```

### 场景分析算法
```python
def extract_scenes(text: str) -> List[str]:
    """提取主要场景"""
    prompt = "识别文本中的主要场景和设置"
    # 最多返回5个场景描述
```

### 数据结构
```python
@dataclass
class Character:
    name: str
    description: str
    first_appearance: int
    last_updated: int
    image_path: str
    appearances: List[int]

@dataclass
class Scene:
    description: str
    chapter_index: int
    image_path: str

@dataclass
class Chapter:
    index: int
    title: str
    content: str
    summary: str
    image_path: str
    processed: bool
    characters: List[str]
    scenes: List[Scene]
```

## 📊 性能优化

### 智能处理策略
- **角色限制**: 每章最多处理10个角色
- **场景限制**: 每章最多处理5个场景
- **图片更新**: 仅在描述显著变化时更新角色图片
- **增量处理**: 只处理新出现的角色和场景

### 内存管理
- **角色数据分离**: 角色数据独立存储
- **进度保存**: 章节和角色数据分别保存
- **中断恢复**: 支持任意位置中断和恢复

## 🎯 使用方法

### 基本用法
```bash
python3 main.py input.txt
```

### 恢复处理
```bash
python3 main.py input.txt --resume
```

### 输出文件结构
```
output/
├── chapter_1_image.png          # 章节配图
├── chapter_1_scene_1.png        # 场景插图
├── chapter_1_scene_2.png
├── character_汪淼_image.png     # 角色肖像
├── character_史强_image.png
└── input_generated.epub         # 最终EPUB
```

## 📈 相比v1.0的改进

### 功能增强
- **角色系统**: 完整的角色识别、描述和插图系统
- **场景插图**: 自动识别和生成场景插图
- **数据持久化**: 角色数据跨章节保持一致性
- **智能更新**: 基于内容变化自动更新角色信息

### 输出质量
- **丰富内容**: EPUB包含角色介绍和场景描述
- **视觉增强**: 大量高质量插图提升阅读体验
- **结构化**: 清晰的HTML结构便于阅读器解析

### 用户体验
- **智能分析**: 自动识别重要内容元素
- **进度保护**: 更完善的中断恢复机制
- **详细反馈**: 实时显示处理进度和结果

## 🔧 系统要求

### 硬件要求
- **CPU**: 支持OpenVINO的x86_64处理器
- **内存**: 建议16GB以上（角色和场景处理需要更多内存）
- **存储**: 至少20GB可用空间（更多图片输出）
- **GPU**: 可选，Intel集成显卡或独立显卡

### 软件依赖
```bash
pip install openvino-genai
pip install optimum-intel
pip install diffusers
pip install huggingface_hub
pip install psutil
pip install ebooklib
```

## 🎉 版本信息

- **版本**: v2.0
- **发布日期**: 2024年3月
- **兼容性**: Python 3.8+
- **平台**: Linux/macOS/Windows

## 📞 技术支持

### 常见问题
1. **角色识别不准确**: 调整提示词或增加角色限制
2. **图片生成失败**: 检查模型加载和存储空间
3. **内存不足**: 减少每章段落数或关闭其他程序
4. **角色数据丢失**: 检查.character文件权限

### 联系方式
- **项目地址**: [GitHub仓库链接]
- **问题反馈**: [Issues页面]
- **使用文档**: [Wiki文档]

---

*TextToEpub v2.0 - 让文本处理更智能，让阅读体验更丰富*
