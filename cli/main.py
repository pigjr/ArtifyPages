#!/usr/bin/env python3
"""
文本转Epub工具 - 本地大模型版本
功能：读取txt文件，分章节处理，生成摘要和图片，输出epub
"""
import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
import time
from dataclasses import dataclass, field
import pickle
import psutil
import gc

# 导入本地模型
import huggingface_hub as hf_hub
import openvino_genai as ov_genai

from optimum.intel.openvino import OVStableDiffusionPipeline
from diffusers import LCMScheduler
    
# 模型配置
llm_model_key = "Qwen2.5-1.5B-Instruct-int4-ov" # Qwen2.5-7B-Instruct-int4-ov
llm_model_id = "OpenVINO/" + llm_model_key
llm_model_path = "models/" + llm_model_key
image_model_id = "OpenVINO/LCM_Dreamshaper_v7-fp16-ov"
image_model_path = "models/LCM_Dreamshaper_v7-fp16-ov"

def load_text_model():
    """加载文本模型"""
    if not os.path.exists(llm_model_path):
        print(f"下载文本模型到 {llm_model_path}...")
        hf_hub.snapshot_download(llm_model_id, local_dir=llm_model_path)
    print("尝试加载文本模型到CPU...")
    return ov_genai.LLMPipeline(llm_model_path, "CPU")

def summarize_text(text: str, model) -> str:
    """生成文本摘要"""
    try:
        prompt = "请用简洁的英文总结以下文本的核心内容，一定不能超过70个token：\n\n" + text + "\n\n"
        print("  🤖 正在生成摘要...")
        
        # 同步生成，确保完成
        result = model.generate(prompt, max_new_tokens=70)
        summary = result.strip()
        
        # 等待生成完成
        if len(summary) > 0:
            print(f"  ✅ 摘要生成完成: {len(summary)} 字")
        else:
            print("  ⚠️  摘要为空，使用原文前300字")
            summary = text[:300] + "..." if len(text) > 300 else text
            
        return summary
        
    except Exception as e:
        print(f"摘要生成失败: {e}")
        # 降级到简单截断
        return text[:300] + "..." if len(text) > 300 else text

def load_image_model():
    # """加载图像模型"""
    # if not os.path.exists(image_model_path):
    #     print(f"下载图像模型到 {image_model_path}...")
    #     hf_hub.snapshot_download(image_model_id, local_dir=image_model_path)
    # print("尝试加载图像模型到CPU...")
    # return ov_genai.VLMPipeline(image_model_path, "CPU")
    print("  🎨 加载图像模型...")
    
    # 检查模型是否存在
    if not os.path.exists(image_model_path):
        print(f"  📥 下载图像模型到 {image_model_path}...")
        hf_hub.snapshot_download(image_model_id, local_dir=image_model_path)
    
    print("尝试加载图像模型到GPU...")
    model = OVStableDiffusionPipeline.from_pretrained(
        image_model_path,
        device="GPU",
        compile=True
    )
    return model

def generate_image(prompt: str, output_path: str, model=None):
    """生成图像 - 使用预加载的模型"""
    try:
        print(f"  🎨 正在生成图像: {prompt}")
        
        # 使用传入的模型或加载新模型
        if model is None:
            model = load_image_model()
        
        # 设置LCM调度器
        model.scheduler = LCMScheduler.from_config(model.scheduler.config)
        
        # 强化安全设置 - 完全禁用安全检查器
        if hasattr(model, 'safety_checker'):
            model.safety_checker = None
        if hasattr(model, 'requires_safety_checker'):
            model.requires_safety_checker = False
        
        # 优化提示词，避免NSFW内容
        safe_prompt = prompt
        negative_prompt = "NSFW"
        
        print("  🎯 开始AI图像生成...")
        
        # 生成图像，添加安全参数
        result = model(
            safe_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=6,  # LCM快速模式
            guidance_scale=1.5,    # LCM推荐值
            width=512,
            height=512,
            generator=None,  # 不使用固定种子，增加多样性
            safety_checker=None, # 设为 None
            requires_safety_checker=False # 明确告诉它不需要
        )
        
        # 保存图像
        image = result.images[0]
        image.save(output_path)
        
        print(f"  ✅ AI图像生成完成: {output_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ AI图像生成失败: {e}")
        print("  🔄 使用占位符图像...")
        return generate_placeholder_image(prompt, output_path)

def generate_placeholder_image(prompt: str, output_path: str):
    """生成占位符图像"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (512, 512), color='lightgray')
        draw = ImageDraw.Draw(img)
        
        # 添加提示词文本
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # 绘制文本
        text_lines = []
        words = prompt.split()
        current_line = ""
        for word in words:
            if len(current_line + word) < 30:
                current_line += word + " "
            else:
                text_lines.append(current_line)
                current_line = word + " "
        if current_line:
            text_lines.append(current_line)
        
        y_offset = 50
        for line in text_lines[:5]:  # 最多显示5行
            draw.text((50, y_offset), line, fill='black', font=font)
            y_offset += 30
        
        img.save(output_path)
        print(f"  📝 占位符图像已保存: {output_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ 占位符图像生成失败: {e}")
        return False

# Epub生成库
try:
    from ebooklib import epub
except ImportError:
    print("正在安装epub库...")
    os.system("pip install ebooklib")
    from ebooklib import epub

@dataclass
class Character:
    """角色信息"""
    name: str
    description: str
    first_appearance: int  # 首次出现的章节索引
    last_updated: int  # 最后更新的章节索引
    image_path: str = ""
    appearances: List[int] = field(default_factory=list)  # 出现的章节列表

@dataclass
class Scene:
    """场景信息"""
    description: str
    chapter_index: int
    image_path: str = ""

@dataclass
class Chapter:
    """章节信息"""
    index: int
    title: str
    content: str
    summary: str = ""
    image_path: str = ""
    processed: bool = False
    characters: List[str] = field(default_factory=list)  # 本章出现的角色
    scenes: List[Scene] = field(default_factory=list)  # 本章的场景

class TextToEpubProcessor:
    def __init__(self, txt_file_path: str, paragraphs_per_chapter: int = 20):
        self.txt_file_path = txt_file_path
        self.paragraphs_per_chapter = paragraphs_per_chapter
        self.chapters: List[Chapter] = []
        self.progress_file = txt_file_path + ".progress"
        self.output_dir = Path('./output')
        
        # 模型状态管理 - 预加载模型
        self.text_model = None
        self.image_model = None
        self.current_model = None  # 当前活跃的模型
        self.model_lock = False    # 模型使用锁
        
        # 角色和场景管理
        self.characters: Dict[str, Character] = {}  # 全局角色字典
        self.character_data_file = txt_file_path + ".characters"  # 角色数据文件
        
        # 统计数据
        self.start_time = time.time()
        self.chapter_stats = {}
        self.total_chars = 0
        self.total_summary_chars = 0
        self.summary_times = []
        self.image_times = []
        
        print("初始化完成，将处理文件: {txt_file_path}")
        print("🔒 模型互斥模式：确保同时只有一个大模型工作")
        
        # 预加载模型
        self.preload_models()
    
    def extract_characters(self, text: str) -> List[str]:
        """提取章节中的主要角色"""
        prompt = f"""请用分析以下文本并识别具有姓名的主要角色。
        只返回角色姓名，每行一个姓名，按出现顺序排列。
        如果没有提到角色，返回 "None"。

        文本:
        {text}

        Characters:"""
        
        model = self.acquire_text_model()
        try:
            result = model.generate(prompt, max_new_tokens=100)
            characters_text = result.strip()
            
            if characters_text.lower() == "none":
                return []
            
            # 解析角色列表

            characters = []
            for line in characters_text.split('\n'):
                name = line.strip().strip('-').strip('*').strip()
                if name and len(name) > 1:  # 过滤单字符
                    characters.append(name)
            
            # 去重，保持顺序
            seen = set()
            unique_characters = []
            for char in characters:
                if char not in seen:
                    seen.add(char)
                    unique_characters.append(char)
            print("Characters:", unique_characters)
            
            return unique_characters[:5]  # 限制最多5个角色
            
        finally:
            self.release_model()
    
    def generate_character_description(self, character_name: str, text: str) -> str:
        """生成角色描述"""
        prompt = f"""请为角色 "{character_name}" 提供一个简洁的英文描述，基于给定的文本。
        重点关注角色的性别、外貌、性格和其在文本中的角色。避免使用格式如"- **性别**:"的语法。

        Text:
        {text}

        Description of {character_name} in English:"""
        model = self.acquire_text_model()
        try:
            result = model.generate(prompt, max_new_tokens=70)
            print("Character description:", result.strip())
            return result.strip()
        finally:
            self.release_model()
    
    def update_character_description(self, character_name: str, existing_description: str, new_text: str) -> str:
        """更新角色描述"""
        prompt = f"""请更新角色 "{character_name}" 的英文描述，结合新信息。
        保持简洁，不超过50个字，避免使用格式如"- **性别**:"。
        将现有和新信息合并为一个简单的英文段落。

        Existing Description:
        {existing_description}

        New Text:
        {new_text}

        Updated Description in English:"""
        
        model = self.acquire_text_model()
        try:
            result = model.generate(prompt, max_new_tokens=70)
            print("Updated character description:", result.strip())
            return result.strip()
        finally:
            self.release_model()
    
    def extract_scenes(self, text: str) -> List[str]:
        """提取章节中的亮点情节"""
        prompt = f"""请用简洁的英文总结文本中的一个亮点情节。
        只返回一个情节描述，不超过30个字。

        Text:
        {text}

        Highlight Scene in English:"""
        
        model = self.acquire_text_model()
        try:
            result = model.generate(prompt, max_new_tokens=50)
            scene_text = result.strip()
            
            if scene_text.lower() == "none":
                return []
            print("Highlight scene:", scene_text)
            
            # 检查是否为英文
            if not self.is_chinese_summary(scene_text):
                return [scene_text]
            else:
                print("  ⚠️  场景描述为中文，重新生成英文场景...")
                # 重新生成英文场景
                retry_prompt = f"""请用简洁的英文总结文本中的一个亮点情节。
                    只返回一个情节描述，不超过30个字。

                    文本:
                    {text}

                    Highlight Scene in English:"""
                
                retry_model = self.acquire_text_model()
                try:
                    retry_result = retry_model.generate(retry_prompt, max_new_tokens=50)
                    retry_scene_text = retry_result.strip()
                    
                    if retry_scene_text.lower() != "none" and not self.is_chinese_summary(retry_scene_text):
                        print("  ✅ 英文场景描述生成成功")
                        return [retry_scene_text]
                    else:
                        print("  ❌ 重新生成英文场景失败")
                        return []
                finally:
                    self.release_model()
            
        finally:
            self.release_model()
    
    def save_character_data(self):
        """保存角色数据"""
        try:
            with open(self.character_data_file, 'wb') as f:
                pickle.dump(self.characters, f)
        except Exception as e:
            print(f"保存角色数据失败: {e}")
    
    def load_character_data(self):
        """加载角色数据"""
        try:
            if os.path.exists(self.character_data_file):
                with open(self.character_data_file, 'rb') as f:
                    self.characters = pickle.load(f)
                print(f"📚 加载了 {len(self.characters)} 个角色的数据")
                return True
        except Exception as e:
            print(f"加载角色数据失败: {e}")
        return False
    
    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # 获取系统总内存
        memory = psutil.virtual_memory()
        total_memory_mb = memory.total / 1024 / 1024
        memory_percent = (memory_mb / total_memory_mb) * 100
        
        return memory_percent
    
    def check_memory_limit(self):
        """检查内存是否超过85%"""
        memory_percent = self.get_memory_usage()
        if memory_percent > 85:
            print(f"\n⚠️  内存使用过高: {memory_percent:.1f}%")
            print("🔄 保存当前进度并退出程序...")
            self.save_progress()
            print("💡 建议：重启程序或释放其他应用程序")
            return True
        return False
    
    def print_chapter_stats(self, chapter_index: int):
        """打印章节统计信息"""
        chapter = self.chapters[chapter_index]
        stats = self.chapter_stats.get(chapter_index, {})
        
        print(f"\n📊 === 第 {chapter_index + 1} 章统计 ===")
        print(f"💾 当前内存占用: {self.get_memory_usage():.1f}%")
        print(f"📝 本章字数: {len(chapter.content)} 字")
        print(f"📄 本章摘要字数: {len(chapter.summary)} 字")
        
        if 'summary_time' in stats:
            print(f"⏱️  生成摘要时间: {stats['summary_time']:.2f} 秒")
        
        if 'image_time' in stats:
            print(f"🎨 生成图像时间: {stats['image_time']:.2f} 秒")
        
        if 'total_time' in stats:
            print(f"⏰ 本章合计时间: {stats['total_time']:.2f} 秒")
        
        # 计算平均时间
        if self.summary_times:
            avg_summary_time = sum(self.summary_times) / len(self.summary_times)
            print(f"📈 生成摘要平均时间: {avg_summary_time:.2f} 秒")
        
        if self.image_times:
            avg_image_time = sum(self.image_times) / len(self.image_times)
            print(f"🖼️  生成图像平均时间: {avg_image_time:.2f} 秒")
        
        # 计算总统计
        elapsed_time = time.time() - self.start_time
        processed_chapters = len([c for c in self.chapters if c.processed])
        if processed_chapters > 0:
            avg_total_time = elapsed_time / processed_chapters
            print(f"📊 平均每章合计时间: {avg_total_time:.2f} 秒")
        
        print(f"⏳ 程序运行总时间: {elapsed_time:.2f} 秒")
        print(f"📈 总进度: {processed_chapters}/{len(self.chapters)} ({processed_chapters/len(self.chapters)*100:.1f}%)")
        
        # 计算预计剩余时间
        if processed_chapters > 0 and processed_chapters < len(self.chapters):
            avg_time_per_chapter = elapsed_time / processed_chapters
            remaining_chapters = len(self.chapters) - processed_chapters
            estimated_remaining_time = avg_time_per_chapter * remaining_chapters
            
            hours = int(estimated_remaining_time // 3600)
            minutes = int((estimated_remaining_time % 3600) // 60)
            
            if hours > 0:
                print(f"⏱️  预计剩余时间: {hours}小时{minutes}分钟")
            else:
                print(f"⏱️  预计剩余时间: {minutes}分钟")
        
        print("=" * 50)
    
    def preload_models(self):
        """预加载两个模型"""
        print("🔄 开始预加载模型...")
        
        # 加载文本模型
        print("  📥 加载文本模型...")
        try:
            self.text_model = load_text_model()
            print("  ✅ 文本模型加载完成")
        except Exception as e:
            print(f"  ❌ 文本模型加载失败: {e}")
            self.text_model = None
        
        # 加载图像模型
        print("  🎨 加载图像模型...")
        try:
            self.image_model = load_image_model()
            print("  ✅ 图像模型加载完成")
        except Exception as e:
            print(f"  ❌ 图像模型加载失败: {e}")
            self.image_model = None
        
        print("🎉 模型预加载完成！")
    
    def acquire_text_model(self):
        # 使用预加载的模型
        if not self.text_model:
            print("  📥 文本模型未预加载，现在加载...")
            self.text_model = load_text_model()
        
        return self.text_model
    
    def release_model(self):
        """释放模型"""
        self.model_lock = False
        self.current_model = None
        print("  🔓 模型已释放")
    
    def acquire_image_model(self):
        """获取图像模型（独占）"""
        while self.model_lock:
            print("  ⏳ 等待其他模型释放...")
            time.sleep(1)
        
        self.model_lock = True
        self.current_model = "image"
        
        # 使用预加载的模型
        if not self.image_model:
            print("  🎨 图像模型未预加载，现在加载...")
            from optimum.intel.openvino import OVStableDiffusionPipeline
            from diffusers import LCMScheduler
            
            model_path = "models/LCM_Dreamshaper_v7-fp16-ov"
            
            # 检查模型是否存在
            if not os.path.exists(model_path):
                print(f"  📥 下载图像模型到 {model_path}...")
                hf_hub.snapshot_download("OpenVINO/LCM_Dreamshaper_v7-fp16-ov", local_dir=model_path)
            
            self.image_model = OVStableDiffusionPipeline.from_pretrained(
                model_path,
                device="CPU",
                compile=True
            )
            
            # 设置LCM调度器
            self.image_model.scheduler = LCMScheduler.from_config(self.image_model.scheduler.config)
        
        return self.image_model
    
    def is_chinese_summary(self, summary: str) -> bool:
        """检测摘要是否包含中文字符"""
        # 更全面的中文检测
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        chinese_matches = chinese_pattern.findall(summary)
        
        # 如果中文字符超过总字符的20%，认为是中文摘要
        if len(chinese_matches) > len(summary) * 0.2:
            return True
        
        # 检查常见中文词汇
        chinese_words = ['的', '是', '在', '有', '和', '人', '这', '大', '为', '来', '及', '个', '了', '以', '到', '地', '要', '于', '得', '下', '就', '时', '也']
        chinese_word_count = sum(1 for word in chinese_words if word in summary)
        
        # 如果有3个以上常见中文词，认为是中文摘要
        return chinese_word_count >= 3
    
    def regenerate_english_summary(self, chapter: Chapter, max_retries: int = 2) -> str:
        """重新生成英文摘要"""
        for attempt in range(max_retries):
            print(f"  🔄 第{attempt + 1}次尝试生成英文摘要...")
            
            # 使用更强的英文提示词
            prompt = f"Please summarize the following text in English, no more than 70 tokens:\n\n{chapter.content}\n\n"
            
            model = self.acquire_text_model()
            try:
                result = model.generate(prompt, max_new_tokens=70)
                summary = result.strip()
                
                # 检查是否为英文
                if not self.is_chinese_summary(summary):
                    print(f"  ✅ 英文摘要生成成功: {summary}")
                    return summary
                else:
                    print(f"  ⚠️  摘要仍包含中文，第{attempt + 1}次失败")
                    
            finally:
                self.release_model()
            
            time.sleep(1)  # 短暂等待
        
        print("  ❌ 多次尝试仍生成英文摘要失败，使用原文前50字符")
        return chapter.content[:50] + "..." if len(chapter.content) > 50 else chapter.content
    
    def safe_summarize_text(self, text: str) -> str:
        model = self.acquire_text_model()
        try:
            return summarize_text(text, model)
        finally:
            self.release_model()
    
    def safe_generate_image(self, prompt: str, output_path: str) -> bool:
        """安全的图像生成（独占模型）"""
        model = self.acquire_image_model()
        try:
            return generate_image(prompt, output_path, model)
        finally:
            self.release_model()
    
    def load_progress(self) -> bool:
        """加载处理进度"""
        try:
            if os.path.exists(self.progress_file):
                with open(self.progress_file, 'rb') as f:
                    data = pickle.load(f)
                    self.chapters = data['chapters']
                    self.chapter_stats = data.get('chapter_stats', {})
                    self.summary_times = data.get('summary_times', [])
                    self.image_times = data.get('image_times', [])
                    self.total_chars = data.get('total_chars', 0)
                    self.total_summary_chars = data.get('total_summary_chars', 0)
                
                # 加载角色数据
                self.load_character_data()
                
                processed_count = len([c for c in self.chapters if c.processed])
                print(f"📚 加载进度: {processed_count}/{len(self.chapters)} 章节已处理")
                return True
        except Exception as e:
            print(f"加载进度失败: {e}")
        return False
    
    def save_progress(self):
        """保存进度"""
        try:
            data = {
                'chapters': self.chapters,
                'chapter_stats': self.chapter_stats,
                'summary_times': self.summary_times,
                'image_times': self.image_times,
                'total_chars': self.total_chars,
                'total_summary_chars': self.total_summary_chars
            }
            with open(self.progress_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"保存进度失败: {e}")
    
    def read_and_split_text(self) -> bool:
        """读取并分割文本"""
        try:
            with open(self.txt_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"读取文件失败: {e}")
            return False
        
        # 初始化章节标题列表
        self.chapter_titles = None
        
        # 尝试按章节分割
        chapters = self.split_by_chapters(content)
        
        if len(chapters) <= 1:
            # 如果无法分割章节，按段落分割
            print("📝 未检测到章节标题，按段落分割...")
            chapters = self.split_by_paragraphs(content)
            self.chapter_titles = None
        else:
            print(f"📖 成功提取 {len(chapters)} 个章节标题")
        
        # 创建章节对象
        self.chapters = []
        for i, chapter in enumerate(chapters):
            # 使用提取的标题或默认标题
            if self.chapter_titles and i < len(self.chapter_titles):
                title = self.chapter_titles[i]
            else:
                title = f"第{i+1}章"
            
            self.chapters.append(Chapter(i, title, chapter.strip()))
        
        print(f"文本分割完成，共 {len(self.chapters)} 章节")
        return True
    
    def split_by_chapters(self, content: str) -> List[str]:
        """按章节分割文本"""
        # 常见章节标题模式
        chapter_patterns = [
            r'第[一二三四五六七八九十百千万\d]+章[^\n]*\n',
            r'Chapter\s+\d+[^\n]*\n',
            r'第\d+节[^\n]*\n',
            r'^\d+\.[^\n]*\n',
            r'^[一二三四五六七八九十百千万]+、[^\n]*\n'
        ]
        
        chapters = []
        chapter_titles = []
        
        for pattern in chapter_patterns:
            matches = list(re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE))
            if len(matches) > 1:
                for i, match in enumerate(matches):
                    start = match.start()
                    end = matches[i+1].start() if i+1 < len(matches) else len(content)
                    chapter_content = content[start:end].strip()
                    chapters.append(chapter_content)
                    
                    # 提取章节标题
                    title_line = match.group().strip()
                    chapter_titles.append(title_line)
                
                # 存储章节标题
                self.chapter_titles = chapter_titles
                print(f"📚 检测到 {len(chapters)} 个章节，已提取标题")
                return [c for c in chapters if c.strip()]
        
        # 如果没有找到章节标题，使用默认标题
        self.chapter_titles = None
        return [content]
    
    def split_by_paragraphs(self, content: str) -> List[str]:
        """按段落分割文本"""
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        chapters = []
        
        for i in range(0, len(paragraphs), self.paragraphs_per_chapter):
            chapter_paragraphs = paragraphs[i:i+self.paragraphs_per_chapter]
            chapters.append('\n\n'.join(chapter_paragraphs))
        
        return chapters
    
    def process_chapter(self, chapter: Chapter) -> bool:
        """处理单个章节 - 确保模型互斥"""
        chapter_start_time = time.time()
        
        try:
            # 显示章节标题
            title_display = f"《{chapter.title}》" if chapter.title != f"第{chapter.index + 1}章" else chapter.title
            print(f"\n📖 正在处理 {title_display} (第 {chapter.index + 1} 章)...")
            
            # 生成摘要（独占文本模型）
            print("  📝 开始生成摘要...")
            summary_start_time = time.time()
            chapter.summary = self.safe_summarize_text(chapter.content)
            summary_time = time.time() - summary_start_time
            self.summary_times.append(summary_time)
            
            print(f"  📄 摘要: {chapter.summary}")
            
            # 检查摘要是否为中文，如果是则重新生成英文摘要
            if self.is_chinese_summary(chapter.summary):
                print("  ⚠️  检测到中文摘要，重新生成英文摘要...")
                chapter.summary = self.regenerate_english_summary(chapter)
                # 更新时间统计
                regenerate_time = time.time() - summary_start_time
                self.summary_times[-1] = regenerate_time  # 替换最后一次时间
            
            
            # 短暂等待，确保模型完全释放
            time.sleep(1)
            
            # 生成图片（独占图像模型）
            print("  🎨 开始生成配图...")
            image_start_time = time.time()
            image_prompt = f"{chapter.summary}"
            chapter.image_path = f"chapter_{chapter.index + 1}_image.png"
            full_image_path = self.output_dir / chapter.image_path
            
            # 调用安全的图像生成函数
            success = self.safe_generate_image(image_prompt, str(full_image_path))
            image_time = time.time() - image_start_time
            self.image_times.append(image_time)
            
            if not success:
                print(f"    ⚠️  图像生成失败，继续处理...")
            
            # 角色分析
            print("  👥 开始分析角色...")
            character_start_time = time.time()
            chapter.characters = self.extract_characters(chapter.content)
            
            # 处理每个角色
            for character_name in chapter.characters:
                if character_name not in self.characters:
                    # 新角色，生成描述和图片
                    print(f"    🆕 发现新角色: {character_name}")
                    description = self.generate_character_description(character_name, chapter.content)
                    
                    character = Character(
                        name=character_name,
                        description=description,
                        first_appearance=chapter.index,
                        last_updated=chapter.index,
                        image_path=f"character_{character_name.replace(' ', '_')}_chapter_{chapter.index + 1}.png"
                    )
                    character_full_path = self.output_dir / character.image_path
                    
                    character_prompt = f"{description}"
                    if self.safe_generate_image(character_prompt, str(character_full_path)):
                        print(f"    🖼️  {character_name} 角色图片生成成功")
                    else:
                        print(f"    ⚠️  {character_name} 角色图片生成失败")
                    
                    self.characters[character_name] = character
                    
                else:
                    # 已有角色，更新描述并生成新图片
                    existing_character = self.characters[character_name]
                    print(f"    🔄 更新角色: {character_name}")
                    updated_description = self.update_character_description(character_name, existing_character.description, chapter.content)
                    
                    # 为当前章节生成新的角色图片
                    chapter_character_image_path = f"character_{character_name.replace(' ', '_')}_chapter_{chapter.index + 1}.png"
                    chapter_character_full_path = self.output_dir / chapter_character_image_path
                    
                    print(f"    🖼️  重新生成 {character_name} 角色图片...")
                    character_prompt = f"{updated_description}"
                    
                    if self.safe_generate_image(character_prompt, str(chapter_character_full_path)):
                        print(f"    ✅ {character_name} 角色图片更新成功")
                    else:
                        print(f"    ⚠️  {character_name} 角色图片更新失败")
                    
                    # 更新全局角色描述
                    existing_character.description = updated_description
                    existing_character.last_updated = chapter.index
            
            character_time = time.time() - character_start_time
            print(f"  ⏱️  角色分析完成，耗时: {character_time:.2f} 秒")
            
            # 场景分析 - 只处理一个亮点情节
            print("  🎭 开始分析场景...")
            scene_start_time = time.time()
            scene_descriptions = self.extract_scenes(chapter.content)
            
            # 只生成一个场景图片
            if len(scene_descriptions) > 0:
                scene = Scene(
                    description=scene_descriptions[0],
                    chapter_index=chapter.index,
                    image_path=f"chapter_{chapter.index + 1}_scene_1.png"
                )
                
                # 生成场景图片
                scene_prompt = f"{scene_descriptions[0]}"
                scene_full_path = self.output_dir / scene.image_path
                
                if self.safe_generate_image(scene_prompt, str(scene_full_path)):
                    print(f"    🖼️  场景图片生成成功")
                else:
                    print(f"    ⚠️  场景图片生成失败")
                
                chapter.scenes.append(scene)
                print(f"    📝 亮点场景: {scene_descriptions[0][:50]}...")
            
            scene_time = time.time() - scene_start_time
            print(f"  ⏱️  场景分析完成，耗时: {scene_time:.2f} 秒")
            
            chapter.processed = True
            self.save_progress()
            self.save_character_data()  # 保存角色数据
            
            # 更新统计信息
            chapter_total_time = time.time() - chapter_start_time
            self.chapter_stats[chapter.index] = {
                'summary_time': summary_time,
                'image_time': image_time,
                'total_time': chapter_total_time
            }
            
            # 更新总字数统计
            self.total_chars += len(chapter.content)
            self.total_summary_chars += len(chapter.summary)
            
            # 打印章节统计
            self.print_chapter_stats(chapter.index)
            
            print(f"  ✅ {title_display} 处理完成")

            if self.check_memory_limit():
                print("内存使用过高，退出程序")
                return False

            return True
            
        except Exception as e:
            print(f"❌ 处理第 {chapter.index + 1} 章失败: {e}")
            # 确保释放模型锁
            if self.model_lock:
                self.release_model()
            return False
    
    def process_all_chapters(self):
        """处理所有章节 - 串行处理"""
        total_chapters = len(self.chapters)
        processed_count = len([c for c in self.chapters if c.processed])
        
        print(f"总共 {total_chapters} 章节，已处理 {processed_count} 章节")
        print("📝 采用串行处理模式，确保每章完整完成后再处理下一章...")
        
        for i, chapter in enumerate(self.chapters):
            if chapter.processed:
                print(f"⏭  跳过第 {i+1} 章（已处理）")
                continue
            
            print(f"\n🔄 开始处理第 {i+1} 章...")
            success = self.process_chapter(chapter)
            
            if not success:
                print(f"❌ 第 {i+1} 章处理失败，继续下一章...")
                continue
            
            # 显示进度
            processed = len([c for c in self.chapters if c.processed])
            progress = (processed / total_chapters) * 100
            print(f"📊 总进度: {processed}/{total_chapters} ({progress:.1f}%)")
            
            # 计算预计剩余时间
            if processed > 0:
                elapsed_time = time.time() - self.start_time
                avg_time_per_chapter = elapsed_time / processed
                remaining_chapters = total_chapters - processed
                estimated_remaining_time = avg_time_per_chapter * remaining_chapters
                
                hours = int(estimated_remaining_time // 3600)
                minutes = int((estimated_remaining_time % 3600) // 60)
                
                if hours > 0:
                    print(f"⏱️  预计剩余时间: {hours}小时{minutes}分钟")
                else:
                    print(f"⏱️  预计剩余时间: {minutes}分钟")
            
            # 确保当前章节完全完成后再继续
            print(f"✅ 第 {i+1} 章处理完成，等待2秒...")
            time.sleep(2)  # 短暂等待，确保资源释放
            
            print("-" * 50)  # 分隔线
    
    def generate_epub(self) -> str:
        """生成Epub文件"""
        print("正在生成Epub文件...")
        
        # 创建Epub书籍
        book = epub.EpubBook()
        
        # 设置元数据
        book.set_identifier(str(hash(self.txt_file_path)))
        book.set_title(Path(self.txt_file_path).stem)
        book.set_language('zh-CN')
        book.add_author('AI Assistant')
        
        # 添加章节
        epub_chapters = []
        spine = ['nav']
        
        for chapter in self.chapters:
            if not chapter.processed:
                continue
            
            # 创建章节内容
            chapter_content = f"<h1>{chapter.title}</h1>\n"
            
            if chapter.image_path:
                # 添加图片
                try:
                    with open(self.output_dir / chapter.image_path, 'rb') as img_file:
                        image_data = img_file.read()
                    
                    image_item = epub.EpubImage(
                        uid=f"image_{chapter.index}",
                        file_name=chapter.image_path,
                        media_type="image/png",
                        content=image_data
                    )
                    book.add_item(image_item)
                    
                    chapter_content += f'<p><img src="{chapter.image_path}" alt="{chapter.title}"/></p>\n'
                except Exception as e:
                    print(f"添加章节图片失败: {e}")
            
            # 添加角色图片和描述
            if chapter.characters:
                chapter_content += "<h3>角色介绍</h3>\n"
                for character_name in chapter.characters:
                    if character_name in self.characters:
                        character = self.characters[character_name]
                        
                        # 添加角色图片
                        if character.image_path and os.path.exists(self.output_dir / character.image_path):
                            try:
                                with open(self.output_dir / character.image_path, 'rb') as img_file:
                                    image_data = img_file.read()
                                
                                image_item = epub.EpubImage(
                                    uid=f"character_{character_name.replace(' ', '_')}",
                                    file_name=character.image_path,
                                    media_type="image/png",
                                    content=image_data
                                )
                                book.add_item(image_item)
                                
                                chapter_content += f'<div class="character">\n'
                                chapter_content += f'<h4>{character_name}</h4>\n'
                                chapter_content += f'<img src="{character.image_path}" alt="{character_name}"/>\n'
                                chapter_content += f'</div>\n'
                            except Exception as e:
                                print(f"添加角色图片失败: {e}")
                        else:
                            chapter_content += f'<div class="character">\n'
                            chapter_content += f'<h4>{character_name}</h4>\n'
                            chapter_content += f'</div>\n'
            
            # 添加场景图片和描述
            if chapter.scenes:
                chapter_content += "<h3>场景描述</h3>\n"
                for i, scene in enumerate(chapter.scenes):
                    # 添加场景图片
                    if scene.image_path and os.path.exists(self.output_dir / scene.image_path):
                        try:
                            with open(self.output_dir / scene.image_path, 'rb') as img_file:
                                image_data = img_file.read()
                            
                            image_item = epub.EpubImage(
                                uid=f"scene_{chapter.index}_{i}",
                                file_name=scene.image_path,
                                media_type="image/png",
                                content=image_data
                            )
                            book.add_item(image_item)
                            
                            chapter_content += f'<div class="scene">\n'
                            chapter_content += f'<img src="{scene.image_path}" alt="场景 {i+1}"/>\n'
                            # chapter_content += f'<p>{scene.description}</p>\n'
                            chapter_content += f'</div>\n'
                        except Exception as e:
                            print(f"添加场景图片失败: {e}")
                    else:
                        chapter_content += f'<div class="scene">\n'
                        # chapter_content += f'<p>{scene.description}</p>\n'
                        chapter_content += f'</div>\n'
            
            # 添加原文
            chapter_content += "<div>\n"
            # 保持原始换行符，将每个段落包装在p标签中
            paragraphs = chapter.content.split('\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    chapter_content += f"<p>{paragraph}</p>\n"
                else:
                    chapter_content += "<p>&nbsp;</p>\n"  # 空段落用nbsp保持
            chapter_content += "</div>\n"
            
            # 创建Epub章节
            epub_chapter = epub.EpubHtml(
                title=chapter.title,
                file_name=f"chapter_{chapter.index + 1}.xhtml",
                content=chapter_content
            )
            
            book.add_item(epub_chapter)
            epub_chapters.append(epub_chapter)
            spine.append(epub_chapter)
        
        # 添加导航
        book.toc = epub_chapters
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        
        # 设置spine
        book.spine = spine
        
        # 写入文件
        output_path = self.output_dir / f"{Path(self.txt_file_path).stem}_generated.epub"
        epub.write_epub(str(output_path), book, {})
        
        print(f"Epub文件已生成: {output_path}")
        return str(output_path)

def main():
    parser = argparse.ArgumentParser(description='文本转Epub工具')
    parser.add_argument('txt_file', nargs='?', help='输入的txt文件路径（可选，不提供则处理input文件夹所有文件）')
    parser.add_argument('--paragraphs', type=int, default=20, help='每章段落数（默认20）')
    parser.add_argument('--resume', action='store_true', help='从上次中断处继续')
    
    args = parser.parse_args()
    
    # 如果没有提供txt文件，处理input文件夹所有文件
    if not args.txt_file:
        input_dir = Path('./input')
        if not input_dir.exists():
            print("创建input文件夹...")
            input_dir.mkdir(exist_ok=True)
            print("请将要处理的txt文件放入input文件夹")
            return
        
        # 获取所有txt文件
        txt_files = list(input_dir.glob('*.txt'))
        if not txt_files:
            print("input文件夹中没有找到txt文件")
            return
        
        print(f"找到 {len(txt_files)} 个txt文件:")
        for i, txt_file in enumerate(txt_files, 1):
            print(f"  {i}. {txt_file.name}")
        
        # 逐个处理每个文件
        for txt_file in txt_files:
            print(f"\n{'='*60}")
            print(f"处理文件: {txt_file.name}")
            print(f"{'='*60}")
            
            # 创建处理器
            processor = TextToEpubProcessor(str(txt_file), args.paragraphs)
            
            try:
                # 加载进度
                if args.resume:
                    processor.load_progress()
                else:
                    # 读取并分割文本
                    if not processor.read_and_split_text():
                        continue
                    processor.save_progress()
                
                # 处理章节
                processor.process_all_chapters()
                
                # 生成Epub
                epub_path = processor.generate_epub()
                print(f"✅ {txt_file.name} 处理完成，Epub文件: {epub_path}")
                
            except KeyboardInterrupt:
                print(f"\n⚠️  {txt_file.name} 处理被中断")
                # 保存进度
                processor.save_progress()
                # 生成部分EPUB
                partial_epub_path = processor.generate_epub()
                print(f"📚 已生成部分EPUB: {partial_epub_path}")
                continue
            except Exception as e:
                print(f"❌ 处理 {txt_file.name} 时出错: {e}")
                continue
        
        print(f"\n🎉 所有文件处理完成！")
        return
    
    # 处理单个文件
    if not os.path.exists(args.txt_file):
        print(f"文件不存在: {args.txt_file}")
        return
    
    # 创建处理器
    processor = TextToEpubProcessor(args.txt_file, args.paragraphs)
    
    try:
        # 加载进度
        if args.resume:
            processor.load_progress()
        else:
            # 读取并分割文本
            if not processor.read_and_split_text():
                return
            processor.save_progress()
        
        # 处理章节
        processor.process_all_chapters()
        
        # 生成Epub
        epub_path = processor.generate_epub()
        print(f"✅ 处理完成，Epub文件: {epub_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️  处理被中断")
        # 保存进度
        processor.save_progress()
        # 生成部分EPUB
        partial_epub_path = processor.generate_epub()
        print(f"📚 已生成部分EPUB: {partial_epub_path}")
    except Exception as e:
        print(f"❌ 处理出错: {e}")
        processor.save_progress()
        
        # 出错时也尝试生成epub文件
        print("正在生成已处理章节的Epub文件...")
        try:
            epub_path = processor.generate_epub()
            print(f"已生成部分Epub文件: {epub_path}")
        except Exception as e2:
            print(f"生成Epub文件失败: {e2}")

if __name__ == "__main__":
    main()
