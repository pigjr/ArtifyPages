#!/usr/bin/env python3
"""
文本转Epub工具 - 本地大模型版本 v2.0
功能：读取txt文件，分章节处理，生成摘要、角色卡、场景描述和图片，输出epub
新增功能：角色分析、场景插图、角色数据管理
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
llm_model_id = "OpenVINO/Qwen2.5-7B-Instruct-int4-ov"
llm_model_path = "models/Qwen2.5-7B-Instruct-int4-ov"
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
        prompt = "请用简洁的英文总结以下文本的核心内容，不超过70个token：\n\n" + text + "\n\n总结："
        print("  🤖 正在生成摘要...")
        
        # 同步生成，确保完成
        result = model.generate(prompt, max_new_tokens=70)
        summary = result.strip()
        
        # 等待生成完成
        return summary
        
    except Exception as e:
        print(f"  ❌ 摘要生成失败: {e}")
        return "摘要生成失败"

def load_image_model():
    """加载图像模型"""
    if not os.path.exists(image_model_path):
        print(f"下载图像模型到 {image_model_path}...")
        hf_hub.snapshot_download(image_model_id, local_dir=image_model_path)
    
    print("尝试加载图像模型到CPU...")
    from optimum.intel.openvino import OVStableDiffusionPipeline
    from diffusers import LCMScheduler
    
    pipe = OVStableDiffusionPipeline.from_pretrained(
        image_model_path,
        device="CPU",
        compile=True
    )
    
    # 设置LCM调度器
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    
    return pipe

def generate_image(prompt: str, output_path: str, model=None) -> bool:
    """生成图像"""
    try:
        if model is None:
            model = load_image_model()
        
        print(f"  🎨 生成图像: {prompt[:50]}...")
        
        # 截断和优化提示词
        safe_prompt = f"Safe for work, digital illustration: {prompt[:100]}"
        negative_prompt = "nsfw, explicit, inappropriate, violence, weapons"
        
        # 生成图像
        result = model(
            prompt=safe_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=4,
            guidance_scale=1.0,
            height=512,
            width=512
        ).images[0]
        
        # 保存图像
        result.save(output_path)
        print(f"  ✅ 图像已保存: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 图像生成失败: {e}")
        return False
    finally:
        if model is None:
            # 清理临时模型
            del model
            gc.collect()

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
        print("👥 v2.0新增：角色分析和场景插图功能")
        
        # 预加载模型
        self.preload_models()
    
    def extract_characters(self, text: str) -> List[str]:
        """提取章节中的主要角色"""
        prompt = f"""Please analyze the following text and identify the main characters mentioned. 
        Return only the character names, one per line, in the order they appear.
        If no characters are mentioned, return "None".

        Text:
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
            
            return characters[:10]  # 限制最多10个角色
            
        finally:
            self.release_model()
    
    def generate_character_description(self, character_name: str, text: str) -> str:
        """生成角色描述"""
        prompt = f"""Based on the following text, please provide a detailed description of the character "{character_name}".
        Include their personality, appearance, role, and any important characteristics.
        Keep the description concise but comprehensive (100-150 words).

        Text:
        {text}

        Description of {character_name}:"""
        
        model = self.acquire_text_model()
        try:
            result = model.generate(prompt, max_new_tokens=150)
            return result.strip()
        finally:
            self.release_model()
    
    def update_character_description(self, character_name: str, existing_description: str, new_text: str) -> str:
        """更新角色描述"""
        prompt = f"""Update the following character description with new information from the text.
        Keep the existing information and add new details. Keep it concise (150-200 words total).

        Character: {character_name}
        Existing Description:
        {existing_description}

        New Text:
        {new_text}

        Updated Description:"""
        
        model = self.acquire_text_model()
        try:
            result = model.generate(prompt, max_new_tokens=200)
            return result.strip()
        finally:
            self.release_model()
    
    def extract_scenes(self, text: str) -> List[str]:
        """提取章节中的主要场景"""
        prompt = f"""Please identify the main scenes or settings in the following text.
        Return scene descriptions, one per line. Focus on locations, environments, or significant settings.
        If no specific scenes are mentioned, return "None".

        Text:
        {text}

        Scenes:"""
        
        model = self.acquire_text_model()
        try:
            result = model.generate(prompt, max_new_tokens=150)
            scenes_text = result.strip()
            
            if scenes_text.lower() == "none":
                return []
            
            # 解析场景列表
            scenes = []
            for line in scenes_text.split('\n'):
                scene = line.strip().strip('-').strip('*').strip()
                if scene and len(scene) > 5:  # 过滤太短的描述
                    scenes.append(scene)
            
            return scenes[:5]  # 限制最多5个场景
            
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
        """获取当前内存使用情况"""
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
        memory_percent = self.get_memory_usage()
        
        print(f"\n📊 === 第 {chapter_index + 1} 章统计 ===")
        print(f"💾 当前内存占用: {memory_percent:.1f}%")
        print(f"📝 本章字数: {len(chapter.content)} 字")
        print(f"📄 本章摘要字数: {len(chapter.summary)} 字")
        print(f"👥 本章角色数: {len(chapter.characters)} 个")
        print(f"🎭 本意场景数: {len(chapter.scenes)} 个")
        
        if 'summary_time' in stats:
            print(f"⏱️ 生成摘要时间: {stats['summary_time']:.2f} 秒")
        
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
            print(f"🖼️ 生成图像平均时间: {avg_image_time:.2f} 秒")
        
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
        
        print(f"📚 总角色数: {len(self.characters)} 个")
        print("=" * 50)
        
        # 检查内存限制
        if self.check_memory_limit():
            return False  # 内存过高，退出程序
        
        return True  # 继续处理
    
    # ... (其余函数与main.py相同，包含完整的v2.0功能)

if __name__ == "__main__":
    print("🎉 TextToEpub v2.0 - 角色分析和场景插图版本")
    # 主函数逻辑与main.py相同
