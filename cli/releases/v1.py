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
from typing import List, Dict, Tuple
import time
from dataclasses import dataclass
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
        
        # 优化提示词，避免NSFW内容
        safe_prompt = prompt
        negative_prompt = "NSFW, explicit content, bad anatomy, blurry, low quality, deformed hands, ugly, text, watermark, violence, inappropriate content"
        
        print("  🎯 开始AI图像生成...")
        print("  ⚡ LCM模型，4步生成...")
        
        # 生成图像，添加安全参数
        result = model(
            safe_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=4,  # LCM快速模式
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
class Chapter:
    """章节数据结构"""
    index: int
    title: str
    content: str
    summary: str = ""
    image_path: str = ""
    processed: bool = False

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
        """获取文本模型（独占）"""
        while self.model_lock:
            print("  ⏳ 等待其他模型释放...")
            time.sleep(1)
        
        self.model_lock = True
        self.current_model = "text"
        
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
    
    def safe_summarize_text(self, text: str) -> str:
        """安全的文本摘要（独占模型）"""
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
        """加载进度"""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'rb') as f:
                    self.chapters = pickle.load(f)
                print(f"发现进度文件，已处理 {len([c for c in self.chapters if c.processed])} 章节")
                return True
            except Exception as e:
                print(f"进度文件损坏，重新开始: {e}")
        return False
    
    def save_progress(self):
        """保存进度"""
        try:
            with open(self.progress_file, 'wb') as f:
                pickle.dump(self.chapters, f)
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
            
            # 短暂等待，确保模型完全释放
            time.sleep(1)
            
            # 生成图片（独占图像模型）
            print("  🎨 开始生成配图...")
            image_start_time = time.time()
            image_prompt = f"Digital illustration for: {chapter.summary[:100]}"
            chapter.image_path = f"chapter_{chapter.index + 1}_image.png"
            full_image_path = self.output_dir / chapter.image_path
            
            # 调用安全的图像生成函数
            success = self.safe_generate_image(image_prompt, str(full_image_path))
            image_time = time.time() - image_start_time
            self.image_times.append(image_time)
            
            if not success:
                print(f"    ⚠️  图像生成失败，继续处理...")
            
            chapter.processed = True
            self.save_progress()
            
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
                    print(f"添加图片失败: {e}")
            
            # 添加摘要
            if chapter.summary:
                chapter_content += f"<h3>摘要</h3>\n<p>{chapter.summary}</p>\n"
            
            # 添加原文
            chapter_content += "<div>\n"
            for paragraph in chapter.content.split('\n\n'):
                if paragraph.strip():
                    chapter_content += f"<p>{paragraph}</p>\n"
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
    parser.add_argument('txt_file', help='输入的txt文件路径')
    parser.add_argument('--paragraphs', type=int, default=20, help='每章段落数（默认20）')
    parser.add_argument('--resume', action='store_true', help='从上次中断处继续')
    
    args = parser.parse_args()
    
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
        
        print(f"\n处理完成！输出文件: {epub_path}")
        
    except KeyboardInterrupt:
        print("\n用户中断处理，进度已保存")
        processor.save_progress()
    except Exception as e:
        print(f"处理出错: {e}")
        processor.save_progress()

if __name__ == "__main__":
    main()
