import torch
from optimum.intel.openvino import OVStableDiffusionPipeline
from diffusers import LCMScheduler
from PIL import Image
import os
import sys
import fcntl

# 防止重复运行的锁机制
def acquire_lock():
    try:
        lock_file = open("test-image.lock", "w")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_file
    except:
        print("脚本已在运行中，退出...")
        sys.exit(1)

# 获取锁
lock = acquire_lock()

# 1. 配置模型和设备
# model_path = "./lcm-dreamshaper-int8" # 刚才下载的模型路径
model_path = "models/LCM_Dreamshaper_v7-fp16-ov"

device = "GPU" # 强制使用 UHD 730 核显。如果报错，可改为 "CPU" 或 "AUTO"

print(f"正在加载模型到设备: {device}... 请稍候，这可能需要一分钟...")

try:
    # 2. 加载 OpenVINO 优化的 LCM 管道
    pipe = OVStableDiffusionPipeline.from_pretrained(
        model_path,
        device=device,
        compile=True # 加载时预编译模型
    )
    
    # 3. 必须：将调度器设置为 LCM
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    
    print("模型加载完成！")

    # 4. 定义咒语 (Prompt)
    # Dreamshaper 对现实主义和风格化图像都很擅长
    prompt = "A cinematic shot of a futuristic Toronto cityscape in Ontario Canada, snowy winter, neon lights reflection on wet street, photorealistic, 8k, extremely detailed"
    
    # 定义反向咒语 (避免不需要的东西)
    negative_prompt = "bad anatomy, blurry, low quality, deformed hands, ugly, text, watermark"

    print(f"正在生成图片，咒语: '{prompt}'")
    print("提示：LCM 模型只需 4 步即可出图。")

    # 5. 运行生成
    # 关键参数：
    # - num_inference_steps=4: 极速模式
    # - guidance_scale=1.5: LCM 建议设置在 1.0 - 2.0 之间
    result = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=4, 
        guidance_scale=1.5,
        width=512, # Intel 核显建议从 512x512 开始
        height=512
    )

    # 6. 保存图片
    image = result.images[0]
    output_name = "generated_lcm_image.png"
    image.save(output_name)
    print(f"成功！图片已保存为: {output_name}")

    # 在 Linux 上尝试自动打开图片 (可选)
    os.system(f"xdg-open {output_name}")

except Exception as e:
    print(f"出错了: {e}")
    if device == "GPU":
        print("\n友情提示：可能是核显内存不足或驱动问题。尝试将代码中的 device 改为 'CPU' 试试看。")
except KeyboardInterrupt:
    print("\n用户停止了运行。")
finally:
    # 释放锁和清理
    if 'lock' in locals():
        lock.close()
    if os.path.exists("test-image.lock"):
        os.remove("test-image.lock")