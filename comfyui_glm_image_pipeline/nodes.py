# -*- coding: utf-8 -*-
"""
GLM-Image Pipeline 节点 for ComfyUI
支持完整的 GLM-Image pipeline 运行

集成 CPU Offload 和显存优化功能
"""
import torch
import numpy as np
import logging
import os
import gc
import folder_paths

logger = logging.getLogger("GLM-Image")

# 注册 GLM-Image 模型文件夹
GLM_IMAGE_DIR = os.path.join(folder_paths.models_dir, "glm_image")
if not os.path.exists(GLM_IMAGE_DIR):
    os.makedirs(GLM_IMAGE_DIR, exist_ok=True)


def get_glm_image_models():
    """扫描 glm_image 目录下的模型"""
    models = []
    
    if os.path.exists(GLM_IMAGE_DIR):
        for name in os.listdir(GLM_IMAGE_DIR):
            path = os.path.join(GLM_IMAGE_DIR, name)
            if os.path.isdir(path) and os.path.exists(os.path.join(path, "model_index.json")):
                models.append(name)
    
    return models


def get_gpu_memory_info():
    """获取 GPU 显存信息"""
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        free = total - allocated
        return {"total_gb": total, "allocated_gb": allocated, "free_gb": free}
    return {"total_gb": 0, "allocated_gb": 0, "free_gb": 0}


class GLMImagePipelineLoader:
    """
    加载完整的 GLM-Image Pipeline (支持显存优化)
    
    GLM-Image 组件:
    - ByT5 Glyph Encoder (~830MB)
    - GLM-4 VL Encoder (~18GB) 
    - DiT Transformer (~13GB)
    - VAE (~500MB)
    
    总计: ~35GB (bfloat16)
    
    显存优化选项:
    - cpu_offload: 模型级别卸载，需要 ~8-10GB 显存
    - sequential_offload: 层级别卸载，需要 ~5GB 显存
    - attention_slicing: 注意力切片，减少峰值显存
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        models = get_glm_image_models()
        if not models:
            models = ["(no models found)"]
        
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "E:/glm_image",
                    "multiline": False,
                    "tooltip": "GLM-Image 模型完整路径"
                }),
                "dtype": (["bfloat16", "float16"], {
                    "default": "bfloat16",
                    "tooltip": "模型精度"
                }),
                "offload_mode": (["none", "model_offload", "sequential_offload"], {
                    "default": "model_offload",
                    "tooltip": "显存优化模式:\n"
                              "- none: 全部加载到GPU (~35GB)\n"
                              "- model_offload: 模型级卸载 (~10GB)\n"
                              "- sequential_offload: 层级卸载 (~5GB，最慢)"
                }),
            },
            "optional": {
                "model_select": (models, {
                    "tooltip": "从 models/glm_image/ 选择模型"
                }),
                "enable_attention_slicing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用注意力切片，减少峰值显存"
                }),
                "enable_vae_slicing": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "启用 VAE 切片"
                }),
                "enable_vae_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用 VAE 分块（用于超大分辨率）"
                }),
                "max_memory_gb": ("FLOAT", {
                    "default": 0,
                    "min": 0,
                    "max": 80,
                    "step": 1,
                    "tooltip": "最大显存使用量(GB)，0=自动"
                }),
            }
        }
    
    RETURN_TYPES = ("GLM_IMAGE_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "GLM-Image"
    DESCRIPTION = """加载 GLM-Image Pipeline (支持显存优化)

显存需求估算:
• none (无优化): ~35GB
• model_offload: ~10GB  
• sequential_offload: ~5GB (速度最慢)

推荐配置:
• 32GB显存: model_offload + attention_slicing
• 24GB显存: sequential_offload + 所有优化
• 16GB显存: sequential_offload + 所有优化 + 降低分辨率"""
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
    
    def load_pipeline(
        self, 
        model_path: str, 
        dtype: str, 
        offload_mode: str,
        model_select: str = None, 
        enable_attention_slicing: bool = True,
        enable_vae_slicing: bool = True,
        enable_vae_tiling: bool = False,
        max_memory_gb: float = 0
    ):
        try:
            from diffusers import GlmImagePipeline
        except ImportError:
            raise RuntimeError(
                "需要安装最新版 diffusers:\n"
                "pip install git+https://github.com/huggingface/diffusers.git"
            )
        
        # 确定最终路径
        final_path = model_path
        if model_select and model_select != "(no models found)":
            select_path = os.path.join(GLM_IMAGE_DIR, model_select)
            if os.path.exists(os.path.join(select_path, "model_index.json")):
                final_path = select_path
        
        # 验证路径
        index_file = os.path.join(final_path, "model_index.json")
        if not os.path.exists(index_file):
            raise ValueError(f"无效的模型路径: {final_path}\n找不到 model_index.json")
        
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        
        # 显示显存信息
        mem_info = get_gpu_memory_info()
        logger.info(f"GPU 显存: {mem_info['free_gb']:.1f}GB 可用 / {mem_info['total_gb']:.1f}GB 总计")
        logger.info(f"加载 GLM-Image Pipeline: {final_path}")
        logger.info(f"精度: {dtype}, Offload模式: {offload_mode}")
        
        # 清理显存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 根据 offload 模式加载
        if offload_mode == "sequential_offload":
            # 顺序卸载模式 - 最省显存但最慢
            logger.info("使用 Sequential CPU Offload 模式 (预计需要 ~5GB 显存)")
            pipe = GlmImagePipeline.from_pretrained(
                final_path,
                torch_dtype=dtype_map[dtype],
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            pipe.enable_sequential_cpu_offload()
            
        elif offload_mode == "model_offload":
            # 模型卸载模式 - 平衡显存和速度
            logger.info("使用 Model CPU Offload 模式 (预计需要 ~10GB 显存)")
            pipe = GlmImagePipeline.from_pretrained(
                final_path,
                torch_dtype=dtype_map[dtype],
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            
            # 设置最大显存
            if max_memory_gb > 0:
                max_memory = {0: f"{int(max_memory_gb)}GB", "cpu": "64GB"}
                pipe.enable_model_cpu_offload(gpu_id=0)
            else:
                pipe.enable_model_cpu_offload()
            
        else:
            # 无卸载模式 - 需要大显存
            logger.info("使用直接加载模式 (预计需要 ~35GB 显存)")
            pipe = GlmImagePipeline.from_pretrained(
                final_path,
                torch_dtype=dtype_map[dtype],
                local_files_only=True,
            )
            pipe = pipe.to("cuda")
        
        # 应用其他优化
        if enable_attention_slicing:
            if hasattr(pipe, 'enable_attention_slicing'):
                pipe.enable_attention_slicing("auto")
                logger.info("已启用 Attention Slicing")
        
        if enable_vae_slicing:
            if hasattr(pipe, 'enable_vae_slicing'):
                pipe.enable_vae_slicing()
                logger.info("已启用 VAE Slicing")
        
        if enable_vae_tiling:
            if hasattr(pipe, 'enable_vae_tiling'):
                pipe.enable_vae_tiling()
                logger.info("已启用 VAE Tiling")
        
        # 显示最终显存使用
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mem_info = get_gpu_memory_info()
        logger.info(f"加载完成! 当前显存使用: {mem_info['allocated_gb']:.1f}GB")
        
        return (pipe,)


# 推荐分辨率预设 (官方推荐，避免使用正方形 1024x1024)
GLM_IMAGE_RESOLUTIONS = {
    "1152x1024 (官方示例)": (1152, 1024),    # 36x32 tokens - 最稳定
    "1024x1056 (官方示例)": (1024, 1056),    # 32x33 tokens
    "1056x1056 (1:1)": (1056, 1056),         # 33x33 tokens - 方形推荐
    "1280x1280 (1:1)": (1280, 1280),         # 40x40 tokens
    "1568x1056 (3:2)": (1568, 1056),
    "1056x1568 (2:3)": (1056, 1568),
    "1472x1088 (4:3)": (1472, 1088),
    "1088x1472 (3:4)": (1088, 1472),
    "1728x960 (16:9)": (1728, 960),
    "960x1728 (9:16)": (960, 1728),
    "768x768 (小尺寸)": (768, 768),           # 24x24 tokens - 省显存
    "custom": (0, 0),
}


class GLMImageGenerate:
    """
    使用 GLM-Image Pipeline 生成图像
    
    支持文字渲染: 在 prompt 中使用引号包裹需要渲染的文字
    例如: "一只猫，图片上写着 'Hello World'"
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("GLM_IMAGE_PIPELINE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful sunset over mountains with text 'Hello World'",
                    "tooltip": "生成提示词，用引号包裹需要渲染的文字"
                }),
                "resolution": (list(GLM_IMAGE_RESOLUTIONS.keys()), {
                    "default": "1152x1024 (官方示例)",
                    "tooltip": "推荐分辨率预设 (避免使用 1024x1024，可能导致 token 生成问题)"
                }),
                "width": ("INT", {
                    "default": 1024, 
                    "min": 512, 
                    "max": 2048, 
                    "step": 32,
                    "tooltip": "自定义宽度 (仅 resolution=custom 时生效)"
                }),
                "height": ("INT", {
                    "default": 1024, 
                    "min": 512, 
                    "max": 2048, 
                    "step": 32,
                    "tooltip": "自定义高度 (仅 resolution=custom 时生效)"
                }),
                "steps": ("INT", {
                    "default": 50, 
                    "min": 1, 
                    "max": 200,
                    "tooltip": "推理步数"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 1.5, 
                    "min": 0.0, 
                    "max": 10.0, 
                    "step": 0.1,
                    "tooltip": "引导强度"
                }),
                "seed": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 0xffffffffffffffff,
                    "tooltip": "随机种子，0 为随机"
                }),
                "control_after_generate": (["fixed", "randomize", "increment", "decrement"], {
                    "default": "randomize",
                    "tooltip": "生成后种子控制"
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "GLM-Image"
    DESCRIPTION = "使用 GLM-Image 生成图像，支持精确文字渲染"
    
    def generate(
        self, 
        pipeline, 
        prompt: str, 
        resolution: str,
        width: int, 
        height: int, 
        steps: int, 
        guidance_scale: float, 
        seed: int,
        control_after_generate: str = "randomize",
    ):
        # 设置随机种子
        if seed == 0:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        # 确定 generator 设备
        try:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        except:
            generator = torch.Generator().manual_seed(seed)
        
        # 使用预设分辨率或自定义
        if resolution != "custom" and resolution in GLM_IMAGE_RESOLUTIONS:
            width, height = GLM_IMAGE_RESOLUTIONS[resolution]
        
        # GLM-Image 分辨率必须是 32 的倍数
        width = (width // 32) * 32
        height = (height // 32) * 32
        
        logger.info(f"生成图像: {width}x{height}, steps={steps}, seed={seed}")
        
        # 生成 (GLM-Image 不支持 negative_prompt)
        result = pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        # 转换为 ComfyUI 格式 (B, H, W, C) float32 0-1
        image = result.images[0]
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0)
        
        return (image_tensor,)


class GLMImageImg2Img:
    """
    GLM-Image 图生图
    
    支持:
    - 图像编辑
    - 风格迁移
    - 多主体一致性
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("GLM_IMAGE_PIPELINE",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Transform the image to oil painting style",
                }),
                "steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "guidance_scale": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "img2img"
    CATEGORY = "GLM-Image"
    
    def img2img(
        self, 
        pipeline, 
        image: torch.Tensor, 
        prompt: str, 
        steps: int, 
        guidance_scale: float, 
        seed: int
    ):
        from PIL import Image
        
        # 转换输入图像
        if image.dim() == 4:
            image = image[0]
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        # 获取尺寸
        width, height = pil_image.size
        width = (width // 32) * 32
        height = (height // 32) * 32
        
        if seed == 0:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        try:
            generator = torch.Generator(device="cuda").manual_seed(seed)
        except:
            generator = torch.Generator().manual_seed(seed)
        
        result = pipeline(
            prompt=prompt,
            image=[pil_image],
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        output_image = result.images[0]
        output_np = np.array(output_image).astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(output_np).unsqueeze(0)
        
        return (output_tensor,)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "GLMImagePipelineLoader": GLMImagePipelineLoader,
    "GLMImageGenerate": GLMImageGenerate,
    "GLMImageImg2Img": GLMImageImg2Img,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GLMImagePipelineLoader": "Load GLM-Image Pipeline",
    "GLMImageGenerate": "GLM-Image Generate",
    "GLMImageImg2Img": "GLM-Image Img2Img",
}
