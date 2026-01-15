# -*- coding: utf-8 -*-
"""
ComfyUI GLM-Image Pipeline Nodes
支持完整的 GLM-Image pipeline 运行

GLM-Image 架构:
- text_encoder: ByT5 Glyph Encoder (830MB) - 文字渲染编码
- vision_language_encoder: GLM-4 VL (18.8GB) - 视觉语言理解
- transformer: DiT (12.9GB) - Diffusion 解码
- vae: AutoencoderKL - 图像重建

需要安装: pip install git+https://github.com/huggingface/diffusers.git
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

WEB_DIRECTORY = "./js"

