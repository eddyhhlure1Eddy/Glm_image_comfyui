# GLM-Image ComfyUI Nodes
# Implements custom nodes for GLM-Image model integration
#
# GLM-Image Architecture:
# - T5 Encoder (~3B) - Text encoding + Glyph rendering
# - GLM-4 VL Encoder (9B) - Autoregressive visual token generation
# - DiT Transformer (7B) - Diffusion decoding
# - VAE - Image reconstruction
#
# Total: ~35.8 GB, requires ~60-80 GB VRAM

import torch
import os
import logging
from typing import Optional, Dict, Any, Tuple

import folder_paths
import comfy.model_management
import comfy.utils


class GLMImageModelLoader:
    """
    Load GLM-Image model components.
    
    This node loads the complete GLM-Image pipeline including:
    - Text encoder (T5)
    - Vision-language encoder (GLM-4 VL, 9B)
    - Diffusion transformer (DiT, 7B)
    - VAE
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING", {
                    "default": "zai-org/GLM-Image",
                    "multiline": False,
                }),
                "dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                }),
                "device_map": (["auto", "cuda", "balanced"], {
                    "default": "auto",
                }),
            },
            "optional": {
                "local_files_only": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("GLM_IMAGE_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "GLM-Image"
    
    def load_model(self, model_path: str, dtype: str, device_map: str, local_files_only: bool = False):
        """Load GLM-Image pipeline from HuggingFace or local path."""
        
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        
        try:
            from diffusers.pipelines.glm_image import GlmImagePipeline
            
            logging.info(f"Loading GLM-Image from {model_path}")
            
            pipeline = GlmImagePipeline.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                local_files_only=local_files_only,
            )
            
            logging.info("GLM-Image pipeline loaded successfully")
            return (pipeline,)
            
        except ImportError:
            raise RuntimeError(
                "GLM-Image requires the latest diffusers library. "
                "Install with: pip install git+https://github.com/huggingface/diffusers.git"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load GLM-Image: {e}")


class GLMImageGenerate:
    """
    Generate images using GLM-Image pipeline.
    
    This is the main generation node that handles the complete pipeline:
    1. Text encoding with T5 + ByT5 (for text rendering)
    2. Autoregressive visual token generation with GLM-4 VL
    3. Diffusion decoding with DiT
    4. VAE reconstruction
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("GLM_IMAGE_PIPELINE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful sunset over mountains with clear text 'Hello World'",
                }),
                "height": ("INT", {
                    "default": 1024,  # 官方推荐: 1152x1024 或 1024x1056
                    "min": 512,
                    "max": 2048,
                    "step": 32,
                }),
                "width": ("INT", {
                    "default": 1152,  # 避免使用 1024x1024，可能导致 token 生成问题
                    "min": 512,
                    "max": 2048,
                    "step": 32,
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "GLM-Image"
    
    def generate(
        self,
        pipeline,
        prompt: str,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        negative_prompt: str = "",
    ):
        """Generate image using GLM-Image pipeline."""
        
        # Validate dimensions
        if height % 32 != 0:
            height = (height // 32) * 32
            logging.warning(f"Height adjusted to {height} (must be divisible by 32)")
        if width % 32 != 0:
            width = (width // 32) * 32
            logging.warning(f"Width adjusted to {width} (must be divisible by 32)")
        
        # Set up generator for reproducibility
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Generate image
        logging.info(f"Generating image: {width}x{height}, steps={num_inference_steps}")
        
        # GLM-Image 不支持 negative_prompt
        result = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        # Get image from result
        if hasattr(result, 'images'):
            image = result.images[0]
        else:
            image = result[0]
        
        # Convert PIL to tensor if needed
        if hasattr(image, 'convert'):
            import numpy as np
            image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
            image = torch.from_numpy(image).unsqueeze(0)
        
        return (image,)


class GLMImageImg2Img:
    """
    Image-to-image generation with GLM-Image.
    
    Supports various editing tasks:
    - Image editing
    - Style transfer
    - Multi-subject consistency
    - Identity preservation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("GLM_IMAGE_PIPELINE",),
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Transform the image style to oil painting",
                }),
                "height": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 32,
                }),
                "width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 32,
                }),
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
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
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
    ):
        """Perform image-to-image transformation."""
        
        # Convert tensor to PIL
        from PIL import Image
        import numpy as np
        
        if image.dim() == 4:
            image = image[0]
        
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        # Validate dimensions
        if height % 32 != 0:
            height = (height // 32) * 32
        if width % 32 != 0:
            width = (width // 32) * 32
        
        generator = torch.Generator(device="cuda").manual_seed(seed)
        
        # Generate
        result = pipeline(
            prompt=prompt,
            image=[pil_image],
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        
        # Convert result
        if hasattr(result, 'images'):
            output_image = result.images[0]
        else:
            output_image = result[0]
        
        if hasattr(output_image, 'convert'):
            output_image = np.array(output_image.convert("RGB")).astype(np.float32) / 255.0
            output_image = torch.from_numpy(output_image).unsqueeze(0)
        
        return (output_image,)


class GLMImageTextEncode:
    """
    Encode text using GLM-Image's dual encoder.
    
    Uses both:
    - T5 encoder for main text understanding
    - ByT5 (Glyph encoder) for accurate text rendering
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("GLM_IMAGE_PIPELINE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": 'A sign that says "Hello World"',
                }),
            },
        }
    
    RETURN_TYPES = ("GLM_TEXT_EMBEDDINGS",)
    RETURN_NAMES = ("text_embeddings",)
    FUNCTION = "encode"
    CATEGORY = "GLM-Image/Advanced"
    
    def encode(self, pipeline, prompt: str):
        """Encode text prompt to embeddings."""
        
        # Use pipeline's text encoder
        if hasattr(pipeline, 'text_encoder') and hasattr(pipeline, 'tokenizer'):
            inputs = pipeline.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            
            with torch.no_grad():
                text_embeddings = pipeline.text_encoder(
                    input_ids=inputs.input_ids.to(pipeline.device),
                    attention_mask=inputs.attention_mask.to(pipeline.device),
                )
            
            return ({"embeddings": text_embeddings, "prompt": prompt},)
        else:
            # Return prompt for later processing
            return ({"prompt": prompt},)


class GLMImagePromptEnhancer:
    """
    Enhance prompts for better GLM-Image generation.
    
    Uses GLM-4.7 (or similar) to improve prompt quality
    for higher image quality output.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A sunset",
                }),
                "enhancement_level": (["minimal", "moderate", "detailed"], {
                    "default": "moderate",
                }),
            },
            "optional": {
                "style": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_prompt",)
    FUNCTION = "enhance"
    CATEGORY = "GLM-Image/Utils"
    
    def enhance(self, prompt: str, enhancement_level: str, style: str = ""):
        """Enhance prompt for better generation quality."""
        
        # Simple enhancement rules (can be expanded with LLM)
        enhancements = {
            "minimal": "",
            "moderate": ", highly detailed, professional quality, 8K resolution",
            "detailed": ", highly detailed, professional quality, 8K resolution, masterpiece, best quality, intricate details, sharp focus",
        }
        
        enhanced = prompt.strip()
        
        if style:
            enhanced = f"{enhanced}, {style} style"
        
        enhanced = f"{enhanced}{enhancements.get(enhancement_level, '')}"
        
        return (enhanced,)


# Node mappings for ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "GLMImageModelLoader": GLMImageModelLoader,
    "GLMImageGenerate": GLMImageGenerate,
    "GLMImageImg2Img": GLMImageImg2Img,
    "GLMImageTextEncode": GLMImageTextEncode,
    "GLMImagePromptEnhancer": GLMImagePromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GLMImageModelLoader": "Load GLM-Image Model",
    "GLMImageGenerate": "GLM-Image Generate",
    "GLMImageImg2Img": "GLM-Image Img2Img",
    "GLMImageTextEncode": "GLM-Image Text Encode",
    "GLMImagePromptEnhancer": "GLM-Image Prompt Enhancer",
}

