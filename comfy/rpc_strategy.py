from abc import ABC, abstractmethod
from typing import Dict, Any


class ModelParameterStrategy(ABC):
    @abstractmethod
    def get_parameters(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        pass


class WanAnimateParameterStrategy(ModelParameterStrategy):
    """WanAnimate ksampler node"""
    
    def get_parameters(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        positive = inputs["positive"]
        negative = inputs["negative"]
        latent = inputs["latent"]

        return {
            'model': inputs.get("model"),
            'seed': inputs.get("seed"),
            'steps': inputs.get("steps"),
            'cfg': inputs.get("cfg"),
            'sampler_name': inputs.get("sampler_name"),
            'scheduler': inputs.get("scheduler"),
            'positive_prompt_embeds': positive[0][0].contiguous(),
            'negative_prompt_embeds': negative[0][0].contiguous(),
            'penultimate_hidden_states': positive[0][1]["clip_vision_output"].penultimate_hidden_states.contiguous(),
            "pose_video_latent": positive[0][1]["pose_video_latent"].contiguous(),
            "face_video_pixels": positive[0][1]["face_video_pixels"].contiguous(),
            "concat_latent_image": positive[0][1]["concat_latent_image"].contiguous(),
            'concat_mask': positive[0][1]["concat_mask"].contiguous(),
            'latent': latent,
            'denoise': inputs.get("denoise"),
            'disable_noise': inputs.get("disable_noise"),
            'start_step': inputs.get("start_step"),
            'last_step': inputs.get("last_step"),
            'force_full_denoise': inputs.get("force_full_denoise"),
        }


class KJWanAnimateParameterStrategy(ModelParameterStrategy):
    """WanAnimate KJ node"""
    
    def get_parameters(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        参数说明：
        inputs 必须包含：
        - text_embeds: {"prompt_embeds": Tensor, "negative_prompt_embeds": Tensor}
        - clip_fea: Tensor
        - wananim_pose_latents: Tensor
        - wananim_face_pixels: Tensor
        - image_cond: Tensor
        - latent: Tensor
        其他参数通过字典键直接传递
        """
        return {
            'model': inputs.get("model"),
            'seed': inputs.get("seed"),
            'steps': inputs.get("steps"),
            'cfg': inputs.get("cfg"),
            'sampler_name': None,  # 固定为None
            'scheduler': inputs.get("scheduler"),
            'positive_prompt_embeds': inputs["text_embeds"]["prompt_embeds"][0].unsqueeze(0).contiguous(),
            'negative_prompt_embeds': inputs["text_embeds"]["negative_prompt_embeds"][0].unsqueeze(0).contiguous(),
            'penultimate_hidden_states': inputs["clip_fea"].contiguous(),
            "pose_video_latent": inputs["wananim_pose_latents"].contiguous(),
            "pose_strength": inputs.get("wananim_pose_strength", 1.0),  # 默认值
            "face_video_pixels": inputs["wananim_face_pixels"].contiguous(),
            "face_strength": inputs.get("wananim_face_strength", 1.0),  # 默认值
            "concat_latent_image": inputs["image_cond"][4:].unsqueeze(0).contiguous(),
            'concat_mask': (1.0 - inputs["image_cond"][:4]).unsqueeze(0).contiguous(),
            'latent': {"samples": inputs["latent"].unsqueeze(0).contiguous()},
            'denoise': None,  # 固定为None
            'disable_noise': True,  # 固定为True
            'start_step': inputs.get("start_step", 0),  # 默认从0开始
            'last_step': inputs.get("end_step", inputs.get("steps")),  # 兼容end_step/steps
            'force_full_denoise': inputs.get("add_noise_to_samples", False),
            "enable_preprocess": False,  # 固定False
            "enable_postprocess": False,  # 固定False
            "shift": inputs.get("shift", 5.0),  # 默认值
        }


class ParameterStrategyFactory:

    _strategies = {
        "kj_wan_animate": KJWanAnimateParameterStrategy(),
        "wan_animate": WanAnimateParameterStrategy(),
        # 注册其他模型的策略...
        # "other_model": OtherModelParameterStrategy(),
    }

    @classmethod
    def get_strategy(cls, model_type: str) -> ModelParameterStrategy:
        strategy = cls._strategies.get(model_type)
        if not strategy:
            raise ValueError(f"Unsupported model type: {model_type}")
        return strategy
