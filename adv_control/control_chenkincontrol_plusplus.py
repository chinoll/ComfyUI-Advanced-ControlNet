# Independent fork of control_plusplus.py with:
#   1. Wider input_hint_block channels [48, 96, 192, 384]
#   2. Experimental "fuse" control type
#
# Kept separate from control_plusplus.py so upstream ControlNet++ nodes are untouched.
from typing import Union

import os
import torch
import torch.nn as nn
from torch import Tensor

from comfy.ldm.modules.diffusionmodules.util import timestep_embedding
from comfy.ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential

import comfy.ops
import comfy.model_base
import comfy.model_management
import comfy.model_detection
import comfy.utils

from .control_plusplus import (ControlNetPlusPlus, ControlNetPlusPlusAdvanced,
                                PlusPlusInput, PlusPlusInputGroup, PlusPlusImageWrapper)
from .utils import (TimestepKeyframeGroup,
                    extend_to_batch_size, broadcast_image_to_extend)
from .logger import logger


class ChenkinPlusPlusType:
    OPENPOSE = "openpose"
    DEPTH = "depth"
    THICKLINE = "hed/pidi/scribble/ted"
    THINLINE = "canny/lineart/mlsd"
    NORMAL = "normal"
    SEGMENT = "segment"
    TILE = "tile"
    REPAINT = "inpaint/outpaint"
    FUSE = "fuse (experimental)"
    NONE = "none"
    _LIST_WITH_NONE = [OPENPOSE, DEPTH, THICKLINE, THINLINE, NORMAL, SEGMENT, TILE, REPAINT, FUSE, NONE]
    _LIST = [OPENPOSE, DEPTH, THICKLINE, THINLINE, NORMAL, SEGMENT, TILE, REPAINT, FUSE]
    _DICT = {OPENPOSE: 0, DEPTH: 1, THICKLINE: 2, THINLINE: 3, NORMAL: 4, SEGMENT: 5, TILE: 6, REPAINT: 7, FUSE: 8, NONE: -1}

    @classmethod
    def to_idx(cls, control_type: str):
        try:
            return cls._DICT[control_type]
        except KeyError:
            raise Exception(f"Unknown control type '{control_type}'.")


class ChenkinControlNetPlusPlus(ControlNetPlusPlus):
    """ControlNetPlusPlus with wider input_hint_block channels [48, 96, 192, 384]."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        operations: comfy.ops.disable_weight_init = kwargs.get("operations", comfy.ops.disable_weight_init)
        device = kwargs.get("device", None)
        dims = 2
        hint_channels = kwargs.get("hint_channels", 3)
        c0, c1, c2, c3 = 48, 96, 192, 384
        self.input_hint_block = TimestepEmbedSequential(
            operations.conv_nd(dims, hint_channels, c0, 3, padding=1, dtype=self.dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(dims, c0, c0, 3, padding=1, dtype=self.dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(dims, c0, c1, 3, padding=1, stride=2, dtype=self.dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(dims, c1, c1, 3, padding=1, dtype=self.dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(dims, c1, c2, 3, padding=1, stride=2, dtype=self.dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(dims, c2, c2, 3, padding=1, dtype=self.dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(dims, c2, c3, 3, padding=1, stride=2, dtype=self.dtype, device=device),
            nn.SiLU(),
            operations.conv_nd(dims, c3, self.model_channels, 3, padding=1, dtype=self.dtype, device=device),
        )


class ChenkinControlNetPlusPlusAdvanced(ControlNetPlusPlusAdvanced):
    """Advanced wrapper that routes type lookups through ChenkinPlusPlusType (FUSE-aware)."""

    def verify_control_type(self, model_name: str, pp_group: PlusPlusInputGroup=None):
        if pp_group is not None:
            for pp_input in pp_group.controls.values():
                if ChenkinPlusPlusType.to_idx(pp_input.control_type) >= self.control_model.num_control_type:
                    raise Exception(f"ControlNet++ model '{model_name}' does not support control_type '{pp_input.control_type}'.")
        if self.single_control_type is not None:
            if ChenkinPlusPlusType.to_idx(self.single_control_type) >= self.control_model.num_control_type:
                raise Exception(f"ControlNet++ model '{model_name}' does not support control_type '{self.single_control_type}'.")

    def get_control_advanced(self, x_noisy: Tensor, t, cond, batched_number, transformer_options):
        # Mirrors ControlNetPlusPlusAdvanced.get_control_advanced but uses ChenkinPlusPlusType for index lookup.
        control_prev = None
        if self.previous_controlnet is not None:
            control_prev = self.previous_controlnet.get_control(x_noisy, t, cond, batched_number, transformer_options)

        if self.timestep_range is not None:
            if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
                if control_prev is not None:
                    return control_prev
                else:
                    return None

        dtype = self.control_model.dtype
        if self.manual_cast_dtype is not None:
            dtype = self.manual_cast_dtype

        output_dtype = x_noisy.dtype

        if self.sub_idxs is not None or self.cond_hint is None or x_noisy.shape[2] * self.compression_ratio != self.cond_hint_shape[2] or x_noisy.shape[3] * self.compression_ratio != self.cond_hint_shape[3]:
            if self.cond_hint is not None:
                del self.cond_hint
            self.cond_hint = [None] * self.control_model.num_control_type
            self.cond_hint_types = torch.tensor([0.0] * self.control_model.num_control_type)
            self.cond_hint_shape = None
            compression_ratio = self.compression_ratio
            for pp_type, pp_input in self.cond_hint_original.controls.items():
                pp_idx = ChenkinPlusPlusType.to_idx(pp_type)
                if pp_idx < 0:
                    pp_idx = 0
                else:
                    self.cond_hint_types[pp_idx] = pp_input.strength
                if self.sub_idxs is not None:
                    actual_cond_hint_orig = pp_input.image
                    if pp_input.image.size(0) < self.full_latent_length:
                        actual_cond_hint_orig = extend_to_batch_size(tensor=actual_cond_hint_orig, batch_size=self.full_latent_length)
                    self.cond_hint[pp_idx] = comfy.utils.common_upscale(actual_cond_hint_orig[self.sub_idxs], x_noisy.shape[3] * compression_ratio, x_noisy.shape[2] * compression_ratio, 'nearest-exact', "center")
                else:
                    self.cond_hint[pp_idx] = comfy.utils.common_upscale(pp_input.image, x_noisy.shape[3] * compression_ratio, x_noisy.shape[2] * compression_ratio, 'nearest-exact', "center")
                self.cond_hint[pp_idx] = self.cond_hint[pp_idx].to(device=x_noisy.device, dtype=dtype)
                self.cond_hint_shape = self.cond_hint[pp_idx].shape
            if self.cond_hint_types.count_nonzero() == 0:
                self.cond_hint_types = None
            else:
                self.cond_hint_types = self.cond_hint_types.unsqueeze(0).to(device=x_noisy.device, dtype=dtype).repeat(x_noisy.shape[0], 1)
        for i in range(len(self.cond_hint)):
            if self.cond_hint[i] is not None:
                if x_noisy.shape[0] != self.cond_hint[i].shape[0]:
                    self.cond_hint[i] = broadcast_image_to_extend(self.cond_hint[i], x_noisy.shape[0], batched_number)
        if self.cond_hint_types is not None and x_noisy.shape[0] != self.cond_hint_types.shape[0]:
            self.cond_hint_types = broadcast_image_to_extend(self.cond_hint_types, x_noisy.shape[0], batched_number, False)

        self.prepare_mask_cond_hint(x_noisy=x_noisy, t=t, cond=cond, batched_number=batched_number, dtype=dtype)

        context = cond.get('crossattn_controlnet', cond['c_crossattn'])
        y = cond.get('y', None)
        if y is not None:
            y = comfy.model_base.convert_tensor(y, dtype, x_noisy.device)
        timestep = self.model_sampling_current.timestep(t)
        x_noisy = self.model_sampling_current.calculate_input(t, x_noisy)

        control = self.control_model(x=x_noisy.to(dtype), hint=self.cond_hint, timesteps=timestep.float(), context=comfy.model_management.cast_to_device(context, x_noisy.device, dtype), y=y, control_type=self.cond_hint_types)
        return self.control_merge(control, control_prev, output_dtype)

    def copy(self):
        c = ChenkinControlNetPlusPlusAdvanced(self.control_model, self.timestep_keyframes, global_average_pooling=self.global_average_pooling, load_device=self.load_device, manual_cast_dtype=self.manual_cast_dtype)
        self.copy_to(c)
        self.copy_to_advanced(c)
        c.single_control_type = self.single_control_type
        return c


def load_chenkincontrolnetplusplus(ckpt_path: str, timestep_keyframe: TimestepKeyframeGroup=None, model=None):
    controlnet_data = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    if "task_embedding" not in controlnet_data:
        raise Exception(f"'{ckpt_path}' is not a valid ControlNet++ model.")

    controlnet_config = None
    supported_inference_dtypes = None

    if "controlnet_cond_embedding.conv_in.weight" in controlnet_data:  # diffusers format
        controlnet_config = comfy.model_detection.unet_config_from_diffusers_unet(controlnet_data)
        diffusers_keys = comfy.utils.unet_to_diffusers(controlnet_config)
        diffusers_keys["controlnet_mid_block.weight"] = "middle_block_out.0.weight"
        diffusers_keys["controlnet_mid_block.bias"] = "middle_block_out.0.bias"

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                k_in = "controlnet_down_blocks.{}{}".format(count, s)
                k_out = "zero_convs.{}.0{}".format(count, s)
                if k_in not in controlnet_data:
                    loop = False
                    break
                diffusers_keys[k_in] = k_out
            count += 1

        count = 0
        loop = True
        while loop:
            suffix = [".weight", ".bias"]
            for s in suffix:
                if count == 0:
                    k_in = "controlnet_cond_embedding.conv_in{}".format(s)
                else:
                    k_in = "controlnet_cond_embedding.blocks.{}{}".format(count - 1, s)
                k_out = "input_hint_block.{}{}".format(count * 2, s)
                if k_in not in controlnet_data:
                    k_in = "controlnet_cond_embedding.conv_out{}".format(s)
                    loop = False
                diffusers_keys[k_in] = k_out
            count += 1

        new_sd = {}
        for k in diffusers_keys:
            if k in controlnet_data:
                new_sd[diffusers_keys[k]] = controlnet_data.pop(k)

        if "control_add_embedding.linear_1.bias" in controlnet_data:  # Union Controlnet
            controlnet_config["union_controlnet_num_control_type"] = controlnet_data["task_embedding"].shape[0]
            for k in list(controlnet_data.keys()):
                new_k = k.replace('.attn.in_proj_', '.attn.in_proj.')
                new_sd[new_k] = controlnet_data.pop(k)

        leftover_keys = controlnet_data.keys()
        if len(leftover_keys) > 0:
            logger.warning("leftover ControlNet++ keys: {}".format(leftover_keys))
        controlnet_data = new_sd
    elif "controlnet_blocks.0.weight" in controlnet_data:  # SD3 diffusers format
        raise Exception("Unexpected SD3 diffusers format for ControlNet++ model. Something is very wrong.")

    pth_key = 'control_model.zero_convs.0.0.weight'
    pth = False
    key = 'zero_convs.0.0.weight'
    if pth_key in controlnet_data:
        pth = True
        key = pth_key
        prefix = "control_model."
    elif key in controlnet_data:
        prefix = ""
    else:
        raise Exception("Unexpected T2IAdapter format for ControlNet++ model. Something is very wrong.")

    if controlnet_config is None:
        model_config = comfy.model_detection.model_config_from_unet(controlnet_data, prefix, True)
        supported_inference_dtypes = model_config.supported_inference_dtypes
        controlnet_config = model_config.unet_config

    load_device = comfy.model_management.get_torch_device()
    if supported_inference_dtypes is None:
        unet_dtype = comfy.model_management.unet_dtype()
    else:
        unet_dtype = comfy.model_management.unet_dtype(supported_dtypes=supported_inference_dtypes)

    manual_cast_dtype = comfy.model_management.unet_manual_cast(unet_dtype, load_device)
    if manual_cast_dtype is not None:
        controlnet_config["operations"] = comfy.ops.manual_cast
    controlnet_config["dtype"] = unet_dtype
    controlnet_config.pop("out_channels")
    controlnet_config["hint_channels"] = controlnet_data["{}input_hint_block.0.weight".format(prefix)].shape[1]
    control_model = ChenkinControlNetPlusPlus(**controlnet_config)

    if pth:
        if 'difference' in controlnet_data:
            if model is not None:
                comfy.model_management.load_models_gpu([model])
                model_sd = model.model_state_dict()
                for x in controlnet_data:
                    c_m = "control_model."
                    if x.startswith(c_m):
                        sd_key = "diffusion_model.{}".format(x[len(c_m):])
                        if sd_key in model_sd:
                            cd = controlnet_data[x]
                            cd += model_sd[sd_key].type(cd.dtype).to(cd.device)
            else:
                logger.warning("WARNING: Loaded a diff controlnet without a model. It will very likely not work.")

        class WeightsLoader(torch.nn.Module):
            pass
        w = WeightsLoader()
        w.control_model = control_model
        missing, unexpected = w.load_state_dict(controlnet_data, strict=False)
    else:
        missing, unexpected = control_model.load_state_dict(controlnet_data, strict=False)

    if len(missing) > 0:
        logger.warning("missing ControlNet++ keys: {}".format(missing))

    if len(unexpected) > 0:
        logger.debug("unexpected ControlNet++ keys: {}".format(unexpected))

    global_average_pooling = False
    filename = os.path.splitext(ckpt_path)[0]
    if filename.endswith("_shuffle") or filename.endswith("_shuffle_fp16"):
        global_average_pooling = True

    control = ChenkinControlNetPlusPlusAdvanced(control_model, timestep_keyframes=timestep_keyframe, global_average_pooling=global_average_pooling, load_device=load_device, manual_cast_dtype=manual_cast_dtype)
    return control
