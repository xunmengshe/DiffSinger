import math
import re
import struct
import sys
from functools import partial

import numpy as np
import onnx
import onnxsim
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.nn import Linear

from modules.commons.common_layers import Mish
from src.diff.net import AttrDict
from utils import load_ckpt
from utils.hparams import hparams, set_hparams


def extract(a, t):
    return a[t].reshape((1, 1, 1, 1))


def linear_beta_schedule(timesteps, max_beta=hparams.get('max_beta', 0.01)):
    """
    linear schedule
    """
    betas = np.linspace(1e-4, max_beta, timesteps)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


beta_schedule = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
}


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * torch.tensor(-emb)).unsqueeze(0))

    def forward(self, x):
        emb = self.emb * x
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class KaimingNormalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight)


class ResidualBlock(nn.Module):
    def __init__(self, encoder_hidden, residual_channels, dilation):
        super().__init__()
        self.residual_channels = residual_channels
        self.dilated_conv = KaimingNormalConv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation)
        self.diffusion_projection = Linear(residual_channels, residual_channels)
        self.conditioner_projection = KaimingNormalConv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = KaimingNormalConv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)
        y = x + diffusion_step

        y = self.dilated_conv(y) + conditioner

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        gate, filter = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)

        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)

        return (x + residual) / math.sqrt(2.0), skip


class DiffNet(nn.Module):
    def __init__(self, in_dims=80):
        super().__init__()
        self.params = params = AttrDict(
            # Model params
            encoder_hidden=hparams['hidden_size'],
            residual_layers=hparams['residual_layers'],
            residual_channels=hparams['residual_channels'],
            dilation_cycle_length=hparams['dilation_cycle_length'],
        )
        self.input_projection = KaimingNormalConv1d(in_dims, params.residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(params.residual_channels)
        dim = params.residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.encoder_hidden, params.residual_channels, 2 ** (i % params.dilation_cycle_length))
            for i in range(params.residual_layers)
        ])
        self.skip_projection = KaimingNormalConv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = KaimingNormalConv1d(params.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    # TODO: swap order of `diffusion_steps` and `cond`
    def forward(self, spec, diffusion_step, cond):
        """

        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :return:
        """
        x = spec.squeeze(1)
        x = self.input_projection(x)  # [B, residual_channel, T]

        x = functional.relu(x)
        diffusion_step = diffusion_step.float()
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)

        # Avoid ConstantOfShape op
        x, skip = self.residual_layers[0](x, cond, diffusion_step)
        # noinspection PyTypeChecker
        for layer in self.residual_layers[1:]:
            x, skip_connection = layer.forward(x, cond, diffusion_step)
            skip += skip_connection

        x = skip / math.sqrt(len(self.residual_layers))

        x = self.skip_projection(x)
        x = functional.relu(x)
        x = self.output_projection(x)  # [B, mel_bins, T]
        return x.unsqueeze(1)


class NaiveNoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('clip_min', to_torch(-1.))
        self.register_buffer('clip_max', to_torch(1.))

    def forward(self, x, noise_pred, t):
        x_recon = (
                extract(self.sqrt_recip_alphas_cumprod, t) * x -
                extract(self.sqrt_recipm1_alphas_cumprod, t) * noise_pred
        )
        x_recon = torch.clamp(x_recon, min=self.clip_min, max=self.clip_max)

        model_mean = (
                extract(self.posterior_mean_coef1, t) * x_recon +
                extract(self.posterior_mean_coef2, t) * x
        )
        model_log_variance = extract(self.posterior_log_variance_clipped, t)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = ((t > 0).float()).reshape(1, 1, 1, 1)
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


class PLMSNoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        to_torch = partial(torch.tensor, dtype=torch.float32)

        # Below are buffers for TorchScript to pass jit compilation.
        self.register_buffer('_1', to_torch(1))
        self.register_buffer('_2', to_torch(2))
        self.register_buffer('_3', to_torch(3))
        self.register_buffer('_5', to_torch(5))
        self.register_buffer('_9', to_torch(9))
        self.register_buffer('_12', to_torch(12))
        self.register_buffer('_16', to_torch(16))
        self.register_buffer('_23', to_torch(23))
        self.register_buffer('_24', to_torch(24))
        self.register_buffer('_37', to_torch(37))
        self.register_buffer('_55', to_torch(55))
        self.register_buffer('_59', to_torch(59))

    def forward(self, x, noise_t, t, t_prev):
        a_t = extract(self.alphas_cumprod, t)
        a_prev = extract(self.alphas_cumprod, t_prev)
        a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

        x_delta = (a_prev - a_t) * ((self._1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x - self._1 / (
                a_t_sq * (((self._1 - a_prev) * a_t).sqrt() + ((self._1 - a_t) * a_prev).sqrt())) * noise_t)
        x_pred = x + x_delta

        return x_pred

    def predict_stage0(self, noise_pred, noise_pred_prev):
        return (noise_pred
                + noise_pred_prev) / self._2

    def predict_stage1(self, noise_pred, noise_list):
        return (noise_pred * self._3
                - noise_list[-1]) / self._2

    def predict_stage2(self, noise_pred, noise_list):
        return (noise_pred * self._23
                - noise_list[-1] * self._16
                + noise_list[-2] * self._5) / self._12

    def predict_stage3(self, noise_pred, noise_list):
        return (noise_pred * self._55
                - noise_list[-1] * self._59
                + noise_list[-2] * self._37
                - noise_list[-3] * self._9) / self._24


class MelExtractor(nn.Module):
    def __init__(self, spec_min, spec_max, keep_bins):
        super().__init__()
        self.register_buffer('spec_min', torch.FloatTensor(spec_min)[None, None, :keep_bins])
        self.register_buffer('spec_max', torch.FloatTensor(spec_max)[None, None, :keep_bins])

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        d = (self.spec_max - self.spec_min) / 2
        m = (self.spec_max + self.spec_min) / 2
        return x * d + m


class GaussianDiffusion(nn.Module):
    def __init__(self, out_dims, timesteps=1000, k_step=1000, spec_min=None, spec_max=None):
        super().__init__()
        self.mel_bins = out_dims
        self.K_step = k_step

        self.denoise_fn = DiffNet(out_dims)
        self.naive_noise_predictor = NaiveNoisePredictor()
        self.plms_noise_predictor = PLMSNoisePredictor()
        self.mel_extractor = MelExtractor(spec_min=spec_min, spec_max=spec_max, keep_bins=hparams['keep_bins'])

        if 'schedule_type' in hparams.keys():
            betas = beta_schedule[hparams['schedule_type']](timesteps)
        else:
            betas = cosine_beta_schedule(timesteps)

        # Below are buffers for state_dict to load into.
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self.naive_noise_predictor = NaiveNoisePredictor()
        self.plms_noise_predictor = PLMSNoisePredictor()
        self.mel_extractor = MelExtractor(spec_min=spec_min, spec_max=spec_max, keep_bins=hparams['keep_bins'])

    def build_submodules(self):
        # Move registered buffers into submodules after loading state dict.
        self.naive_noise_predictor.register_buffer('sqrt_recip_alphas_cumprod', self.sqrt_recip_alphas_cumprod)
        self.naive_noise_predictor.register_buffer('sqrt_recipm1_alphas_cumprod', self.sqrt_recipm1_alphas_cumprod)
        self.naive_noise_predictor.register_buffer(
            'posterior_log_variance_clipped', self.posterior_log_variance_clipped)
        self.naive_noise_predictor.register_buffer('posterior_mean_coef1', self.posterior_mean_coef1)
        self.naive_noise_predictor.register_buffer('posterior_mean_coef2', self.posterior_mean_coef2)
        self.plms_noise_predictor.register_buffer('alphas_cumprod', self.alphas_cumprod)

    def forward(self, condition, speedup):
        device = condition.device
        condition = condition.transpose(1, 2)  # (1, n_frames, 256) => (1, 256, n_frames)

        n_frames = condition.shape[2]
        step_range = torch.arange(0, self.K_step, speedup, dtype=torch.long, device=device).flip(0)
        x = torch.randn((1, 1, self.mel_bins, n_frames), device=device)

        if speedup > 1:
            plms_noise_stage = torch.tensor(0, dtype=torch.long, device=device)
            noise_list = torch.zeros((0, 1, 1, self.mel_bins, n_frames), device=device)
            for t in step_range:
                noise_pred = self.denoise_fn(x, t, condition)
                t_prev = t - speedup
                t_prev = t_prev * (t_prev > 0)

                if plms_noise_stage == 0:
                    x_pred = self.plms_noise_predictor(x, noise_pred, t, t_prev)
                    noise_pred_prev = self.denoise_fn(x_pred, t_prev, condition)
                    noise_pred_prime = self.plms_noise_predictor.predict_stage0(noise_pred, noise_pred_prev)
                elif plms_noise_stage == 1:
                    noise_pred_prime = self.plms_noise_predictor.predict_stage1(noise_pred, noise_list)
                elif plms_noise_stage == 2:
                    noise_pred_prime = self.plms_noise_predictor.predict_stage2(noise_pred, noise_list)
                else:
                    noise_pred_prime = self.plms_noise_predictor.predict_stage3(noise_pred, noise_list)

                noise_pred = noise_pred.unsqueeze(0)
                if plms_noise_stage < 3:
                    noise_list = torch.cat((noise_list, noise_pred), dim=0)
                    plms_noise_stage = plms_noise_stage + 1
                else:
                    noise_list = torch.cat((noise_list[-2:], noise_pred), dim=0)

                x = self.plms_noise_predictor(x, noise_pred_prime, t, t_prev)

            # from dpm_solver import NoiseScheduleVP, model_wrapper, DpmSolver
            # ## 1. Define the noise schedule.
            # noise_schedule = NoiseScheduleVP(betas=self.betas)
            #
            # ## 2. Convert your discrete-time `model` to the continuous-time
            # # noise prediction model. Here is an example for a diffusion model
            # ## `model` with the noise prediction type ("noise") .
            #
            # model_fn = model_wrapper(
            #     self.denoise_fn,
            #     noise_schedule,
            #     model_kwargs={"cond": condition}
            # )
            #
            # ## 3. Define dpm-solver and sample by singlestep DPM-Solver.
            # ## (We recommend singlestep DPM-Solver for unconditional sampling)
            # ## You can adjust the `steps` to balance the computation
            # ## costs and the sample quality.
            # dpm_solver = DpmSolver(model_fn, noise_schedule)
            #
            # steps = t // hparams["pndm_speedup"]
            # x = dpm_solver.sample(x, steps=steps)
        else:
            for t in step_range:
                pred = self.denoise_fn(x, t, condition)
                x = self.naive_noise_predictor(x, pred, t)

        mel = self.mel_extractor(x)
        return mel


class DiffDecoder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = build_model()
        self.model.eval()
        self.model.to(device)

    def forward(self, condition, speedup):
        mel = self.model.forward(condition, speedup)  # (1, n_frames, mel_bins)
        return mel


def build_model():
    model = GaussianDiffusion(
        out_dims=hparams['audio_num_mel_bins'],
        timesteps=hparams['timesteps'],
        k_step=hparams['K_step'],
        spec_min=hparams['spec_min'],
        spec_max=hparams['spec_max'],
    )
    model.eval()
    load_ckpt(model, hparams['work_dir'], 'model', strict=False)
    model.build_submodules()
    return model


def _fix_cast_nodes(graph, logs=None):
    if logs is None:
        logs = []
    for sub_node in graph.node:
        if sub_node.op_type == 'If':
            for attr in sub_node.attribute:
                branch = onnx.helper.get_attribute_value(attr)
                _fix_cast_nodes(branch, logs)
        elif sub_node.op_type == 'Loop':
            for attr in sub_node.attribute:
                if attr.name == 'body':
                    body = onnx.helper.get_attribute_value(attr)
                    _fix_cast_nodes(body, logs)
        elif sub_node.op_type == 'Cast':
            for i, sub_attr in enumerate(sub_node.attribute):
                if sub_attr.name == 'to':
                    to = onnx.helper.get_attribute_value(sub_attr)
                    if to == onnx.TensorProto.DOUBLE:
                        float32 = onnx.helper.make_attribute('to', onnx.TensorProto.FLOAT)
                        sub_node.attribute.remove(sub_attr)
                        sub_node.attribute.insert(i, float32)
                        logs.append(sub_node.name)
                        break
    return logs


def _fold_shape_gather_equal_if_to_squeeze(graph, subgraph, logs=None):
    if logs is None:
        logs = []

    # Do folding in sub-graphs recursively.
    for node in subgraph.node:
        if node.op_type == 'If':
            for attr in node.attribute:
                branch = onnx.helper.get_attribute_value(attr)
                _fold_shape_gather_equal_if_to_squeeze(graph, branch, logs)
        elif node.op_type == 'Loop':
            for attr in node.attribute:
                if attr.name == 'body':
                    body = onnx.helper.get_attribute_value(attr)
                    _fold_shape_gather_equal_if_to_squeeze(graph, body, logs)

    # Do folding in current graph.
    i_shape = 0
    while i_shape < len(subgraph.node):
        if subgraph.node[i_shape].op_type == 'Shape':
            shape_node = subgraph.node[i_shape]
            shape_out = shape_node.output[0]
            i_gather = i_shape + 1
            while i_gather < len(subgraph.node):
                if subgraph.node[i_gather].op_type == 'Gather' and subgraph.node[i_gather].input[0] == shape_out:
                    gather_node = subgraph.node[i_gather]
                    gather_out = gather_node.output[0]
                    i_equal = i_gather + 1
                    while i_equal < len(subgraph.node):
                        if subgraph.node[i_equal].op_type == 'Equal' and (
                                subgraph.node[i_equal].input[0] == gather_out
                                or subgraph.node[i_equal].input[1] == gather_out):
                            equal_node = subgraph.node[i_equal]
                            equal_out = equal_node.output[0]
                            i_if = i_equal + 1
                            while i_if < len(subgraph.node):
                                if subgraph.node[i_if].op_type == 'If' and subgraph.node[i_if].input[0] == equal_out:
                                    # Found the substructure to be folded.
                                    if_node = subgraph.node[i_if]
                                    # Search and clean initializer values.
                                    squeeze_axes_tensor = None
                                    for tensor in subgraph.initializer:
                                        if tensor.name == gather_node.input[1]:
                                            squeeze_axes_tensor = tensor
                                            subgraph.initializer.remove(tensor)
                                        elif tensor.name == equal_node.input[1]:
                                            subgraph.initializer.remove(tensor)
                                    # Create 'Squeeze' node.
                                    squeeze_node = onnx.helper.make_node(
                                        op_type='Squeeze',
                                        inputs=shape_node.input,
                                        outputs=if_node.output
                                    )
                                    squeeze_axes = onnx.helper.make_attribute(
                                        key='axes',
                                        value=[struct.unpack('q', squeeze_axes_tensor.raw_data)[0]]  # unpack int64
                                    )
                                    squeeze_node.attribute.extend([squeeze_axes])
                                    # Replace 'Shape', 'Gather', 'Equal', 'If' with 'Squeeze'.
                                    subgraph.node.insert(i_shape, squeeze_node)
                                    subgraph.node.remove(shape_node)
                                    subgraph.node.remove(gather_node)
                                    subgraph.node.remove(equal_node)
                                    subgraph.node.remove(if_node)
                                    logs.append((shape_node.name, gather_node.name, equal_node.name, if_node.name))
                                    break
                                i_if += 1
                            else:
                                break
                        i_equal += 1
                    else:
                        break
                i_gather += 1
            else:
                break
        i_shape += 1
    return logs


def _extract_conv_nodes(graph, weight_pattern, alias_prefix):
    node_dict = {}  # key: pattern match, value: (alias, node)
    logs = []

    def _extract_conv_nodes_recursive(subgraph):
        to_be_removed = []
        for sub_node in subgraph.node:
            if sub_node.op_type == 'If':
                for attr in sub_node.attribute:
                    branch = onnx.helper.get_attribute_value(attr)
                    _extract_conv_nodes_recursive(branch)
            elif sub_node.op_type == 'Loop':
                for attr in sub_node.attribute:
                    if attr.name == 'body':
                        body = onnx.helper.get_attribute_value(attr)
                        _extract_conv_nodes_recursive(body)
            elif sub_node.op_type == 'Conv' and re.match(weight_pattern, sub_node.input[1]):
                # Found node to extract
                cached = node_dict.get(sub_node.input[1])
                if cached is None:
                    out_alias = f'{alias_prefix}.{len(node_dict)}'
                    node_dict[sub_node.input[1]] = (out_alias, sub_node)
                else:
                    out_alias = cached[0]
                out = sub_node.output[0]
                # Search for nodes downstream the extracted node and match them to the renamed output
                for dep_node in subgraph.node:
                    for dep_idx, dep_input in enumerate(dep_node.input):
                        if dep_input == out:
                            dep_node.input.remove(out)
                            dep_node.input.insert(dep_idx, out_alias)
                # Add the node to the remove list
                to_be_removed.append(sub_node)
                logs.append(sub_node.name)
        [subgraph.node.remove(_n) for _n in to_be_removed]

    for i, n in enumerate(graph.node):
        if n.op_type == 'If':
            for a in n.attribute:
                b = onnx.helper.get_attribute_value(a)
                _extract_conv_nodes_recursive(b)
            for key in reversed(node_dict):
                alias, node = node_dict[key]
                # Rename output of the node
                out_name = node.output[0]
                node.output.remove(node.output[0])
                node.output.insert(0, alias)
                # Insert node into the main graph
                graph.node.insert(i, node)
                # Rename value info of the output
                for v in graph.value_info:
                    if v.name == out_name:
                        v.name = alias
                        break
            break
    return logs


def _remove_unused_values(graph):
    used_values = set()
    cleaned_values = []

    def _record_usage_recursive(subgraph):
        for node in subgraph.node:
            # For 'If' and 'Loop' nodes, do recording recursively
            if node.op_type == 'If':
                for attr in node.attribute:
                    branch = onnx.helper.get_attribute_value(attr)
                    _record_usage_recursive(branch)
            elif node.op_type == 'Loop':
                for attr in node.attribute:
                    if attr.name == 'body':
                        body = onnx.helper.get_attribute_value(attr)
                        _record_usage_recursive(body)
            # For each node, record its inputs and outputs
            for input_value in node.input:
                used_values.add(input_value)
            for output_value in node.output:
                used_values.add(output_value)

    def _clean_unused_recursively(subgraph):
        # Do cleaning in sub-graphs recursively.
        for node in subgraph.node:
            if node.op_type == 'If':
                for attr in node.attribute:
                    branch = onnx.helper.get_attribute_value(attr)
                    _clean_unused_recursively(branch)
            elif node.op_type == 'Loop':
                for attr in node.attribute:
                    if attr.name == 'body':
                        body = onnx.helper.get_attribute_value(attr)
                        _clean_unused_recursively(body)

        # Do cleaning in current graph.
        i = 0
        while i < len(subgraph.initializer):
            if subgraph.initializer[i].name not in used_values:
                cleaned_values.append(subgraph.initializer[i].name)
                subgraph.initializer.remove(subgraph.initializer[i])
            else:
                i += 1
        i = 0
        while i < len(subgraph.value_info):
            if subgraph.value_info[i].name not in used_values:
                cleaned_values.append(subgraph.value_info[i].name)
                subgraph.value_info.remove(subgraph.value_info[i])
            else:
                i += 1

    _record_usage_recursive(graph)
    _clean_unused_recursively(graph)
    return cleaned_values


def fix(src, target):
    model = onnx.load(src)

    # The output dimension are wrongly hinted by TorchScript
    in_dims = model.graph.input[0].type.tensor_type.shape.dim
    out_dims = model.graph.output[0].type.tensor_type.shape.dim
    out_dims.remove(out_dims[1])
    out_dims.insert(1, in_dims[1])
    print(f'| annotate output: \'{model.graph.output[0].name}\'')

    # Fix 'Cast' nodes in sub-graphs that wrongly cast tensors to float64
    fixed_casts = _fix_cast_nodes(model.graph)
    print('| fix node(s): ')
    for i, log in enumerate(fixed_casts):
        if i == len(fixed_casts) - 1:
            end = '\n'
        elif i % 10 == 9:
            end = ',\n'
        else:
            end = ', '
        print(f'\'{log}\'', end=end)

    # Run #1 of the simplifier to fix missing value info and type hints and remove unnecessary 'Cast'.
    print('Running ONNX simplifier...')
    model, check = onnxsim.simplify(model, include_subgraph=True)
    assert check, 'Simplified ONNX model could not be validated'

    in_dims = model.graph.input[0].type.tensor_type.shape.dim
    out_dims = model.graph.output[0].type.tensor_type.shape.dim

    then_branch = None
    for node in model.graph.node:
        if node.op_type == 'If':
            # Add type hint to let the simplifier fold 'Shape', 'Gather', 'Equal', 'If' to 'Squeeze'
            if_out = node.output[0]
            for info in model.graph.value_info:
                if info.name == if_out:
                    if_out_dim = info.type.tensor_type.shape.dim
                    while len(if_out_dim) > 0:
                        if_out_dim.remove(if_out_dim[0])
                    if_out_dim.insert(0, in_dims[0])  # batch_size
                    if_out_dim.insert(1, in_dims[0])  # 1
                    if_out_dim.insert(2, out_dims[2])  # mel_bins
                    if_out_dim.insert(3, in_dims[1])  # n_frames
                    print(f'| annotate node: \'{node.name}\'')

            # Manually fold 'Shape', 'Gather', 'Equal', 'If' to 'Squeeze' in sub-graphs
            folded_groups = []
            for attr in node.attribute:
                branch = onnx.helper.get_attribute_value(attr)
                folded_groups += _fold_shape_gather_equal_if_to_squeeze(model.graph, branch)
                if attr.name == 'then_branch':
                    # Save branch for future use
                    then_branch = branch
            print('| fold node group(s): ')
            print(', '.join(['[' + ', '.join([f'\'{n}\'' for n in log]) + ']' for log in folded_groups]))
            break

    # Optimize 'Concat' nodes for shapes
    concat_node = None
    shape_prefix_name = 'noise.shape.prefix'
    list_length_name = 'list.length'
    for node in model.graph.node:
        if node.op_type == 'Concat':
            concat_node = node
            for i, ini in enumerate(model.graph.initializer):
                if ini.name == node.input[0]:
                    shape_prefix = onnx.helper.make_tensor(
                        name=shape_prefix_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=(3,),
                        vals=[out_dims[0].dim_value, 1, out_dims[2].dim_value]
                    )
                    list_length = onnx.helper.make_tensor(
                        name=list_length_name,
                        data_type=onnx.TensorProto.INT64,
                        dims=(1,),
                        vals=[0]
                    )
                    model.graph.initializer.extend([shape_prefix, list_length])
                    break
            for i in range(3):
                node.input.remove(node.input[0])
            node.input.insert(0, shape_prefix_name)
            print(f'| optimize node: \'{node.name}\'')
            break
    for node in then_branch.node:
        if node.op_type == 'Concat':
            concat_inputs = list(node.input)
            dep_nodes = []
            for dep_node in then_branch.node:
                if dep_node.op_type == 'Unsqueeze' and dep_node.output[0] in concat_inputs:
                    dep_nodes.append(dep_node)
            [then_branch.node.remove(d_n) for d_n in dep_nodes]
            while len(node.input) > 0:
                node.input.remove(node.input[0])
            node.input.extend([list_length_name, concat_node.output[0]])
            print(f'| optimize node: \'{node.name}\'')
            break

    # Extract 'Conv' nodes and cache results of conditioner projection
    # of each residual layer from loop bodies to improve performance.
    extracted_convs = _extract_conv_nodes(
        model.graph,
        r'model\.denoise_fn\.residual_layers\.\d+\.conditioner_projection\.weight',
        'cache'
    )

    print(f'| extract node(s):')
    for i, log in enumerate(extracted_convs):
        if i == len(extracted_convs) - 1:
            end = '\n'
        elif i % 10 == 9:
            end = ',\n'
        else:
            end = ', '
        print(f'\'{log}\'', end=end)

    # Remove unused initializers and value infos
    cleaned_values = _remove_unused_values(model.graph)
    print(f'| clean value(s):')
    for i, log in enumerate(cleaned_values):
        if i == len(cleaned_values) - 1:
            end = '\n'
        elif i % 15 == 14:
            end = ',\n'
        else:
            end = ', '
        print(f'\'{log}\'', end=end)

    # Run #2 of the simplifier to further optimize the graph and reduce dangling sub-graphs.
    print('Running ONNX simplifier...')
    model, check = onnxsim.simplify(model, include_subgraph=True)
    assert check, 'Simplified ONNX model could not be validated'

    onnx.save(model, target)
    print('Graph fixed and optimized.')


def export(model_path):
    set_hparams(print_hparams=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    decoder = DiffDecoder(device)
    n_frames = 10

    with torch.no_grad():
        shape = (1, 1, hparams['audio_num_mel_bins'], n_frames)
        noise_t = torch.randn(shape, device=device)
        noise_list = torch.randn((3, *shape), device=device)
        condition = torch.rand((1, hparams['hidden_size'], n_frames), device=device)
        step = (torch.rand((), device=device) * hparams['K_step']).long()
        speedup = (torch.rand((), device=device) * step / 10.).long()
        step_prev = torch.maximum(step - speedup, torch.tensor(0, dtype=torch.long, device=device))

        print('Tracing modules...')
        decoder.model.denoise_fn = torch.jit.trace(
            decoder.model.denoise_fn,
            (
                noise_t,
                step,
                condition
            )
        )
        decoder.model.naive_noise_predictor = torch.jit.trace(
            decoder.model.naive_noise_predictor,
            (
                noise_t,
                noise_t,
                step
            ),
            check_trace=False
        )
        decoder.model.plms_noise_predictor = torch.jit.trace_module(
            decoder.model.plms_noise_predictor,
            {
                'forward': (
                    noise_t,
                    noise_t,
                    step,
                    step_prev
                ),
                'predict_stage0': (
                    noise_t,
                    noise_t
                ),
                'predict_stage1': (
                    noise_t,
                    noise_list
                ),
                'predict_stage2': (
                    noise_t,
                    noise_list
                ),
                'predict_stage3': (
                    noise_t,
                    noise_list
                ),
            }
        )
        decoder.model.mel_extractor = torch.jit.trace(
            decoder.model.mel_extractor,
            (
                noise_t
            )
        )

        decoder = torch.jit.script(decoder)
        condition = torch.rand((1, n_frames, hparams['hidden_size']), device=device)
        speedup = torch.tensor(10, dtype=torch.long, device=device)
        dummy = decoder.forward(condition, speedup)

        torch.onnx.export(
            decoder,
            (
                condition,
                speedup
            ),
            model_path,
            input_names=[
                'condition',
                'speedup'
            ],
            output_names=[
                'mel'
            ],
            dynamic_axes={
                'condition': {
                    1: 'n_frames'
                }
            },
            opset_version=11,
            example_outputs=(
                dummy
            )
        )
        print('PyTorch ONNX export finished.')


if __name__ == '__main__':
    exp = '1110_opencpop_ds1000_m128_n512x20'
    sys.argv = [
        f'inference/ds_cascade.py',
        '--config',
        f'checkpoints/{exp}/config.yaml',
        '--exp_name',
        exp
    ]
    path = f'onnx/assets/{exp}.onnx'
    export(path)
    fix(path, path)
    print(f'| export \'model\' to \'{path}\'.')
