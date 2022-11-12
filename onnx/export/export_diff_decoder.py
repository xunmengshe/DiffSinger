import math
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


def traceit(func):
    def run(*args, **kwargs):
        print(f'Invoking function \'{func.__name__}\'')
        res = func(*args, **kwargs)
        return res

    return run


def extract(a, t):
    return a[t].reshape((1, 1, 1, 1))
    # return a.gather(-1, t).reshape((1, 1, 1, 1))
    # b, *_ = t.shape
    # out = a.gather(-1, t)
    # shape = (1, *((1,) * (len(x_shape) - 1)))
    # print(b, shape)
    # return out.reshape((1, 1, 1, 1))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


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
        # Make an initializer
        # emb = torch.from_numpy(np.exp(np.arange(half_dim) * -emb)[None, :].astype(np.float32)).to(device)

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
        # gate, filter = torch.chunk(y, 2, dim=1)
        # gate, filter = y[:, :self.residual_channels, :], y[:, self.residual_channels:, :]

        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        # Using torch.split instead of torch.chunk to avoid using onnx::Slice
        residual, skip = torch.split(y, [self.residual_channels, self.residual_channels], dim=1)
        # residual, skip = torch.chunk(y, 2, dim=1)
        # residual, skip = y[:, :self.residual_channels, :], y[:, self.residual_channels:, :]

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

        x_delta = (a_prev - a_t) * ((1. / (a_t_sq * (a_t_sq + a_prev_sq))) * x - 1. / (
                a_t_sq * (((1. - a_prev) * a_t).sqrt() + ((1. - a_t) * a_prev).sqrt())) * noise_t)
        x_pred = x + x_delta

        return x_pred

    # noinspection PyMethodMayBeStatic
    def predict_stage0(self, noise_pred, noise_pred_prev):
        return (noise_pred
                + noise_pred_prev) / self._2

    # noinspection PyMethodMayBeStatic
    def predict_stage1(self, noise_pred, noise_list):
        return (noise_pred * self._3
                - noise_list[-1]) / self._2

    # noinspection PyMethodMayBeStatic
    def predict_stage2(self, noise_pred, noise_list):
        return (noise_pred * self._23
                - noise_list[-1] * self._16
                + noise_list[-2] * self._5) / self._12

    # noinspection PyMethodMayBeStatic
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

    def build_submodule(self):
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
        noise_list = torch.zeros((0, 1, 1, self.mel_bins, n_frames), device=device)

        if speedup > 1:
            plms_noise_stage = torch.tensor(0, dtype=torch.long, device=device)
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
    model.build_submodule()
    return model


def _fix_cast_nodes(graph):
    for sub_node in graph.node:
        if sub_node.op_type == 'If':
            for attr in sub_node.attribute:
                branch = onnx.helper.get_attribute_value(attr)
                _fix_cast_nodes(branch)
        elif sub_node.op_type == 'Loop':
            for attr in sub_node.attribute:
                if attr.name == 'body':
                    body = onnx.helper.get_attribute_value(attr)
                    _fix_cast_nodes(body)
        elif sub_node.op_type == 'Cast':
            for i, sub_attr in enumerate(sub_node.attribute):
                if sub_attr.name == 'to':
                    to = onnx.helper.get_attribute_value(sub_attr)
                    if to == onnx.TensorProto.DOUBLE:
                        float32 = onnx.helper.make_attribute('to', onnx.TensorProto.FLOAT)
                        sub_node.attribute.remove(sub_attr)
                        sub_node.attribute.insert(i, float32)
                        print(f'Fix node: \'{sub_node.name}\'')
                        break


def fix(src, target):
    model = onnx.load(src)

    # The output dimension are wrongly hinted by TorchScript
    in_dims = model.graph.input[0].type.tensor_type.shape.dim
    out_dims = model.graph.output[0].type.tensor_type.shape.dim
    out_dims.remove(out_dims[1])
    out_dims.insert(1, in_dims[1])
    print(f'Fix output: \'{model.graph.output[0].name}\'')

    # Fix 'Cast' nodes in sub-graphs that wrongly cast tensors to float64
    _fix_cast_nodes(model.graph)

    # Run #1 of the simplifier to fix missing value info and type hints and remove unnecessary 'Cast'.
    model, check = onnxsim.simplify(model, include_subgraph=True)
    assert check, 'Simplified ONNX model could not be validated'

    # Add type hint to let the simplifier wrap 'Shape', 'Gather', 'Equal', 'If' to 'Squeeze'
    in_dims = model.graph.input[0].type.tensor_type.shape.dim
    out_dims = model.graph.output[0].type.tensor_type.shape.dim
    for node in model.graph.node:
        if node.op_type == 'If':
            if_out = node.output[0]
            for info in model.graph.value_info:
                if info.name == if_out:
                    loop_out_dim = info.type.tensor_type.shape.dim
                    while len(loop_out_dim) > 0:
                        loop_out_dim.remove(loop_out_dim[0])
                    loop_out_dim.insert(0, in_dims[0])  # batch_size
                    loop_out_dim.insert(1, in_dims[0])  # 1
                    loop_out_dim.insert(2, out_dims[2])  # mel_bins
                    loop_out_dim.insert(3, in_dims[1])  # n_frames
                    print(f'Fix node: \'{node.name}\'')
            break

            # for attr in node.attribute:
            #     if attr.name == 'body':
            #         body = onnx.helper.get_attribute_value(attr)
            #
            #         # Make input dimension hints.
            #         sub_input_name = None
            #         for sub_input in body.input:
            #             sub_dims = sub_input.type.tensor_type.shape.dim
            #             if len(sub_dims) == 4:
            #                 sub_input_name = sub_input.name
            #                 sub_dims.remove(sub_dims[0])
            #                 sub_dims.insert(0, in_dims[0])  # batch_size
            #                 sub_dims.remove(sub_dims[1])
            #                 sub_dims.insert(1, in_dims[0])  # 1
            #                 sub_dims.remove(sub_dims[2])
            #                 sub_dims.insert(2, out_dims[2])  # mel_bins
            #                 sub_dims.remove(sub_dims[3])
            #                 sub_dims.insert(3, in_dims[1])  # n_frames
            #                 print(f'Fix input: \'{sub_input.name}\'')
            #
            #         # Wrap 'Shape', 'Gather', 'Equal', 'If' to 'Squeeze'.
            #         num = None
            #         squeeze_index = None
            #         squeeze_input = None
            #         squeeze_output = None
            #
            #         shape_node, gather_node, equal_node, if_node = None, None, None, None
            #
            #         for node_index, sub_node in enumerate(body.node):
            #             if sub_node.op_type == 'Shape':
            #                 for shape_input in sub_node.input:
            #                     if shape_input == sub_input_name:
            #                         num = sub_node.name.split('_')[1]
            #                         squeeze_index = node_index
            #                         squeeze_input = sub_node.input
            #                         shape_node = sub_node
            #                         for sub_node2 in body.node:
            #                             if sub_node2.op_type == 'Gather':
            #                                 for gather_input in sub_node2.input:
            #                                     if gather_input == shape_node.output[0]:
            #                                         gather_node = sub_node2
            #                                         break
            #                         break
            #                 else:
            #                     break
            #         for sub_node in body.node:
            #             if sub_node.op_type == 'If':
            #                 squeeze_output = sub_node.output
            #                 if_node = sub_node
            #                 for sub_node2 in body.node:
            #                     if sub_node2.op_type == 'Equal':
            #                         for equal_output in sub_node2.output:
            #                             if equal_output == if_node.input[0]:
            #                                 equal_node = sub_node2
            #                                 break
            #                 break
            #
            #         squeeze = onnx.helper.make_node(
            #             op_type='Squeeze',
            #             name=f'Squeeze_{num}',
            #             inputs=squeeze_input,
            #             outputs=squeeze_output
            #         )
            #
            #         axes = onnx.helper.make_attribute('axes', [1])
            #         squeeze.attribute.extend([axes])
            #         body.node.insert(squeeze_index, squeeze)
            #         body.node.remove(shape_node)
            #         body.node.remove(gather_node)
            #         body.node.remove(equal_node)
            #         body.node.remove(if_node)
            #         print(f'Fix nodes: '
            #               f'\'{shape_node.name}\', \'{gather_node.name}\', \'{equal_node.name}\', \'{if_node.name}\'')

    # Run #2 of the simplifier to further optimize the graph and reduce dangling sub-graphs.
    model, check = onnxsim.simplify(model, include_subgraph=True)
    assert check, 'Simplified ONNX model could not be validated'

    onnx.save(model, target)


def export(path):
    set_hparams(print_hparams=True)
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

        # torch.onnx.export(
        #     decoder.model.denoise_fn,
        #     (
        #         noise_t,
        #         step,
        #         condition
        #     ),
        #     'onnx/assets/diffnet.onnx',
        #     input_names=[
        #         'noise_t',
        #         'step',
        #         'condition'
        #     ],
        #     output_names=[
        #         'denoised'
        #     ],
        #     dynamic_axes={
        #         'noise_t': {
        #             3: 'n_frames',
        #         },
        #         'condition': {
        #             2: 'n_frames'
        #         }
        #     },
        #     opset_version=11
        # )

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
            path,
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


if __name__ == '__main__':
    exp = '1104_opencpop_ds1000_m128_n384x20'
    sys.argv = [
        f'inference/ds_cascade.py',
        '--config',
        f'checkpoints/{exp}/config.yaml',
        '--exp_name',
        exp
    ]
    export(f'onnx/assets/plms.onnx')
    # fix(f'onnx/assets/{exp}.onnx', f'onnx/assets/{exp}.onnx')
    fix('onnx/assets/plms.onnx', 'onnx/assets/plms.onnx')
