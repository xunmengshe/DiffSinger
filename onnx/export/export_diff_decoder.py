import math
import struct
import sys
from collections import deque
from functools import partial

import numpy as np
import onnx
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
    return a.gather(-1, t).reshape((1, 1, 1, 1))
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
        self.register_buffer('emb', torch.exp(torch.arange(half_dim) * -emb).unsqueeze(0))

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

        return (residual + skip) / math.sqrt(2), skip


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


class GaussianDiffusion(nn.Module):
    def __init__(self, out_dims, timesteps=1000, k_step=1000, spec_min=None, spec_max=None):
        super().__init__()
        self.mel_bins = out_dims
        self.denoise_fn = DiffNet(out_dims)

        # if exists(betas):
        #     betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        # else:
        if 'schedule_type' in hparams.keys():
            betas = beta_schedule[hparams['schedule_type']](timesteps)
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.K_step = k_step

        self.noise_list = deque(maxlen=4)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self.register_buffer('spec_min', torch.FloatTensor(spec_min)[None, None, :hparams['keep_bins']])
        self.register_buffer('spec_max', torch.FloatTensor(spec_max)[None, None, :hparams['keep_bins']])

        self.register_buffer('step_range', torch.arange(0, k_step, dtype=torch.long).flip(0).unsqueeze(1))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t) * x_start +
                extract(self.posterior_mean_coef2, t) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond):
        noise_pred = self.denoise_fn(x, t, cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        # if clip_denoised:
        x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond):
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = ((t > 0).float()).reshape(1, 1, 1, 1)
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_plms(self, x, t, interval, cond, clip_denoised=True, repeat_noise=False):
        """
        Use the PLMS method from [Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778).
        """

        def get_x_pred(x, noise_t, t):
            a_t = extract(self.alphas_cumprod, t)
            a_prev = extract(self.alphas_cumprod, torch.max(t - interval, torch.zeros_like(t)))
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

            x_delta = (a_prev - a_t) * ((1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x - 1 / (
                        a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt())) * noise_t)
            x_pred = x + x_delta

            return x_pred

        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond)

        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(x_pred, max(t - interval, 0), cond)
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(noise_list) == 1:
            noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
        elif len(noise_list) == 2:
            noise_pred_prime = (23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]) / 12
        else:
            noise_pred_prime = (55 * noise_pred - 59 * noise_list[-1] + 37 * noise_list[-2] - 9 * noise_list[-3]) / 24

        x_prev = get_x_pred(x, noise_pred_prime, t)
        noise_list.append(noise_pred)

        return x_prev

    def forward(self, condition):
        '''
            conditioning diffusion, use fastspeech2 encoder output as the condition
        '''
        # ret = self.fs2(txt_tokens, mel2ph, spk_embed, ref_mels, f0, uv, energy,
        #                skip_decoder=True, infer=infer, **kwargs)
        # cond = ret['decoder_inp'].transpose(1, 2)
        condition = condition.transpose(1, 2)  # (1, n_frames, 256) => (1, 256, n_frames)
        device = condition.device

        t = self.K_step
        # print('===> gaussion start.')
        x = torch.randn((1, 1, self.mel_bins, condition.shape[2]), device=device)
        # if hparams.get('pndm_speedup') and hparams['pndm_speedup'] > 1:
        #     # obsolete: pndm_speedup, now use dpm_solver.
        #     # self.noise_list = deque(maxlen=4)
        #     # iteration_interval = hparams['pndm_speedup']
        #     # for i in tqdm(reversed(range(0, t, iteration_interval)), desc='sample time step',
        #     #               total=t // iteration_interval):
        #     #     x = self.p_sample_plms(x, torch.full((b,), i, device=device, dtype=torch.long), iteration_interval,
        #     #                            cond)
        #
        #     from inference.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
        #     ## 1. Define the noise schedule.
        #     noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)
        #
        #     ## 2. Convert your discrete-time `model` to the continuous-time
        #     # noise prediction model. Here is an example for a diffusion model
        #     ## `model` with the noise prediction type ("noise") .
        #     def my_wrapper(fn):
        #         def wrapped(x, t, **kwargs):
        #             ret = fn(x, t, **kwargs)
        #             self.bar.update(1)
        #             return ret
        #
        #         return wrapped
        #
        #     model_fn = model_wrapper(
        #         my_wrapper(self.denoise_fn),
        #         noise_schedule,
        #         model_type="noise",  # or "x_start" or "v" or "score"
        #         model_kwargs={"cond": condition}
        #     )
        #
        #     ## 3. Define dpm-solver and sample by singlestep DPM-Solver.
        #     ## (We recommend singlestep DPM-Solver for unconditional sampling)
        #     ## You can adjust the `steps` to balance the computation
        #     ## costs and the sample quality.
        #     dpm_solver = DPM_Solver(model_fn, noise_schedule)
        #
        #     steps = t // hparams["pndm_speedup"]
        #     self.bar = tqdm(desc="sample time step", total=steps)
        #     x = dpm_solver.sample(
        #         x,
        #         steps=steps,
        #         order=3,
        #         skip_type="time_uniform",
        #         method="singlestep",
        #     )
        # else:

        # for i in tqdm(reversed(range(0, t)), desc='sample time step', total=t):
        for i in self.step_range:
            # x = x + i.float() / 1000.0  # Dummy network
            x = self.p_sample(x, i, condition)
        x = x.squeeze(1).permute(0, 2, 1)
        mel = self.denorm_spec(x)
        return mel

    def denorm_spec(self, x):
        d = (self.spec_max - self.spec_min) / 2
        m = (self.spec_max + self.spec_min) / 2
        return x * d + m


class DiffDecoder(nn.Module):
    def __init__(self, hparams, device):
        super().__init__()
        self.hparams = hparams
        self.device = device
        self.model = build_model()
        self.model.eval()
        self.model.to(self.device)

    def forward(self, condition):
        with torch.no_grad():
            mel = self.model.forward(condition)  # (1, n_frames, mel_bins)
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
    return model


def fix(src, target):
    model = onnx.load(src)
    for node in model.graph.node:
        if node.name.startswith('Loop'):
            for attr in node.attribute:
                if attr.name == 'body':
                    body = onnx.helper.get_attribute_value(attr)
                    for sub_node in body.node:
                        if sub_node.name.startswith('Cast'):
                            for i, sub_attr in enumerate(sub_node.attribute):
                                if sub_attr.name == 'to':
                                    to = onnx.helper.get_attribute_value(sub_attr)
                                    if to == onnx.TensorProto.DOUBLE:  # float64
                                        float32 = onnx.helper.make_attribute('to', onnx.TensorProto.FLOAT)
                                        sub_node.attribute.remove(sub_attr)
                                        sub_node.attribute.insert(i, float32)
                                        print(f'Fixed node: \'{sub_node.name}\'')
                        elif sub_node.name.startswith('Clip'):
                            min_val, max_val = sub_node.input[1], sub_node.input[2]
                            for top_node in model.graph.node:
                                if top_node.name.startswith('Constant') and top_node.output[0] in [min_val, max_val]:
                                    tensor = onnx.helper.get_attribute_value(top_node.attribute[0])
                                    if tensor.data_type == onnx.TensorProto.DOUBLE:
                                        value = struct.unpack('d', tensor.raw_data)[0]
                                        tensor.data_type = onnx.TensorProto.FLOAT
                                        tensor.raw_data = struct.pack('f', value)
                                        print(f'Fixed node \'{top_node.name}\'')
    # TODO: fix wrong output dimension hint
    onnx.checker.check_model(model)
    onnx.save(model, target)


def main():
    sys.argv = [
        f'inference/ds_cascade.py',
        '--config',
        f'checkpoints/1106_opencpop_ds1000_bin128/config.yaml',
        '--exp_name',
        '1106_opencpop_ds1000_bin128'
    ]

    set_hparams(print_hparams=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    decoder = DiffDecoder(hparams, device)
    n_frames = 10

    with torch.no_grad():
        noised = torch.rand((1, 1, 128, n_frames), device=device)
        condition = torch.rand((1, 256, n_frames), device=device)
        step = torch.full((1,), 114514, dtype=torch.long, device=device)

        # torch.onnx.export(
        #     decoder.model.denoise_fn,
        #     (
        #         noised,
        #         step,
        #         condition
        #     ),
        #     'onnx/assets/diffnet.onnx',
        #     input_names=[
        #         'noised',
        #         'step',
        #         'condition'
        #     ],
        #     output_names=[
        #         'denoised'
        #     ],
        #     dynamic_axes={
        #         'noised': {
        #             3: 'n_frames',
        #         },
        #         'condition': {
        #             2: 'n_frames'
        #         }
        #     },
        #     opset_version=11
        # )

        decoder.model.denoise_fn = torch.jit.trace(decoder.model.denoise_fn, (noised, step, condition))
        decoder = torch.jit.script(decoder)
        condition = torch.rand((1, n_frames, 256), device=device)
        dummy = decoder.forward(condition)

        torch.onnx.export(
            decoder,
            (
                condition,
            ),
            'onnx/assets/diff_decoder.onnx',
            input_names=[
                'condition'
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
    main()
    fix('onnx/assets/diff_decoder.onnx', 'onnx/assets/diff_decoder.onnx')
