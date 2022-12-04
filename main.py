# coding=utf8
import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

from inference.infer_utils import cross_fade, trans_key
from inference.ds_cascade import DiffSingerCascadeInfer
from inference.ds_e2e import DiffSingerE2EInfer
from utils.audio import save_wav
from utils.hparams import set_hparams, hparams

sys.path.insert(0, '/')
root_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PYTHONPATH'] = f'"{root_dir}"'

parser = argparse.ArgumentParser(description='Run DiffSinger inference')
parser.add_argument('proj', type=str, help='Path to the input file')
parser.add_argument('--exp', type=str, required=False, help='Selection of model')
parser.add_argument('--out', type=str, required=False, help='Path of the output folder')
parser.add_argument('--title', type=str, required=False, help='Title of output file')
parser.add_argument('--num', type=int, required=False, default=1, help='Number of runs')
parser.add_argument('--key', type=int, required=False, default=0, help='Number of key')
parser.add_argument('--seed', type=int, required=False, help='Random seed of the inference')
parser.add_argument('--speedup', type=int, required=False, default=0, help='PNDM speed-up ratio')
parser.add_argument('--pitch', action='store_true', required=False, default=False, help='Enable manual pitch mode')
parser.add_argument('--mel', action='store_true', required=False, default=False,
                    help='Save intermediate mel format instead of waveform')
args = parser.parse_args()

name = os.path.basename(args.proj).split('.')[0] if not args.title else args.title
exp = args.exp
assert exp is not None, 'Default value of exp is deprecated. You must specify \'--exp\' to run inference.'

out = args.out
if not out:
    out = os.path.dirname(os.path.abspath(args.proj))

sys.argv = [
    f'{root_dir}/inference/ds_e2e.py' if not args.pitch else f'{root_dir}/inference/ds_cascade.py',
    '--config',
    f'{root_dir}/checkpoints/{exp}/config.yaml',
    '--exp_name',
    exp
]

if args.speedup > 0:
    sys.argv += ['--hparams', f'pndm_speedup={args.speedup}']

with open(args.proj, 'r', encoding='utf-8') as f:
    params = json.load(f)
    if args.key != 0:
        params = trans_key(params, args.key)
        if not args.title:
            name += f'_{args.key}key'
        print(f"音调基于原音频{args.key}key")

if not isinstance(params, list):
    params = [params]

set_hparams(print_hparams=False)
sample_rate = hparams['audio_sample_rate']

infer_ins = None
if len(params) > 0:
    if args.pitch:
        infer_ins = DiffSingerCascadeInfer(hparams, load_vocoder=not args.mel)
    else:
        warnings.warn(
            message='SVS MIDI-B version (implicit pitch prediction) is deprecated and '
            'the \'--pitch\' argument will default to true and be removed in the future. '
            'Please select or train a checkpoint of MIDI-A version (controllable pitch prediction).',
            category=DeprecationWarning
        )
        infer_ins = DiffSingerE2EInfer(hparams, load_vocoder=not args.mel)


def infer_once(path: str, save_mel=False):
    if save_mel:
        result = []
    else:
        result = np.zeros(0)
    current_length = 0

    for param in params:
        if 'seed' in param:
            print(f'| set seed: {param["seed"] & 0xffff_ffff}')
            torch.manual_seed(param["seed"] & 0xffff_ffff)
            torch.cuda.manual_seed_all(param["seed"] & 0xffff_ffff)
        elif args.seed:
            print(f'| set seed: {args.seed & 0xffff_ffff}')
            torch.manual_seed(args.seed & 0xffff_ffff)
            torch.cuda.manual_seed_all(args.seed & 0xffff_ffff)
        else:
            torch.manual_seed(torch.seed() & 0xffff_ffff)
            torch.cuda.manual_seed_all(torch.seed() & 0xffff_ffff)

        if save_mel:
            mel, f0 = infer_ins.infer_once(param, return_mel=True)
            result.append({
                'offset': param.get('offset', 0.),
                'mel': mel,
                'f0': f0
            })
        else:
            seg_audio = infer_ins.infer_once(param)
            silent_length = round(param.get('offset', 0) * sample_rate) - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, seg_audio)
            else:
                result = cross_fade(result, seg_audio, current_length + silent_length)
            current_length = current_length + silent_length + seg_audio.shape[0]

    if save_mel:
        print(f'| save mel: {path}')
        torch.save(result, path)
    else:
        print(f'| save audio: {path}')
        save_wav(result, path, sample_rate)


os.makedirs(out, exist_ok=True)
suffix = '.wav' if not args.mel else '.mel.pt'
if args.num == 1:
    infer_once(os.path.join(out, f'{name}{suffix}'), save_mel=args.mel)
else:
    for i in range(1, args.num + 1):
        infer_once(os.path.join(out, f'{name}-{str(i).zfill(3)}{suffix}'), save_mel=args.mel)
