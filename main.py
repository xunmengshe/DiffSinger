# coding=utf8
import argparse
import json
import os
import sys

import numpy as np
import torch

from crossfade import cross_fade
from inference.svs.ds_cascade import DiffSingerCascadeInfer
from inference.svs.ds_e2e import DiffSingerE2EInfer
from utils.audio import save_wav
from utils.hparams import set_hparams, hparams

root_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PYTHONPATH'] = f'"{root_dir}"'

parser = argparse.ArgumentParser(description='Run DiffSinger inference')
parser.add_argument('proj', type=str, help='Path to the input file')
parser.add_argument('--exp', type=str, required=False, help='Selection of model')
parser.add_argument('--out', type=str, required=False, help='Path of the output folder')
parser.add_argument('--title', type=str, required=False, help='Title of output file')
parser.add_argument('--num', type=int, required=False, default=1, help='Number of runs')
parser.add_argument('--seed', type=int, required=False, help='Random seed of the inference')
parser.add_argument('--speedup', type=int, required=False, default=0, help='PNDM speed-up ratio')
parser.add_argument('--pitch', action='store_true', required=False, default=False, help='Enable manual pitch mode')
args = parser.parse_args()

name = os.path.basename(args.proj).split('.')[0] if not args.title else args.title
exp = args.exp
if not exp:
    if args.pitch:
        exp = '0909_opencpop_ds100_pitchcontrol'
    elif os.path.exists(os.path.join(root_dir, 'checkpoints/0814_opencpop_ds_rhythm_fix')):
        exp = '0814_opencpop_ds_rhythm_fix'
    else:
        exp = '0814_opencpop_500k（修复无参音素）'
out = args.out
if not out:
    out = os.path.dirname(os.path.abspath(args.proj))

sys.argv = [
    f'{root_dir}/inference/svs/ds_e2e.py' if not args.pitch else f'{root_dir}/inference/svs/ds_cascade.py',
    '--config',
    f'{root_dir}/usr/configs/midi/e2e/opencpop/ds100_adj_rel.yaml' if not args.pitch else f'{root_dir}/usr/configs/midi/cascade/opencs/ds100_rel.yaml',
    '--exp_name',
    exp
]

if args.speedup > 0:
    sys.argv += ['--hparams', f'pndm_speedup={args.speedup}']

with open(args.proj, 'r', encoding='utf-8') as f:
    params = json.load(f)

if not isinstance(params, list):
    params = [params]

set_hparams(print_hparams=False)
sample_rate = hparams['audio_sample_rate']

infer_ins = None
if len(params) > 0:
    infer_ins = DiffSingerE2EInfer(hparams) if not args.pitch else DiffSingerCascadeInfer(hparams)


def infer_once(path: str):
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
        seg_audio = infer_ins.infer_once(param)
        silent_length = round(param.get('offset', 0) * sample_rate) - current_length
        if silent_length >= 0:
            result = np.append(result, np.zeros(silent_length))
            result = np.append(result, seg_audio)
        else:
            result = cross_fade(result, seg_audio, current_length + silent_length)
        current_length = current_length + silent_length + seg_audio.shape[0]
    print(f'| save audio: {path}')
    save_wav(result, path, sample_rate)


os.makedirs(out, exist_ok=True)
if args.num == 1:
    infer_once(os.path.join(out, f'{name}.wav'))
else:
    for i in range(1, args.num + 1):
        infer_once(os.path.join(out, f'{name}-{str(i).zfill(3)}.wav'))
