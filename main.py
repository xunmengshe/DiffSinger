# coding=utf8
import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

from utils.infer_utils import cross_fade, trans_key
from inference.ds_cascade import DiffSingerCascadeInfer
from inference.ds_e2e import DiffSingerE2EInfer
from utils.audio import save_wav
from utils.hparams import set_hparams, hparams
from utils.slur_utils import merge_slurs

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
parser.add_argument('--pitch', action='store_true', required=False, default=None, help='Enable manual pitch mode')
parser.add_argument('--forced_automatic_pitch_mode', action='store_true', required=False, default=False)
parser.add_argument('--mel', action='store_true', required=False, default=False,
                    help='Save intermediate mel format instead of waveform')
args = parser.parse_args()

# Deprecation for --pitch
warnings.filterwarnings(action='default')
if args.pitch is not None:
    warnings.warn(
        message='The argument \'--pitch\' is deprecated and will be removed in the future. '
                'The program now automatically detects which mode to use.',
        category=DeprecationWarning,
    )
    warnings.filterwarnings(action='default')

name = os.path.basename(args.proj).split('.')[0] if not args.title else args.title
exp = args.exp
assert exp is not None, 'Default value of exp is deprecated. You must specify \'--exp\' to run inference.'
if not os.path.exists(f'{root_dir}/checkpoints/{exp}'):
    for ckpt in os.listdir(os.path.join(root_dir, 'checkpoints')):
        if ckpt.startswith(exp):
            print(f'| match ckpt by prefix: {ckpt}')
            exp = ckpt
            break
    assert os.path.exists(f'{root_dir}/checkpoints/{exp}'), 'There are no matching exp in \'checkpoints\' folder. ' \
                                                            'Please specify \'--exp\' as the folder name or prefix.'
else:
    print(f'| found ckpt by name: {exp}')


out = args.out
if not out:
    out = os.path.dirname(os.path.abspath(args.proj))

sys.argv = [
    f'{root_dir}/inference/ds_e2e.py' if not args.pitch else f'{root_dir}/inference/ds_cascade.py',
    '--exp_name',
    exp,
    '--infer'
]

if args.speedup > 0:
    sys.argv += ['--hparams', f'pndm_speedup={args.speedup}']

with open(args.proj, 'r', encoding='utf-8') as f:
    params = json.load(f)
if not isinstance(params, list):
    params = [params]

if args.key != 0:
    params = trans_key(params, args.key)
    if not args.title:
        name += f'_{args.key}key'
    print(f"音调基于原音频{args.key}key")

set_hparams(print_hparams=False)
sample_rate = hparams['audio_sample_rate']

# Check for vocoder path
assert os.path.exists(os.path.join(root_dir, hparams['vocoder_ckpt'])), \
    f'Vocoder ckpt \'{hparams["vocoder_ckpt"]}\' not found. ' \
    f'Please put it to the checkpoints directory to run inference.'

infer_ins = None
if len(params) > 0:
    if hparams['use_pitch_embed']:
        infer_ins = DiffSingerCascadeInfer(hparams, load_vocoder=not args.mel)
    else:
        warnings.warn(
            message='SVS MIDI-B version (implicit pitch prediction) is deprecated. '
            'Please select or train a model of MIDI-A version (controllable pitch prediction).',
            category=DeprecationWarning
        )
        warnings.filterwarnings(action='default')
        infer_ins = DiffSingerE2EInfer(hparams, load_vocoder=not args.mel)


def infer_once(path: str, save_mel=False):
    if save_mel:
        result = []
    else:
        result = np.zeros(0)
    current_length = 0

    for i, param in enumerate(params):
        # Ban automatic pitch mode by default
        param_have_f0 = 'f0_seq' in param and param['f0_seq']
        if hparams['use_pitch_embed'] and not param_have_f0:
            if not args.forced_automatic_pitch_mode:
                assert param_have_f0, 'You are using automatic pitch mode which may not produce satisfactory ' \
                                      'results. When you see this message, it is very likely that you forgot to ' \
                                      'freeze the f0 sequence into the input file, and this error is to inform ' \
                                      'you that a double-check should be applied. If you do want to test out the ' \
                                      'automatic pitch mode, please force it on manually.'
            warnings.warn(
                message='You are using forced automatic pitch mode. As this mode is only for testing purpose, '
                        'please note that you must know clearly what you are doing, and be aware that the result '
                        'may not be satisfactory.',
                category=UserWarning
            )
            warnings.filterwarnings(action='default')
            param['f0_seq'] = None

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

        if not hparams.get('use_midi', False):
            merge_slurs(param)
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
        sys.stdout.flush()
        print('| finish segment: %d/%d (%.2f%%)' % (i + 1, len(params), (i + 1) / len(params) * 100))

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
