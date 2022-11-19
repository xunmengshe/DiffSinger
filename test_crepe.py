from utils.hparams import set_hparams, hparams
import torch
from src.vocoders.hifigan import HifiGAN
import torchcrepe
import sys
import resampy
import numpy as np
from utils.audio import save_wav
sys.argv = [
    'inference/ds_e2e.py',
    '--config',
    'configs/midi/e2e/opencpop/ds100_adj_rel.yaml',
    '--exp_name',
    '0909'
]

def get_pitch(wav_data, mel, hparams, threshold=0.3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #crepe只支持16khz采样率，需要重采样
    wav16k = resampy.resample(wav_data, hparams['audio_sample_rate'], 16000)
    wav16k_torch = torch.FloatTensor(wav16k).unsqueeze(0).to(device)
    
    #频率范围
    f0_min = 50
    f0_max = 800
    
    #重采样后按照hopsize=80,也就是5ms一帧分析f0    
    f0, pd = torchcrepe.predict(wav16k_torch, 16000, 80, f0_min, f0_max, pad=True, model='full', batch_size=1024, device=device, return_periodicity=True)
    
    #滤波，去掉静音，设置uv阈值，参考原仓库readme
    pd = torchcrepe.filter.median(pd, 3)
    pd = torchcrepe.threshold.Silence(-60.)(pd, wav16k_torch, 16000, 80)
    f0 = torchcrepe.threshold.At(threshold)(f0, pd)
    f0 = torchcrepe.filter.mean(f0, 3)
    
    #将nan频率（uv部分）转换为0频率
    f0 = torch.where(torch.isnan(f0), torch.full_like(f0,0), f0)
    
    '''
    np.savetxt('问棋-crepe.csv',np.array([0.005*np.arange(len(f0[0])),f0[0].cpu().numpy()]).transpose(),delimiter=',')
    '''
    
    #去掉0频率，并线性插值
    nzindex = torch.nonzero(f0[0]).squeeze()
    f0 = torch.index_select(f0[0],dim=0, index=nzindex).cpu().numpy()
    time_org = 0.005*nzindex.cpu().numpy()
    time_frame = np.arange(len(mel))*hparams['hop_size']/hparams['audio_sample_rate']
    f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
    return f0

set_hparams()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocoder = HifiGAN()
wav, mel = vocoder.wav2spec("infer_out/example_out.wav")
f0 = get_pitch(wav, mel, hparams,threshold=0.05)

with torch.no_grad():
    c = torch.FloatTensor(mel).unsqueeze(0).transpose(2, 1).to(device)
    f0_cp = torch.FloatTensor(f0[None, :]).to(device)
    wav_out = vocoder.model(c, f0_cp).view(-1).cpu().numpy()
    save_wav(wav_out, 'infer_out/test-crepe.wav', hparams['audio_sample_rate'])






