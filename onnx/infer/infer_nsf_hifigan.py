import time

import numpy as np
import onnxruntime as ort
from scipy.io import wavfile

mel = np.load('onnx/assets/mel.npy')
f0 = np.load('onnx/assets/f0.npy')

print('mel', mel.shape)
print('f0', f0.shape)

session = ort.InferenceSession(
    'onnx/assets/nsf_hifigan.onnx',
    providers=['CPUExecutionProvider']
)

start = time.time()
wav = session.run(['waveform'], {'mel': mel, 'f0': f0})[0]
end = time.time()

print('waveform', wav.shape)
print('cost', end - start)


wavfile.write('onnx/assets/waveform.wav', 44100, wav[0])
