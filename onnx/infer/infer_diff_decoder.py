import os
import time

import numpy as np
import onnxruntime as ort


# os.add_dll_directory(r'D:\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin')
# os.add_dll_directory(r'D:\NVIDIA GPU Computing Toolkit\cuDNN\bin')

condition = np.load('onnx/assets/condition.npy')
speedup = np.array(1, dtype=np.int64)

print('condition', condition.shape)
print('speedup', speedup.shape)

options = ort.SessionOptions()
session = ort.InferenceSession(
    'onnx/assets/1110_opencpop_ds1000_m128_n512x20.onnx',
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    sess_options=options
)

start = time.time()
mel = session.run(['mel'], {'condition': condition, 'speedup': speedup})[0]
end = time.time()

print('mel', mel.shape)
print('cost', end - start)

np.save('onnx/assets/mel_test.npy', mel)
