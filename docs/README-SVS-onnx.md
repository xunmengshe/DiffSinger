# DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2105.02446)
[![GitHub Stars](https://img.shields.io/github/stars/MoonInTheRiver/DiffSinger?style=social)](https://github.com/MoonInTheRiver/DiffSinger)
[![downloads](https://img.shields.io/github/downloads/MoonInTheRiver/DiffSinger/total.svg)](https://github.com/MoonInTheRiver/DiffSinger/releases)
 | [Interactive🤗 SVS](https://huggingface.co/spaces/Silentlin/DiffSinger)

## DiffSinger (ONNX Deployment)

### 0. Environment Preparation

Run with the command to install extra requirements for exporting the model to ONNX format.

```bash
pip install onnx onnxsim  # Used for graph repairing and optimization
```

The `onnxruntime` package is required to run inference with ONNX model and ONNXRuntime. See the [official guide](https://onnxruntime.ai/) for instructions to install packages matching your hardware. CUDA, DirectML and default CPU are recommended since the model has been tested on these execution providers.

Note that the scripts are tested on PyTorch 1.8.

### 1. Export to ONNX format

Run with the command

```bash
python onnx/export/export_diff_decoder.py --exp EXP [--target TARGET]
```

where `EXP` is the name of experiment, `TARGET` is the path for the target onnx file.

This script will export the diffusion decoder to the ONNX format and do a lot of optimization (50% faster than PyTorch with ONNXRuntime).

Note:

- FastSpeech2 modules are not currently included, so the output model takes a tensor of shape [1, 256, n_frames] as input `condition`.
- DPM-Solver acceleration is not currently included, but PNDM is wrapped into the model. Use any `speedup` larger than 1 to enable it.

### 2. Inference with ONNXRuntime

See `onnx/infer/infer_diff_decoder` for details.

#### Issues related to CUDAExecutionProvider

In some cases, especially when you are using virtual environment, you may get the following error when creating a session with CUDAExecutionProvider, even if you already installed CUDA and cuDNN on your system:

```text
2022-11-28 13:30:53.1135333 [E:onnxruntime:Default, provider_bridge_ort.cc:1266 onnxruntime::TryGetProviderInfo_CUDA] D:\a\_work\1\s\onnxruntime\core\session\provider_bridge_ort.cc:1069 onnxruntime::ProviderLibrary::Get [ONNXRuntimeError] : 1 : FAIL : LoadLibrary failed with error 126 "" when trying to load "your_project_root\venv\lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll"

Traceback (most recent call last):
  File ".\onnx\infer\infer_diff_decoder.py", line 18, in <module>
    session = ort.InferenceSession(
  File "your_project_root\venv\lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 347, in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
  File "your_project_root\venv\lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py", line 395, in _create_inference_session
    sess.initialize_session(providers, provider_options, disabled_optimizers)
RuntimeError: D:\a\_work\1\s\onnxruntime\python\onnxruntime_pybind_state.cc:574 onnxruntime::python::CreateExecutionProviderInstance CUDA_PATH is set but CUDA wasn't able to be loaded. Please install the correct version of CUDA and cuDNN as mentioned in the GPU requirements page (https://onnxruntime.ai/docs/ref
erence/execution-providers/CUDA-ExecutionProvider.html#requirements), make sure they're in the PATH, and that your GPU is supported.
```

There are two ways to solve this problem.

1. Simply import PyTorch and leave it unused before you create the session:

```python
import torch
```

This seems stupid but if your PyTorch is built with CUDA, then CUDAExecutionProvider will just work.

2. When importing PyTorch, its `__init__.py` actually adds CUDA and cuDNN to the system DLL path. This can be done manually, with the following line before creating the session:

```python
import os
os.add_dll_directory(r'path/to/your/cuda/dlls')
os.add_dll_directory(r'path/to/your/cudnn/dlls')
```

See [official requirements](http://www.onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements) for all DLLs that should be included in the paths above.

In this way you can also switch between your system CUDA and PyTorch CUDA in your virtual environment.
