---
base_model: meta-llama/Llama-3.2-1B
library_name: transformers.js
license: llama3.2
---

https://huggingface.co/meta-llama/Llama-3.2-1B with ONNX weights to be compatible with Transformers.js.

Note: Having a separate repo for ONNX weights is intended to be a temporary solution until WebML gains more traction. If you would like to make your models web-ready, we recommend converting to ONNX using [🤗 Optimum](https://huggingface.co/docs/optimum/index) and structuring your repo like this one (with ONNX weights located in a subfolder named `onnx`).