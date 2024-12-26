# Install and run training tasks with intel NPU

As of now, the intel community has not provide a convinient enough way to use NPU with ecosystem, such as PyTorch, TensorFlow, etc. So we have to tune a bit of it.

* install `intel-npu-acceleration-library` on `git+https://github.com/intel/intel-npu-acceleration-library.git`. Note NPU is not working with wsl yet, due to linux kernel version according to [this issue](https://github.com/intel/intel-npu-acceleration-library/issues/13)


## Benchmark

Training a minimum FFN with pytorch, sample time of each epoch for with and without NPU is as follows:
| device | epoch time | Accuracy | loss | NPU Utilization | CPU Utilization | GPU Utilization |
| --- | --- | --- | --- | --- | --- | --- |
| CPU | 15s | 71.2% | 0.78 | 0 | 40% | 0 |
| NPU | 30s | 22.7% | 2.274130 | 7% | 20% | 0 |

⚠️ NPU is not suitable for training yet.