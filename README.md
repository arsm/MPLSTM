# Multi-Perspective-LSTM-for-Joint-Visual-Representation-Learning
## MPLSTM &mdash; Official TensorFlow-Keras Implementation

![Teaser image](MPLSTM.png)

**Multi-Perspective LSTM for Joint Visual Representation Learning**<br>
Alireza Sepas-Moghaddam, Fernando Pereira, Paulo Lobato Correia, Ali Etemad<br>

CVPR'21 Paper: http://arxiv.org/abs/<br>

Abstract: *We present a novel LSTM cell architecture capable of Multi-Perspective LSTM (MP-LSTM) cell architecture learning both intra- and inter-perspective relationships available in visual sequences captured from multiple perspectives. Our architecture adopts a novel recurrent joint learning strategy that uses additional gates and memories at the cell level. We demonstrate that by using the proposed cell to create a network, more effective and richer visual representations are learned for recognition tasks. We validate the performance of our proposed architecture in the context of two multi-perspective visual recognition tasks namely lip reading and face recognition. Three relevant datasets are considered and the results are compared against fusion strategies, other existing multi-input LSTM architectures, and alternative recognition solutions. The experiments show the superior performance of our solution over the considered benchmarks, both in terms of recognition accuracy and computational complexity.*

## Requirements

* Both Linux and Windows are supported. Linux is recommended for performance and compatibility reasons.
* 64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.14.3 or newer.
* We recommend TensorFlow 1.14, which we used for all experiments in the paper, but TensorFlow 1.15 is also supported on Linux. TensorFlow 2.x is not supported.
* On Windows you need to use TensorFlow 1.14, as the standard 1.15 installation does not include necessary C++ headers.
* One or more high-end NVIDIA GPUs, NVIDIA drivers, CUDA 10.0 toolkit and cuDNN 7.5. To reproduce the results reported in the paper, you need an NVIDIA GPU with at least 16 GB of DRAM.
* Docker users: use the [provided Dockerfile](./Dockerfile) to build an image with the required library dependencies.

## Citation

```
@inproceedings{Sepas2021MPLSTM,
  title     = {Multi-Perspective {LSTM} for Joint Visual Representation Learning},
  author    = {Alireza Sepas-Moghaddam and Fernando Pereira and Paulo Lobato Correia and Ali Etemad},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2021}
}
```
