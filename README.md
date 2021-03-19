# Multi-Perspective-LSTM-for-Joint-Visual-Representation-Learning
## MPLSTM &mdash; Official TensorFlow-Keras Implementation

![Teaser image](MPLSTM.png)

**Multi-Perspective LSTM for Joint Visual Representation Learning**<br>
Alireza Sepas-Moghaddam, Fernando Pereira, Paulo Lobato Correia, Ali Etemad<br>

CVPR'21 Paper: http://arxiv.org/abs/<br>

Abstract: *We present a novel LSTM cell architecture capable of Multi-Perspective LSTM (MP-LSTM) cell architecture learning both intra- and inter-perspective relationships available in visual sequences captured from multiple perspectives. Our architecture adopts a novel recurrent joint learning strategy that uses additional gates and memories at the cell level. We demonstrate that by using the proposed cell to create a network, more effective and richer visual representations are learned for recognition tasks. We validate the performance of our proposed architecture in the context of two multi-perspective visual recognition tasks namely lip reading and face recognition. Three relevant datasets are considered and the results are compared against fusion strategies, other existing multi-input LSTM architectures, and alternative recognition solutions. The experiments show the superior performance of our solution over the considered benchmarks, both in terms of recognition accuracy and computational complexity.*

## Requirements

* Both Linux and Windows are supported. Linux is recommended for performance and compatibility reasons.
* 64-bit Python 3.6 installation. We recommend Anaconda3 with numpy 1.19.5 or newer.
* We recommend TensorFlow 1.14, which we used for all experiments in the paper, but newer versions of TensorFlow 1.15 might work.
* You need to use TensorFlow 2.1.5.
* One or more high-end NVIDIA GPUs, NVIDIA drivers, CUDA 10.0 toolkit and cuDNN 7.5. 

For inquiries, please contact [alireza.sepasmoghaddam@queensu.ca](mailto:alireza.sepasmoghaddam@queensu.ca)<br>


## Preparing datasets

Datasets are stored as multi-resolution TFRecords, similar to the [original StyleGAN](https://github.com/NVlabs/stylegan). Each dataset consists of multiple `*.tfrecords` files stored under a common directory, e.g., `~/datasets/ffhq/ffhq-r*.tfrecords`. In the following sections, the datasets are referenced using a combination of `--dataset` and `--data-dir` arguments, e.g., `--dataset=ffhq --data-dir=~/datasets`.


## Citation

```
@inproceedings{Sepas2021MPLSTM,
  title     = {Multi-Perspective {LSTM} for Joint Visual Representation Learning},
  author    = {Alireza Sepas-Moghaddam and Fernando Pereira and Paulo Lobato Correia and Ali Etemad},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2021}
}
```
