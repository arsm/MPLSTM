# Multi-Perspective Long Short Term Memory

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
* You need to use [keras-vggface package](https://github.com/rcmalli/keras-vggface) to extract spatial embeddings. 
* One or more high-end NVIDIA GPUs, NVIDIA drivers, CUDA 10.0 toolkit and cuDNN 7.5. 




## Preparing datasets

The [OuluVS2](http://www.ee.oulu.fi/research/imag/OuluVS2/index.html), [Light Field Faces in the Wild (LFFW)](http://www.img.lx.it.pt/LFFW/), and [Light Field Faces in the Wild (LFFW)](http://www.img.lx.it.pt/LFFW/) datasets are used to evaluate the performance of MPLSTM. After you have downloaded the dataset successfully, you need to split the data into training, validation, and testing 



stored as multi-resolution TFRecords, similar to the . Each dataset consists of multiple `*.tfrecords` files stored under a common directory, e.g., `~/datasets/ffhq/ffhq-r*.tfrecords`. In the following sections, the datasets are referenced using a combination of `--dataset` and `--data-dir` arguments, e.g., `--dataset=ffhq --data-dir=~/datasets`.


| Additional material | &nbsp;
| :--- | :----------
| [StyleGAN2](https://drive.google.com/open?id=1QHc-yF5C3DChRwSdZKcx1w6K8JvSxQi7) | Main Google Drive folder
| &boxvr;&nbsp; [stylegan2-paper.pdf](https://drive.google.com/open?id=1fnF-QsiQeKaxF-HbvFiGtzHF_Bf3CzJu) | High-quality version of the paper
| &boxvr;&nbsp; [stylegan2-video.mp4](https://drive.google.com/open?id=1f_gbKW6FUUHKkUxciJ_lQx29mCq_fSBy) | High-quality version of the video
| &boxvr;&nbsp; [images](https://drive.google.com/open?id=1Sak157_DLX84ytqHHqZaH_59HoEWzfB7) | Example images produced using our method
| &boxv;&nbsp; &boxvr;&nbsp;  [curated-images](https://drive.google.com/open?id=1ydWb8xCHzDKMTW9kQ7sL-B1R0zATHVHp) | Hand-picked images showcasing our results
| &boxv;&nbsp; &boxur;&nbsp;  [100k-generated-images](https://drive.google.com/open?id=1BA2OZ1GshdfFZGYZPob5QWOGBuJCdu5q) | Random images with and without truncation
| &boxvr;&nbsp; [videos](https://drive.google.com/open?id=1yXDV96SFXoUiZKU7AyE6DyKgDpIk4wUZ) | Individual clips of the video as high-quality MP4
| &boxur;&nbsp; [networks](https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/) | Pre-trained networks
| &ensp;&ensp; &boxvr;&nbsp;  stylegan2-ffhq-config-f.pkl | StyleGAN2 for <span style="font-variant:small-caps">FFHQ</span> dataset at 1024&times;1024
| &ensp;&ensp; &boxvr;&nbsp;  stylegan2-car-config-f.pkl | StyleGAN2 for <span style="font-variant:small-caps">LSUN Car</span> dataset at 512&times;384
| &ensp;&ensp; &boxvr;&nbsp;  stylegan2-cat-config-f.pkl | StyleGAN2 for <span style="font-variant:small-caps">LSUN Cat</span> dataset at 256&times;256
| &ensp;&ensp; &boxvr;&nbsp;  stylegan2-church-config-f.pkl | StyleGAN2 for <span style="font-variant:small-caps">LSUN Church</span> dataset at 256&times;256
| &ensp;&ensp; &boxvr;&nbsp;  stylegan2-horse-config-f.pkl | StyleGAN2 for <span style="font-variant:small-caps">LSUN Horse</span> dataset at 256&times;256
| &ensp;&ensp; &boxur;&nbsp;&#x22ef;  | Other training configurations used in the paper



## Inquiries

For inquiries, please contact [alireza.sepasmoghaddam@queensu.ca](mailto:alireza.sepasmoghaddam@queensu.ca)<br>

## Citation

```
@inproceedings{Sepas2021MPLSTM,
  title     = {Multi-Perspective {LSTM} for Joint Visual Representation Learning},
  author    = {Alireza Sepas-Moghaddam and Fernando Pereira and Paulo Lobato Correia and Ali Etemad},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2021}
}
```
