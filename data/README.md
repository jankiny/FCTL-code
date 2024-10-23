# Data

We put all data into the `data` directory, such as:

```text
data/
├── cifar-10
├── cifar-100
├── CUB
├── fgvc-aircraft-2013b
├── Imagenet_crop
├── Imagenet_resize
├── LSUN_crop
├── LSUN_resize
├── MNIST
├── svhn
├── tiny-imagenet-200
```

We provide the dataset resources in below table:

| Name            | Resource                                                                                                                                          | Note                     |
|-----------------|---------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------|
| MNIST           | [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)                                                                 | from torchvision.dataset |
| SVHN            | [train_32x32.mat](http://ufldl.stanford.edu/housenumbers/train_32x32.mat) [test_32x32.mat](http://ufldl.stanford.edu/housenumbers/test_32x32.mat) | from torchvision.dataset |
| CIFAR10         | [cifar-10-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)                                                                 | from torchvision.dataset |
| CIFAR100        | [cifar-100-python.tar.gz](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)                                                               | from torchvision.dataset |
| TinyImageNet    | [Tiny Imagenet](https://github.com/rmccorm4/Tiny-Imagenet-200)                                                                                    |                          |
| Imagenet_crop   | [Imagenet_crop.tar.gz](https://github.com/sjy-1995/MoEP-AE-OSR-code/blob/master/da1tasets/OOD_datasets/Imagenet_crop.tar.gz)                      | from paper "MoEP-AE"     |
| Imagenet_resize | [Imagenet_resize.tar.gz](https://github.com/sjy-1995/MoEP-AE-OSR-code/blob/master/datasets/OOD_datasets/Imagenet_resize.tar.gz)                   | from paper "MoEP-AE"     |
| LSUN_crop       | [LSUN_crop.tar.gz](https://github.com/sjy-1995/MoEP-AE-OSR-code/blob/master/datasets/OOD_datasets/LSUN_crop.tar.gz)                               | from paper "MoEP-AE"     |
| LSUN_resize     | [LSUN_resize.tar.gz](https://github.com/sjy-1995/MoEP-AE-OSR-code/blob/master/datasets/OOD_datasets/LSUN_resize.tar.gz)                           | from paper "MoEP-AE"     |
| CUB             | [CUB](http://www.vision.caltech.edu/datasets/cub_200_2011/)                                                                                       | from Caltech Vision Lab  |
| Aircraft        | [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)                                                                             |                          |

Acknowledgement: We sincerely thank [MoEP-AE](https://github.com/sjy-1995/MoEP-AE-OSR-code), [MLS](https://www.robots.ox.ac.uk/~vgg/research/osr/) and related institutions for providing and collecting dataset resources.

