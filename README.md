### Original, Wasserstein, and Wasserstein-Gradient-Penalty DCGAN

(\*) This repo is a modification of [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow).

(\*) The full credit of the model structure design goes to [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow). 

I started with [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) because its DCGAN implementation is not fixed for one dataset, which is not a common setting. Most WGAN and WGAN-GP implementations only work on 'mnist' or one given dataset.



**A couple of modifications** I've made that could be helpful to people who try to implement GAN on their own for the first time.
1. Added `model_type` which could be one of 'GAN' (original), 'WGAN' (Wasserstein distance as loss), and 'WGAN_GP' (Wasserstein distance as loss function with gradient penalty), each corresponding to one variation of GAN model.
2. `UnifiedDCGAN` can build and train the graph differently according to `model_type`.
3. Some model methods were reconstructed so that the code is easier to read through.
4. Many comments were added for important, or potential confusing functions, like conv and deconv operations in `ops.py`.



If you are interested in the math behind the loss functions of GAN and WGAN, read **[here](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)**.


The `download.py` file stays same as in [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow). I keep this file in the repo for the sake of easily fetching dataset for testing.


**Related Papers**:

- Goodfellow, Ian, et al. ["Generative adversarial nets."](https://arxiv.org/pdf/1406.2661.pdf) NIPS, 2014.
- Martin Arjovsky, Soumith Chintala, and Léon Bottou. ["Wasserstein GAN."](https://arxiv.org/pdf/1701.07875.pdf) arXiv preprint arXiv:1701.07875 (2017).
- Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville. [Improved training of wasserstein gans.](https://arxiv.org/pdf/1704.00028.pdf) arXiv preprint arXiv:1704.00028 (2017).
