# coco_mcu

## Description
Submission to the [Visual Wake Words Challenge](https://docs.google.com/document/u/2/d/e/2PACX-1vStp3uPhxJB0YTwL4T__Q5xjclmrj6KRs55xtMJrCyi82GoyHDp2X0KdhoYcyjEzKe4v75WBqPObdkP/pub) @ CVPR'19.

## Contributions
* Equally contributed: Javier Fernández-Marqués and Fernando García-Redondo
* Support: Shidhartha Das and Paul Whatmough

## Contact
fernando.garciaredondo@arm.com and javier.fernandez-marques@arm.com

## Accuracy over validation dataset
The network achieves a best accuracy on the validation dataset of 90.2%.

## Network Architecture.
### Related work
The scheleton of the developed network is based on [Google's MobilenetV3](https://arxiv.org/abs/1905.02244), taking the basic bottleneck layer implementation from [Bisonai](https://github.com/Bisonai/mobilenetv3-tensorflow) as a basic CNN layer.

### Proposed network
The network architecture is described in the file `mb_att.py`, and depicted in the following picture:

![NN](https://github.com/fgr1986/arm_coco/blob/master/arm_coco.png)

The attention and bneck layers are defined as follows:

![NN](https://github.com/fgr1986/arm_coco/blob/master/bneck_mobilenet_v3.png)

### Key points in NN interpretation for MCU deployement

#### Total operations
TF Profiler reports total number of flops 1983336

#### Quantization
* The network is quantized using TF (v1.13) [quantization functions](https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/quantize/python/quantize_graph.py)
* Activations are quantized using 8 bits
* Weights are quantized using 2 bits
* The quantization delay is non-zero (see `mb_att.py`).

#### ROM Size
* The network is composed of a total of 1,000,914 parameters
* Each parameter uses 2b in the ROM, giving a total of 244.4KB

#### RAM Usage
During the development of the proposed model, we had in mind how MCU NN libs (see CMSIS: https://github.com/ARM-software/CMSIS_5) are able to optimize not only the computation but also the memory fingerprint, allowing, when possible, to reuse temporal buffers, or perform in-place computations. Following these principles,
* Point-wise additions and multiplications are considered in-place. Should a custom operation `f(t)` be performed over the tensor `t`, a single uint8 can hold temporarily the value `aux = f(t_i)`, later replacing the memory space where `t_i` was in.
* Let `h(t)=g(f(t))`, being `t` not required for later operations, after `f(t)` has been computed, the memory space previously occupied by `t` can be freed and reasigned.
* Let `h(t) = y(g(t), f(t))` a graph section with two paths or branches, dependent on `t`, we compute first one branch i.e. `g(t)`, then `f(t)`, so `t` no longer requires to be hold in memory, and its memory space reallocated to compute `y()`.


Taking into account the previous considerations, and given the architecture defined above (described in detail in `mb_att.py`), the RAM memory peack takes place in the nodes:
* Input conv: 256x256x3 + 86x86x8 = 249.8KB
* Bneck0: 86x86x8 [shared input] + 86x86x16 = 173.34KB. After computation buffer of 86x86x16
* Att0: Max peak with 86x86x16 [bneck0 out] + 86x86x8 [shared input that can be discarted after first max_pool] + 43x43x8 = 218KB. After first pooling the peak is: 86x86x16 + 43x43x8 + 22x22x(16+16+32) = 164.1KB. After computation buffer of 22x22x16
* Pointwise Mult Att0*Bneck0 = [attention is upscaled] 2x86x86x16 = 231.12KB

* Att1: Max peak with 86x86x8 [shared input that can be discarted after first max_pool] + 43x43x8 = 218KB. After first pooling the peak is: 86x86x16 + 43x43x16 + 22x22x(16+16+32) = 178.9KB. After computation buffer of 22x22x16
* Bneck1: 86x86x16 [shared input] + 86x86x16 + 22x22x16 [att1 output] = 238.7KB. After computation buffer of 86x86x16
* Mult Att1*Bneck1 =  [attention is upscaled] 2x86x86x16 = 231.12KB

* Residual is stored: 22x22x16 = 7.5KB

* Bneck2: Peak with 86x86x16 + res + 43x43x32 = 180.9KB, output 43x43x24
* Bneck3: Peak with 43x43x24 + res + 43x43x64 = 166.4KB, output 43x43x32
* Bneck4: Peak with 43x43x32 + res + 22x22x256 = 186.34KB, output 22x22x128
* Bneck5: Peak with 22x22x128 + res + 22x22x128 [res] + 22x22x256 = 249.56KB, output 22x22x128
* Bneck6: Peak with 22x22x128 + res + 22x22x128 [res] + 22x22x256 = 249.56KB, output 22x22x128 
* Bneck7: Peak with 22x22x128 + res + 22x22x128 = 128.56KB, output 22x22x64
* Bneck8: Peak with 22x22x64 + res + 22x22x64 = 68KB, output 22x22x32, upscaling to 43x43x32
* Bneck9: Peak with 43x43x32 + res + 43x43x32 = 123.1KB, output 43x43x16
* Add: Peak with 43x43x16x2 = 57.8KB, output 43x43x16, avg pool to 10x10x16
* Last Layer: Peak with 10x10x(16 + 512 + 1200) = 167.2KB

**Peak use: 249KB**

### Key points in NN design and training
#### General description
The proposed NN customizes the recently presented mobilenet_v3, introducing:
* Attention mechanishms in the first layers so, giving the variety of input images, the features extraction can focus on interesting regions.
* Deep residual connections to avoid gradient vanishing.
* Instead of the traditional CNN approach, and giving the varied images dataset (size, etc), we try to keep as much resolution as possible as we deepen into the network.

#### Data augmentation
We have applied data augmentation during training to avoid overfiting. `input_data.py` file describe the mechanishms, wich include random cropping, hue and brightness variation, rotations, etc.

#### Regularization and other anti-overfiting mechanishms
We used L2 regularization together with dropout layers. Additionally, 2b weights quantization helps reducing overfiting effects

#### Learning rate scheduling
A custom learning rate scheduler has been used (see `mb_att.py`) so both learning and quantization caused oscilations are controlled.

