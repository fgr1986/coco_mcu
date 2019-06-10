# coco_mcu
Javier Fernández Marques, Fernando García Redondo proposal for Visual Wake Words Challenge@CVPR 2019

## Contributions
* Equally contributed: Javier Fernández Marques, Fernando García Redondo
* Support: Sid Das and Paul Whatmough

## Contact
fernando.garciaredondo@arm.com, Javier.Fernandez-Marques@arm.com

## Description
Proposal for [Visual Wake Words Challenge@CVPR-2019](https://docs.google.com/document/u/2/d/e/2PACX-1vStp3uPhxJB0YTwL4T__Q5xjclmrj6KRs55xtMJrCyi82GoyHDp2X0KdhoYcyjEzKe4v75WBqPObdkP/pub)

## Accuracy over validation dataset
The network achieves a best accuracy on the validation dataset of 90.2%.

## Network Architecture.
### Related work
The scheleton of the developed network is based on [Google's MobilenetV3](https://arxiv.org/abs/1905.02244), taking the basic bottleneck layer implementation from [Bisonai](https://github.com/Bisonai/mobilenetv3-tensorflow) as a basic CNN layer.

### Proposed network
The network architecture is described in `mb_att.py`, and depicted in the following picture:

![NN](https://github.com/fgr1986/arm_coco/blob/master/arm_coco.png)

The attention and bneck layers are defined as follows:

![NN](https://github.com/fgr1986/arm_coco/blob/master/bneck_mobilenet_v3.png)

Note that bneck layers follow the description in https://arxiv.org/abs/1905.02244

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
We take into account how existing NNs libraries for MCUs (e.g. [Arm's CMSIS](https://github.com/ARM-software/CMSIS_5))
* Point-wise additions and multiplications are considered in-place. Should a custom operation ´f(t)´ be performed over the tensor ´t´, a single uint8 can hold temporarily the value ´aux = f(t_i)´, later replacing the memory space where ´t_i´ was in.
* Let ´h(t)=g(f(t))´, being ´t´ not required for later operations, after ´f(t)´ has been computed, the memory space previously occupied by ´t´ can be freed and reasigned.
* Let ´h(t) = y(g(t), f(t))´ a graph section with two paths or branches, dependent on ´t´, we compute first one branch i.e. 'g(t)´, then ´f(t)´, so ´t´ no longer requires to be hold in memory, and its memory space reallocated to compute ´y()´.


Taking into account the previous considerations, and given the architecture defined above (described in detail in `mb_att.py`), the RAM memory peak takes place in the nodes:
* Input conv: 256x256x3 + 86x86x8 = 249.8KB
* Bneck0: 86x86x8 [shared input] + 86x86x16 = 173.34KB. After computation buffer of 86x86x16
* Att0: Max peak with 86x86x16 [bneck0 out] + 86x86x8 [shared input that can be discarted after first max_pool] + 43x43x8 = 218KB. After first pooling the peak is: 86x86x16 + 43x43x8 + 22x22x(16+16+32) = 164.1KB. After computation buffer of 22x22x16
* Pointwise Mult Att0*Bneck0 = [attention is upscaled] 2x86x86x16 = 231.12KB

* Att1: Max peak with 86x86x8 [shared input that can be discarted after first max_pool] + 43x43x8 = 218KB. After first pooling the peak is: 86x86x16 + 43x43x16 + 22x22x(16+16+32) = 178.9KB. After computation buffer of 22x22x16
* Bneck1: 86x86x16 [shared input] + 86x86x16 + 22x22x16 [att1 output] = 238.7KB. After computation buffer of 86x86x16
* Mult Att1*Bneck1 =  [attention is upscaled] 2x86x86x16 = 231.12KB

### Key points in NN design and training
#### General description
The proposed NN customizes the recently presented mobilenet_v3, introducing:
* Attention mechanishms in the first layers so, giving the variety of input images, the features extraction can focus on interesting regions.
* Deep residual connections to avoid gradient vanishing.
* Instead of the traditional CNN approach, and giving the varied images dataset (size, etc), we try to keep as much resolution as possible as we deepen into the network.

#### Data augmentation
We have applied data augmentation during training to avoid overfiting. `input_data.py` describes the mechanishms, wich include random cropping, hue and brightness variation, rotations, etc.

#### Regularization and other anti-overfiting mechanishms
We used L2 regularization together with dropout layers. Additionally, 2b weights quantization helps reducing overfiting effects

#### Learning rate scheduling
A custom learning rate scheduler has been used (see  `mb_att.py`) so both learning and quantization caused oscilations are controlled.

