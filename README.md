# coco_mcu

## Description
Submission to the [Visual Wake Words Challenge](https://docs.google.com/document/u/2/d/e/2PACX-1vStp3uPhxJB0YTwL4T__Q5xjclmrj6KRs55xtMJrCyi82GoyHDp2X0KdhoYcyjEzKe4v75WBqPObdkP/pub) @ CVPR'19. All the code developed by the contributors (see contributors section) is released under MIT license.

## Contributions
* Equally contributed: Javier Fernández-Marqués and Fernando García-Redondo
* Support: Shidhartha Das and Paul Whatmough
* Advise: Ramón Matas

## Contact
fernando.garciaredondo@arm.com and javier.fernandez-marques@arm.com

## Performance Metrics Summary
Following the instructions in [Visual Wake Words Challenge](https://docs.google.com/document/u/2/d/e/2PACX-1vStp3uPhxJB0YTwL4T__Q5xjclmrj6KRs55xtMJrCyi82GoyHDp2X0KdhoYcyjEzKe4v75WBqPObdkP/pub), we generated the TFrecords using the script  ‘build_visualwakewords_data.py’ from [slim/dataset](https://github.com/tensorflow/models/tree/master/research/slim/datasets) lib.

* Best Validation accuracy: 88.85%
* Validation on minival_dataset: 87.96%
* Model size: 244.4KB
* Peak memory usage: 249 KB
* MACs for inference: 56.8 Million OPs

In the following subsection we provide a description of our model architecture and further information on how the parameters above were obtained. Our model is still training and we will report the final validation once the 140 epochs are completed.

## Network Architecture.
### Related work
The skeleton of the developed network is based on [Google's MobilenetV3](https://arxiv.org/abs/1905.02244), taking the basic bottleneck layer implementation from [Bisonai](https://github.com/Bisonai/mobilenetv3-tensorflow) as a basic CNN layer.

### Proposed network
The network architecture is described in the file `mb_att.py`, and depicted in the following picture:

![NN](https://github.com/fgr1986/arm_coco/blob/fergar01/arm_coco_small.png)

The attention and bneck layers are defined as follows:

![NN](https://github.com/fgr1986/arm_coco/blob/master/bneck_mobilenet_v3.png)

### Key points in NN interpretation for MCU deployement

#### Total operations
TF Profiler reports total number of flops 1983338, which is called at the beginning of the training. See `get_flops()` method in `keras_q_model.py`.

#### Quantization
* The network is quantized using TF (v1.13) [quantization functions](https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/quantize/python/quantize_graph.py)
* Activations are quantized using 8 bits.
* Weights are quantized using 2 bits.
* The quantization delay is non-zero as it gets applied after 10K batches (approx. at epoch 10 for a batch size of 64). For more info see `mb_att.py`.

#### ROM Size
* The network is composed of a total of 1,000,914 parameters.
* Each parameter uses 2-bits in ROM, giving a total of 244.4KB.

#### RAM Usage
During the development of the proposed model, we had in mind how Neural Networks libraries for MCUs (see Arm's [CMSIS](https://github.com/ARM-software/CMSIS_5)) are able to optimize not only the computation but also the memory fingerprint, allowing, when possible, to reuse temporal buffers, or perform in-place computations. Following these principles,
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

* Bneck2: Peak with 86x86x16 + 43x43x32 = 180.9KB, output 43x43x24
* Bneck3: Peak with 43x43x24 + 22x2x64  + 22x22x32= 89KB, output 22x22x32
* Bneck4: Peak with 22x22x32 + 11x11x(256 + 128) = 60.5KB, output 11x11x128
* Bneck5: Peak with 11x11x128 + 6x6x128 [res] + 6x6x256 = 60.5KB, output 6x6x128
* Bneck6: Peak with 6x6x128 + 6x6x128 [res] + 6x6x256 = 18KB, output 6x6x128 
* LastStage Layer: Peak with 6x6x(128 + 512 + 1200) = 64.7KB

**Peak use: 249KB**

### Key points in NN design and training
#### General description
The proposed NN customizes the recently presented mobilenet_v3, introducing:
* 2-bit weights allow 4x number of weights in the final layers.
* Attention mechanisms  in the first layers so, giving the variety of input images, the features extraction can focus on interesting regions.
* Instead of the traditional CNN approach, and giving the varied images dataset (size, etc), we try to keep as much resolution as possible as we deepen into the network.

#### Data augmentation
We have applied data augmentation during training to avoid overfitting. `input_data.py` file describes the mechanisms, wich includes random cropping, hue and brightness variation, rotations, etc.

#### Regularization and other anti-overfitting mechanishms
We used L2 regularization together with dropout layers. Additionally, 2b weights quantization helps reducing overfitting effects

#### Learning rate scheduling
A custom learning rate scheduler has been used (see `mb_att.py`) so both learning and quantization caused oscillations are controlled.

