# arm_coco
Javier Fernández Marques &amp; Fernando García Redondo proposal for Visual Wake Words Challenge@CVPR 2019

## Contact
fernando.garciaredondo@arm.com, Javier.Fernandez-Marques@arm.com

## Description
Proposal for Visual Wake Words Challenge@CVPR 2019
[https://docs.google.com/document/u/2/d/e/2PACX-1vStp3uPhxJB0YTwL4T__Q5xjclmrj6KRs55xtMJrCyi82GoyHDp2X0KdhoYcyjEzKe4v75WBqPObdkP/pub]

## Accuracy over validation dataset
The network achieves a best accuracy on the validation dataset of 90.2%.

## Network Architecture.
### Related work
The scheleton of the developed network is based on Google's MobilenetV3 (https://arxiv.org/abs/1905.02244), taking the basic bottleneck layer implementation from Bisonai (https://github.com/Bisonai/mobilenetv3-tensorflow) as a basic CNN layer.

### Proposed network
The network architecture is described in the file ´mb_att.py´, and depicted in the following picture:

![NN](https://raw.githubusercontent.com/fgr1986/arm_coco/master/arm_coco.svg)

### Key points
* The network is quantized to use
### MCU deploying considerations
