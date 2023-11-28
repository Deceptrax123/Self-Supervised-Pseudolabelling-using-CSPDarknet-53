# Self Supervised Pretraining using Autoencoders to detect multiple objects
![Output](outputs/Regularization/w_0.95_reg_2/output.png)

## Methodology
The Model was trained using pixel wise mean square error loss with additional penalty to pixels within a box

## Architecture
|Architecture|Description|
|-|----|
|Backbone|CSPDarknet-53|
|Decoder|Inverse CSPDarknet-53|
|Skip Connections|Between Residual Blocks from encoder to decoder|

## Usage
To improve detections on YOLO models by learning a set of weights before fine tuning.