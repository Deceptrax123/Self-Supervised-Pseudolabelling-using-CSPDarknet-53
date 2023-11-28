# Self Supervised Pretraining using Autoencoders to detect multiple objects
![Output_1](outputs/Regularization/w_0.95_reg_2/output.png)
![Output_2](outputs/Regularization/w_0.95_reg_2/output_2.png)

## Methodology
The Model was trained using pixel wise mean square error loss with additional penalty to pixels within a box

## Architecture
|Architecture|Description|
|-|----|
|Backbone|CSPDarknet-53|
|Decoder|Inverse CSPDarknet-53|
|Skip Connections|Between Residual Blocks from encoder to decoder|

## Usage
. To improve detections on YOLO models by learning a set of weights before fine tuning.
. Saves on computation cost while training object detectors from scratch