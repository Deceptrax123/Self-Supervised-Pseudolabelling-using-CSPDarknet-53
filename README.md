# Self Supervised Pretraining using Autoencoders to detect multiple objects

![Output_1](Experiments/Regularization/w_0.95_lambda_5/Blurred_X/output.png)
![Output_2](Experiments/Regularization/w_0.95_lambda_5/no_blur/output_2.png)

## Methodology
- The Model was trained using pixel wise mean square error loss with additional penalty added to pixels that belong to a bounding box.
- Dataset prepared by blurring wheat heads(X) to reconstruct the image with wheat heads(Y)
- Global Wheat Dataset,2021 was used to pretrain the model

## Experiments Performed
### Dataset type and Loss Penalty
- [x] Blurring Wheat Heads with No Loss Penalty on Bounding Box Pixels
- [x] Blurring Wheat Heads with Loss Penalty on Bounding Box Pixels
- [x] Without Blurring Wheat Heads with Loss penalty on Bounding Box Pixels

You can view the set of results displayed above in the Experiments Directory

### Initializing Technique
- [x] He Normal
- [ ] COCO weights of CSPDarknet-53
  
  

## Architecture
|Architecture|Description|
|-|----|
|Backbone|CSPDarknet-53|
|Decoder|Inverse CSPDarknet-53|
|Skip Connections|Between Residual Blocks from encoder to decoder|

## Usage
- To improve detections on YOLO models by learning a set of weights before fine tuning.
- SSL method to detect multiple objects in the same image
- Saves on computation cost while training object detectors from scratch