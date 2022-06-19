# Binary Segmentation Network for Tumor Detection in MRIs

Link to Dataset: ![](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/data).
**NB**: To utilise this dataset, accept the rules of the kaggle competition first of all.

The Neural network architecture used in this implementation is a custom architecture I devised to enable residual connections between and two kinds of networks. The architecture consists of an encoder and a decoder network with skip/residual connections between both networks. In the encoder, the ResNet34 backbone was utilised for feature extraction in the encoder.
The encoder outputs five feature maps, each map from each previous layer of the network. The feature maps are then fed into the decoder network that consists of convolutional transpose layers. The model outputs three mask segments from the three various classes (large bowel, small bowel and stomach). 

The implementated network is not limited to MRI dataset, it can also be extended to other datasets for semantic image segmentation.

The model training and results on the tumor Segmentation implementation on Large bowel, small bowel and stomach of patient MRI scan of the gut area are in this jupyter notebook [here](https://github.com/ches-001/Binary-Segmentation-Network-for-Tumor-Detection-in-MRIs/blob/main/tumor_segmentation.ipynb).

### How to use:

clone the repository like so:
`git clone https://github.com/ches-001/Binary-Segmentation-Network-for-Tumor-Detection-in-MRIs`

Within the `script` folder, copy the `binary_segmentation` folder to working directory and import the neural network like so:

```
from binary_segmentation import SegmentNet
```

Similarly, the encoder and decoder networks can be imported in same fashion like so:
```
from binary_segmentation import SpatialEncoder, Decoder
```

Endevour to checkout the jupyter notebook on how to use these networks.

The loss function utilised in this implementation is a combination of three kinds of losses, namely: **Probability distribution loss**, **Region based loss** and **Boundary based loss**

The loss is given as: $$L = (1+Lbce) * [1 - (w_1*(1-Lhd) + w_2*(1 - Ldc))]$$
Where w_1 and w_2 are weights values attached to the boundary and region based losses respectively.
Here we add 1 to the probability distribution loss to prevent vanishing gradients when it finally gets minimised.

The probability distribution loss is the Binary Cross Entropy loss. This loss aims to reduce the probability distribution between ground truth and prediction.
The loss is given as: $$ Lbce = y * log(x) + (1 - y) * log(1 - x) $$

The region based loss used is the dice loss. This loss aims to improve the jaccard index (intersection over union) between ground truth and prediction.
The loss is given as: $$ Ldc = 1 - [2 * (Y n X) / (Y + X)] $$

The boundary based loss used is the hausdorff distance metric: minimising this metric ensures that the 2D distance between surfaces of ground tuth and prediction is tending towards 0 than to 1.
The loss is given as: $$ Lhd = max[ max_x(min_y(||x - y||)), max_y(min_x(||y - x||)) ] $$, where x ∈ X and y ∈ Y

The optimizer used for this implementation is the ADAM (Adaptive Moment Estimation) optimizer with initial learning rate of 1e-3. A step learning rate scheduler was utilised with step size of 15 and gamma value of 0.1. The model was trained from scratch with no pretrained weights for 50 epochs.


Similar to the other two imports, the loss function can be imported and used like so:

```
from binary_segmentation import SegmentationMetric

metricfunc = SegmentationMetric()
loss = metricfunc(pred, target)
```

You can also get the BCE loss, the Hausdorff distance and the Dice coefficient score of a given prediction/target pair like so:

```
bce_loss, hausdorff, dice_score = metricfunc.metric_values(pred, target)
```


### Results
![result image3](https://user-images.githubusercontent.com/70514310/173208446-8ab71d8f-c0b9-441a-a6dc-a1a874d03f3c.png)

![result image2](https://user-images.githubusercontent.com/70514310/173208460-2e14db9c-256f-43b2-862f-e497a52151a9.png)

![result image1](https://user-images.githubusercontent.com/70514310/173208469-dc221ea6-2e9a-4001-9105-ad4dd79c4a1a.png)

