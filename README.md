# Binary Segmentation Network for Tumor Detection in MRIs
<hr>

Link to Dataset: ![](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/data).
**NB**: To utilise this dataset, accept the rules of the kaggle competition first of all.

The Neural network architecture used in this implementation is a custom architecture I devised to enable residual connections between and two kinds of networks. The architecture consists of an encoder and a decoder network with skip/residual connections between both networks. In the encoder, the ResNet34 backbone was utilised for feature extraction in the encoder.
The encoder outputs five feature maps, each map from each previous layer of the network. The feature maps are then fed into the decoder network that consists of convolutional transpose layers. The model outputs three mask segments from the three various classes (large bowel, small bowel and stomach). 

The implementated network is not limited to MRI dataset, it can also be extended to other datasets for semantic image segmentation.

The model training and results on the tumor Segmentation implementation on Large bowel, small bowel and stomach of patient MRI scan of the gut area are in this jupyter notebook [here](https://github.com/ches-001/Binary-Segmentation-Network-for-Tumor-Detection-in-MRIs/blob/main/tumor_segmentation.ipynb).

<br><br>


## HOW TO USE:
<hr>

clone the repository like so:
`git clone https://github.com/ches-001/Binary-Segmentation-Network-for-Tumor-Detection-in-MRIs`

Within the `script` folder, copy the `binary_segmentation` folder to working directory and import the neural network like so:



### The Model
```python
from binary_segmentation import SegmentNet

if __name__ == '__main__':
    # model parameters
    input_channels = 1
    last_fmap_channel = 512
    output_channel = 1 
    n_classes = 3

    model = SegmentNet(input_channels, last_fmap_channel, output_channel, n_classes)
```

The encoder and decoder networks can be imported a similar fashion like so:
```python
from binary_segmentation import SpatialEncoder, Decoder

if __name__ == '__main__':
    # model parameters
    input_channels = 1
    last_fmap_channel = 512
    output_channel = 1 
    n_classes = 3

    encoder = SpatialEncoder(input_channels, dropout=0.2, pretrained=False)
    decoder = Decoder(last_fmap_channel, output_channel, n_classes, dropout=0.2)

    fmap_1, fmap_2, fmap_3, fmap_4, fmap_5 = encoder(torch.randn(10, 1, 224, 224))
```
The encoder uses ResNet34 as backbone network, so the pretrained should be set to `True` if `input_channels == 3`.

The input shape to the encoder should be any size, such that when divided by two atleast four times would yield an even number, eg: (224, 224), (256, 256), (512, 512) or any other multiples of 224 or 256. This is so that the 5 feature maps passed into the decoder tally with the `ConvTranspose2d` layers of the decoder. Failure to do so might result in an error.

You can overwrite the backbone network to another ResNet likes ResNet50 like so:

```python
from torchvision.models.resnet import Bottleneck

#resnet 50 backbone
spatial = SpatialEncoder(input_channels, block=Bottleneck, block_layers=[3, 4, 6, 3])
```
**NOTE** when changing to another ResNet backbone, the output channel of the last convolutional layer is the `last_fmap_channel` of the decoder network, Eg: for ResNet50, the last_fmap_channel is 2048.

Endevour to checkout the jupyter notebook on how to use these networks.


<br>

### Dataset Class
The dataset class is used to compile the dataset to pass to the dataloader, this class can be used like so:

```python
from binary_segmentation import ImageDataset

dataset = ImageDataset(self, images, images_df)
```
The `images` argument is a list or tuple of training images and the `images_df` is a pandas dataframe with metadata corresponding to each of the images in the list/tuple.
In this implementation, because of the size of the image data it was decided that loading them all at once to populate a list was best as it would be faster than simply loading each sample from ROM storage.

To use data augmentation techniques alongside the image dataset, you can do as follows:

```python
from binary_segmentation import data_augmentation

T = data_augmentation()

dataset = ImageDataset(self, images, images_df, transform=T, tp=0.5)
```
The `tp` argument corresponds to the probability of a given sample being transformed, `tp=1.0` implies that the transforms will be applied to the data samples all the time. Refer to ['the transforms code file']('https://github.com/ches-001/Binary-Segmentation-Network-for-Tumor-Detection-in-MRIs/blob/main/script/binary_segmentation/transforms.py') for more details on the keyword arguments of the `transforms.data_augmentation()`. You can also refer to the jupyter notebook or the code files for the inner workings of the classes as well as how they are used.


<br>

### Loss functions
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

```python
from binary_segmentation import SegmentationMetric

metricfunc = SegmentationMetric(w1=0.5, w2=o.5, debug=False)
loss = metricfunc(pred, target)
```
**NOTE:** You should enable debug mode when training with CUDA to catch unforseen errors and give you the corresponding location of said error. Debug mode is slightly slower than normal mode.

You can also get the BCE loss, the Hausdorff distance and the Dice coefficient score of a given prediction/target pair like so:

```python
bce_loss, hausdorff, dice_score = metricfunc.metric_values(pred, target)
```

<br>

### The Pipeline Class
The pipeline class can also be used in similar fashion as other classes, it takes the segmentation model, loss function, optimizer and device as positional arguments like so:

```python
from binary_segmentation import FitterPipeline

pipeline = FitterPipeline(model, lossfunc, optimizer, device)
```

When the model is passed to the pipeline, the weights of the `Conv2d` layers are initialised with *Xavier init*. This can be disabled by setting `weight_init = False`.

You can also add a custom weight initialiser like so:

```python
def init_weight_func(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if torch.is_tensor(m.bias):
            m.bias.data.fill_(0.01)

pipeline = FitterPipeline(
    model, lossfunc, optimizer, device, weight_init=True, 
    custom_weight_initializer=init_weight_func)
```

You can also use a learning rate scheduler for your optimizer after it has been passed to the pipeline class like so:
```python
import torch

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    pipeline.optimizer, step_size=28, gamma=0.1, verbose=True)
```

To train and test your model on the pipeline class, you can use the `train()` and `test()` methods like so:

```python
train_metric_values = pipeline.train(train_dataloader, epochs=10, verbose=False)
test_metric_values = pipeline.test(test_dataloader, verbose=True)

#save model
pipeline.save_model()
```

If you would like to test the model after every single epoch of training, then you can do as follows:

```python
#list for storing metric values at every iteration
training_losses, training_hds, training_dice_coeffs = [], [], []
testing_losses, testing_hds, testing_dice_coeffs = [], [], []

best_loss = np.inf
for i in range(EPOCHS):
    train_loss, training_hd, train_dice_coeff = pipeline.train(train_dataloader)
    training_losses.append(train_loss[0])
    training_hds.append(training_hd[0])
    training_dice_coeffs.append(train_dice_coeff[0])

    
    test_loss, testing_hd, test_dice_coeff = pipeline.test(test_dataloader)
    testing_losses.append(test_loss)
    testing_hds.append(testing_hd)
    testing_dice_coeffs.append(test_dice_coeff)
    
    lr_scheduler.step()
    if test_loss < best_loss:
        best_loss = test_loss
        pipeline.save_model()
```

<br>


## RESULTS

<hr>

![result image3](https://user-images.githubusercontent.com/70514310/173208446-8ab71d8f-c0b9-441a-a6dc-a1a874d03f3c.png)

![result image2](https://user-images.githubusercontent.com/70514310/173208460-2e14db9c-256f-43b2-862f-e497a52151a9.png)

![result image1](https://user-images.githubusercontent.com/70514310/173208469-dc221ea6-2e9a-4001-9105-ad4dd79c4a1a.png)

