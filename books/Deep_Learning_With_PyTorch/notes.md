## Notes from "Deep Learning with PyTorch"

##### Author: Vishnu Subramanian

##### [Amazon](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-pytorch)

#### Types of Tensors
-  `torch.Tensor` is alias for default tensor type i.e `torch.FloatTensor`
-  `x = torch.Tensor([3.1]) # 0-D tensors aka scalars` - 1-D tensors with 1 element
-  `x = torch.FloatTensor([1,2,3,4,5])`
-  `b = np.array([[1,2,3]. [4,5,6], [7,8,9])` then `a = torch.from_numpy(b)` gives a 2-D tensor
-  `a = np.array(Image.open('img.jpg').resize(224,224))` then `a = torch.from_numpy(a)` gives 3-D tensor of dim `224*224*3`

- Tensors can be moved to GPU using .cuda() eg. `a = a.cuda()`
- Variable = Tensor (data) + gradient + reference to function that created it
- Gradient = rate of change of loss function wrt `(W,b)``])`

#### Grad Calculation
- `x = torch.tensor(torch.ones(2,2), requires_grad=True) `
- `y = x.mean()` # Makes y a 0-D Tensor with value 1 and grad_fn MeanBackward1
- `y.backward()` makes `x.grad` = [[1,1],[1,1]] # Before call to backward x.grad is None
- `x.grad.zero_()` # Zeroes the gradients

`d/dx` the variable on which backward is called, and that gives value of gradient of x.
- `z = x**2` # Makes z a 0-D Tensor with value 1 and grad_fn MeanBackward1
- `z.backward(gradient=torch.ones(z.size()))` makes `x.grad` = [[2,2],[2,2]]

- Also remember that [gradients are accumulated](https://medium.com/@zhang_yang/how-pytorch-tensors-backward-accumulates-gradient-8d1bf675579b). So if `y = x**2` and `z = x**3` and we call backward on both `z` and `y`, `x.grad` will be 5 (because 2+3). Hence zero-ing out gradients is necessary

#### Learnable Params

These are tensors/params for which we have `requires_grad=True` . Usually, for inputs, we keep `requires_grad=False` because we don't want to change the input. In `Y = W*x + b` , `y` and `x` have `False` while `W` and `b` have `requires_gradient = True`

- `r = torch.nn.ReLU()` followed by `r(torch.tensor([1.,2.,3.,0.,-1.]))` gives `tensor([1., 2., 3., 0., 0.])`
- Similarly can also do with `LeakyReLU()`

#### Creating a network

- Create class extending `nn.Module` and implement `__init__()` and `forward()` methods.
- Layers are defined in `__init__()` and their interactions in `forward()`
- Once network is ready, we simply need to calculate Loss (which *has* to be a scalar) using some Loss function and a Optimiser
- Loss in PyTorch methods is `loss(prediction, actual)`
- `opt = optim.SGD(model.parameters(), lr=0.01)`. Here we give learning rate and learnable params to the optimizer.

```
for inp, tgt in dataset:
    opt.zero_grad() # Zeroes out gradients of learnable params to prevent accumulation (default behaviour)
    op = model(inp)
    loss = loss_fm(op, tgt)
    loss.backward()
    opt.step() # This performs actual update of learnable params
```

#### Loading Data

Assuming we have 2 directories `valid` and `train` and each dir has `N` dirs each consisting one class. You can load using

```
from torchvision import transforms
from torchvision.datasets import ImageFolder

trfs = transforms.Compose([transforms.Resize((225,225)), 
                           transforms.ToTensor(), 
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Other transforms include transforms.RandomHorizontalFlip() and transforms.RandomRotation(0.2)

train = ImageFolder('path/to/train/', trfs)
train.classes # Prints out classes
train.class_to_idx # Prints out classes with index

# train[0] has the whole image (3 matrix for RGB) along with class_index `([[[][]],[[],[]]], 0)`

train_dataloader = torch.utils.data.DataLoader(train, batch_size=64, num_workers=3)

```
Here, num_workers is for parallelization and is usually set to less than number of cores of machine.

Helper method to display a normalised tensor as image


```
def imshow(inp):
   inp = inp.numpy().transpose([1, 2, 0])
   mean = np.array([0.485, 0.456, 0.406])
   std = np.array([0.229, 0.224, 0.225])
   inp = std * inp + mean
   inp = inp.clip(0,1)
   plt.imshow(inp)

```

#### Overfitting 

Decreased by:

- Getting more data
- Reducing Network size/layers
- Simpler models using L1/L2 regularization of weights. (L1 = Sum of abs value of weight coeff added to cost while L2 = Sum of squares of weight coeff added)
- L2 can be applied to Optimizer as `opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)` where weight decay is L2
- Dropout 

```
d = torch.nn.Dropout()
inp = torch.tensor([1.,2.,3.,4.,5.,6.])

```

#### Picking Learning Rate

- StepLR: Reduce LR by gamma times every N steps `sc = StepLR(optimizer, step_size=20, gamma=0.1)`
- MultiStepLR: Steps are different sizes `sc = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1`
- ExponentialLR: Exponentially decrease LR

```
for epoch in epochs:
    scheduler.step() # to increment step
```

- ReduceLROnPlateau: `sc = ReduceLRRateOnPlateau(optimizer, 'min')`

```
for epoch in epochs:
    .
    .
    val_loss = calc_loss()
    sc.step(val_loss)
```

#### CNN 
- The mean and std deviation for a dataset can be [calculated like this if needed](http://forums.fast.ai/t/image-normalization-in-pytorch/7534/7)
- In Neural Style Transfer, the input layer has `requires_gradient = True` since we are editing the input image itself (and miniziming 2 losses: content loss and style loss)


#### Image Kernels

From [Wikipedia](https://en.wikipedia.org/wiki/Kernel_(image_processing))
- Identity

```
0 0 0 
0 1 0
0 0 0
```

- Edge Detection

```
 1  0 -1     -1 -1 -1
 0  0  0  or -1  8 -1
-1  0  1     -1 -1 -1
```

- Sharpen

```
 0 -1  0
-1  5 -1
 0 -1  0
```

- Gaussian Blur

```
1/16 * 1 2 1
       2 4 2
       1 2 1
```

- Box Blur

```
1/9 * 1 1 1
      1 1 1
      1 1 1
```

#### Layers of NN

Conv Layers

```
self.conv1 = nn.Conv2d(3,10, kernel_size=5)

# Here 3 is number of channels (RGB, this would be 1 for greyscale) , 10 is number of outchannels (and decides how many kernels are applied)
# And size is simply the kernel size
```

Convolutional layers are usually followed by pooling layers (With dropout applied to conv layer output) to reduce size of data and to allow better generalization

- MaxPool simply picks the max value from the set

We usually see Non-linear layers after pooling layers

- This simply operates on every element of the layer matrix

View is used to reshape the layer

```
x =torch.tensor([[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]],[[2.,3.,4.],[5.,.6,7.],[3.,2.,1.]]])
x.view(-1, 9) # Flattens the layer. -1 says don't flatten the first dimension
tensor([[1.0000, 2.0000, 3.0000, 4.0000, 5.0000, 6.0000, 7.0000, 8.0000, 9.0000],
        [2.0000, 3.0000, 4.0000, 5.0000, 0.6000, 7.0000, 3.0000, 2.0000, 1.0000]])

x.view(-1)
tensor([1.0000, 2.0000, 3.0000, 4.0000, 5.0000, 6.0000, 7.0000, 8.0000, 9.0000, 2.0000, 3.0000, 4.0000, 5.0000, 0.6000, 7.0000, 3.0000, 2.0000, 1.0000])
```

#### Training/Validation

- We call `model.train()` before training step and `model.eval()` before validation. The latter sets `self.training = True` for each module in the model, allowing things like pass-through Dropout (i.e Dropout has no effect) and BatchNorm. By default all modules are initialized in `train` mode.
- The entire model can be moved to GPU using `model.cuda()`

#### Transfer Learning

- Pretrained models have features and classifiers
- For VGG we can freeze the convolutional layers(called `features` in VGG) and simply retrain the fully connected dense layers (called `classifier` in VGG) with out data.

```
from torchvision import models
vgg = models.vgg16(pretrained=True)

for param in vgg.features.parameters(): param.requires_grad = False # Locks the layer weights of the model

vgg.classifier[6].out_features = 2 # Assuming we want to train for binary classification, We update the final linear layer

optimizer = optim.SGD(vgg.classifier.parameters(), lr= 0.001, momentum = 0.5)

# With just 20 epochs this gave good results on Dogs vs Cats (>98%)

for layer in vgg.classifier.children():
    if(type(layer) == nn.Dropout):
        layer.p = 0.2 # Change from 0.5 to 0.2 
```
