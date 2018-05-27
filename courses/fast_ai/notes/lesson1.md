# Lesson 1

### Random Notes
- Epoch = how many times we look at a sample image during training i.e one epoch means one pass of the full training set.
- One epoch = One forward pass _and_ one backward pass of all the training examples
- Each epoch is broken down into iterations, each of which is a  mini-batch of size X 
- Batch size is size to look at before making any changes. Can also be seen as number of examples looked at in parallel.
- Iterations = Number of batches processed. Example: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.


```
# Show an image
import matplotlib.pyplot as plt
img = plt.imread(f'{PATH}valid/cats/{files[0]}')
plt.imshow(img);
```