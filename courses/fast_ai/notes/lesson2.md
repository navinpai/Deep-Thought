# Lesson 2

### Random Notes

- There are different Optimisers which calculate gradient for the loss function and apply gradients to variables

momentum helps SGD to navigate along the relevant directions and softens the oscillations in the irrelevant. It simply adds a fraction of the direction of the previous step to a current step. This achieves amplification of speed in the correct direction and softens oscillation in wrong directions. This fraction is usually in the (0, 1) range. It also makes sense to use adaptive momentum. In the beginning of learning a big momentum will only hinder your progress, so it makes sense to use something like 0.01 and once all the high gradients disappeared you can use a bigger momentum. There is one problem with momentum: when we are very close to the goal, our momentum in most of the cases is very high and it does not know that it should slow down. This can cause it to miss or oscillate around the minima

Nesterov Accelerated Gradient overcomes this problem by starting to slow down early. In momentum we first compute gradient and then make a jump in that direction amplified by whatever momentum we had previously. NAG does the same thing but in another order: at first we make a big jump based on our stored information, and then ge calculate the gradient and make a small correction. This seemingly irrelevant change gives significant practical speedups.

AdaGrad or adaptive gradient allows the learning rate to adapt based on parameters. It performs larger updates for infrequent parameters and smaller updates for frequent one. Because of this it is well suited for sparse data (NLP or image recognition). Another advantage is that it basically eliminates the need to tune the learning rate. Each parameter has its own learning rate and due to the peculiarities of the algorithm the learning rate is monotonically decreasing. This causes the
biggest problem: at some point of time the learning rate is so small that the system stops learning.

AdaDelta resolves the problem of monotonically decreasing learning rate in AdaGrad. In AdaGrad the learning rate was calculated approximately as one divided by the sum of square roots. At each stage you add another square root to the sum, which causes denominator to constantly increase. In AdaDelta instead of summing all past square roots it uses sliding window which allows the sum to decrease. RMSprop is very similar to AdaDelta

Adam or adaptive momentum is an algorithm similar to AdaDelta. But in addition to storing learning rates for each of the parameters it also stores momentum changes for each of them separately.


### References

[Gradient Descent vs Adagrad vs Momentum in TensorFlow](https://stackoverflow.com/questions/36162180/gradient-descent-vs-adagrad-vs-momentum-in-tensorflow)