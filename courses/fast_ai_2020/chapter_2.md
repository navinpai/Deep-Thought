### Random Thoughts

-  Null hypothesis = There is no difference between 2 things. In case of variables, it would suggest that there is no relation between 2 variables _(independent and dependent)_.
- The `p-value` is the evidence against a null hypothesis. The smaller the p-value, the stronger the evidence that you should reject the null hypothesis. Usually the decision cutoff is `p ≤ 0.05`.


### Papers

- *Designing Great Data Products* by *Jeremy Howard, Margit Zwemer and Mike Loukides* ([pdf](https://www.oreilly.com/radar/drivetrain-approach-data-products/))


### Code Samples

```
from fastai.vision.all import *
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
```


### References

- Model Zoo ([link](https://modelzoo.co/))