### Random Thoughts

-  Null hypothesis = There is no difference between 2 things. In case of variables, it would suggest that there is no relation between 2 variables _(independent and dependent)_.
- The `p-value` is the evidence against a null hypothesis. The smaller the p-value, the stronger the evidence that you should reject the null hypothesis. Usually the decision cutoff is `p ≤ 0.05`.


### Papers

- *Designing Great Data Products* by *Jeremy Howard, Margit Zwemer and Mike Loukides* ([pdf](https://www.oreilly.com/radar/drivetrain-approach-data-products/))


### Code Samples

#### Vision

```
from fastai.vision.all import *
path = untar_data(URLs.PETS)/'images'

def is_cat(x): return x[0].isupper()
dls = ImageDataLoaders.from_name_func(
    path, get_image_files(path), valid_pct=0.2, seed=42,
    label_func=is_cat, item_tfms=Resize(224))

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)


img = PILImage.create(uploader.data[0])
is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
```

#### Segmentation

```
path = untar_data(URLs.CAMVID_TINY)
dls = SegmentationDataLoaders.from_label_func(
    path, bs=8, fnames = get_image_files(path/"images"),
    label_func = lambda o: path/'labels'/f'{o.stem}_P{o.suffix}',
    codes = np.loadtxt(path/'codes.txt', dtype=str)
)

learn = unet_learner(dls, resnet34)
learn.fine_tune(8)


learn.show_results(max_n=6, figsize=(7,8))
```

#### Text Sentiment

```
from fastai.text.all import *

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test')
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 1e-2)

learn.predict("I really liked that movie!")
```

### References

- Model Zoo ([link](https://modelzoo.co/))