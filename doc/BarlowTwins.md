# Barlow Twins 
```yaml
Date: May 20, 2024
Author: Liyao
```

<a href="https://arxiv.org/pdf/2103.03230"><img alt="Static Badge" src="https://img.shields.io/badge/%20Barlow%20-Twins-%23B31B1B?style=flat&logo=arxiv"></a> 



## Code

- https://github.com/facebookresearch/barlowtwins
- https://lightning.ai/docs/pytorch/2.1.0/notebooks/lightning_examples/barlow-twins.html



## Dataset

```python
trainset = PathMNIST(split="train", download=False)
mean = np.array([0.,0.,0.])
std = np.array([0.,0.,0.])
for i in range(len(trainset)):
    x = np.array(trainset[i][0])
    mean += np.mean(x, axis=(0,1))
mean /= len(trainset)

# mean
array([188.83897503, 135.91045926, 179.98635682])
array([0.73765225, 0.53090023, 0.70307171])

# std
array([31.53896524, 45.07444572, 31.72982184])
array([0.12319908, 0.17607205, 0.12394462])
```

