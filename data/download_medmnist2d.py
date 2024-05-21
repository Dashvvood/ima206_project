import os
import medmnist
print(medmnist.__version__)
from medmnist import (
    PathMNIST, 
    ChestMNIST, 
    DermaMNIST, 
    OCTMNIST, 
    PneumoniaMNIST, 
    RetinaMNIST,
    BreastMNIST, 
    BloodMNIST, 
    TissueMNIST, 
    OrganAMNIST, 
    OrganCMNIST, 
    OrganSMNIST
)

MNIST2D = [
    PathMNIST, 
    ChestMNIST, 
    DermaMNIST, 
    OCTMNIST, 
    PneumoniaMNIST, 
    RetinaMNIST,
    BreastMNIST, 
    BloodMNIST, 
    TissueMNIST, 
    OrganAMNIST, 
    OrganCMNIST, 
    OrganSMNIST
]

os.makedirs("./medmnist2d/", exist_ok=True)
for data_fn in MNIST2D:
    data_fn(split="test", download=True, root="./medmnist2d/")
    