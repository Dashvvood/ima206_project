import argparse
import numpy as np
import medmnist
from medmnist.dataset import MedMNIST
from collections import defaultdict
import json

def get_subset_indices(dataset: MedMNIST, proportion=0.1) -> dict[int, list]:
    n_class = len(dataset.info["label"])
    
    L = defaultdict(list)  # class idx 2 sample idx of this class
    subL = defaultdict(list) # subset of L aka output
    
    # count indices for each class
    for i, data in enumerate(dataset): 
        class_idx = data[1][0]
        L[class_idx].append(i)
    
    # choose N * proportion indices of each class
    for class_idx in range(n_class):
        subL[class_idx] =np.random.choice(
            L[class_idx], 
            size=int(np.ceil(len(L[class_idx]) * proportion)),
            replace=False
        ).astype(np.int32).tolist() # int64 is not JSON serializable
    return dict(subL)


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description="""
        Extract the specified proportion of data from each category and return the data indices.    
    """)

    parser.add_argument("--root", type=str, default="./data/medmnist")
    parser.add_argument("--name", type=str, default="PathMNIST")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-p", "--proportion", type=float, default=0.01)
    parser.add_argument("--download", action="store_true", default=False)


    args, unknown_args = parser.parse_known_args()
    print(f"{args = }")
    print(f"{unknown_args = }")
    
    
    CertainMedMNIST = getattr(medmnist, args.name)
    train_dataset = CertainMedMNIST(
        split="train", 
        transform=None, 
        root=args.root,
        download=args.download
    )

    np.random.seed(args.seed)
    
    subset_indices = get_subset_indices(train_dataset, args.proportion)

    with open(f"./{args.name}_{args.proportion}_subset_seed_{args.seed}.json", 'w') as f:
        json.dump(subset_indices, f, indent=4)
