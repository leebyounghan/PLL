from .news20 import news20_dataset
from .trec import trec_Dataset
from .IMDB import IMDB_dataset
from .reuters import reuters_dataset
import torch
import pdb


def OOD_dataset(root_dir):

    in_train, in_test = news20_dataset(root_dir)
    ood_test = reuters_dataset(root_dir)[0]
    dataset = {
        "train_x": in_train['sentence'].tolist(),
        "train_y": torch.tensor(in_train['label'].tolist()),
        "test_in": in_test["sentence"].tolist(),
        "test_out": ood_test["sentence"].tolist(),
        "test_in_emb": in_test["sentence"].tolist(),
        "test_out_emb": ood_test["sentence"].tolist(),
        "train_text": in_train['sentence'].tolist(),
    }
    print(len(dataset["test_in"])+len(dataset["test_out"]))
    pdb.set_trace()
    return dataset
