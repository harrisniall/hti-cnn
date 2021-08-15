import os
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from pyll.base import TorchModel, AverageMeter
from pyll.session import PyLL

"""
This is a simple script to load the datasets (including any preprocessing steps) and save them as torch tensors. 
It is inefficient from a storage point of view, but helped improve performance from about 30 mins per epoch 
to 20 mins. 
"""


target_directory = "/home/jovyan/nfs/home/niall/data/cellpainting_torch_preprocessed"


def main():
    session = PyLL()
    datasets = session.datasets

    try:
        for k, dataset in datasets.items():
            print("Processing {} data".format(k))
            process_samples(session, dataset)
    except Exception as err:
        print("Did not run: ", err)
        raise err
    finally:
        print("All samples processed!")


def process_samples(session, dataset):
    batch_time = AverageMeter()
    end = time.time()
    for i, sample in enumerate(dataset):

        X = sample["input"]
        k = sample["ID"]

        torch.save(X, os.path.join(target_directory, f"{k}.pt"))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % session.config.print_freq == 0:
            print('Sample: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(i,
                                                                            len(dataset),
                                                                            batch_time=batch_time)
                  )


if __name__ == '__main__':
    main()
