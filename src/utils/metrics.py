import warnings
import torch
from timeit import default_timer


# Simple Base-Matching Accuracy
def batch_accuracy(recons, batch_in):
    dim = recons.dim()-1
    diff = torch.flatten(torch.argmax(recons, dim=dim)) - torch.flatten(torch.argmax(batch_in, dim=dim))

    return (len(diff)-torch.count_nonzero(diff))/len(diff)


def get_align_dist(arr1, arr2, ld_build=None):
    if not ld_build:
        raise Warning("Compiled ld_build not provided.")
        return -1
    a = torch.argmax(arr1, dim=2)
    a = a.cpu().numpy()
    b = torch.argmax(arr2, dim=2)
    b = b.cpu().numpy()

    return ld_build(a, b)

def timer(func, *args):
    tic = default_timer()
    func(*args)
    toc = default_timer()
    return toc-tic

def time_metric(shape=(8,2048,4)):

    a = torch.randn(*shape)
    b = torch.zeros(*shape)

    time = timer(get_align_dist, a, b)
    print("Align:", time)
    time = timer(batch_accuracy, a, b)
    print("Accuracy:", time)

# for debugging purposes
# time_metric()
