from levenshtein import _levenshtein
import torch

def test_levenshtein():
    dataset1 = torch.randn(20, 2048)
    dataset2 = torch.randn(20, 2048)
    dataset1 = dataset1.numpy()  #.cpu().detach().numpy() 
    dataset2 = dataset2.numpy()  #.cpu().detach().numpy()

    score_mean, nrm_score_mean = _levenshtein(dataset1, dataset2)
    print(score_mean, nrm_score_mean)

test_levenshtein()
