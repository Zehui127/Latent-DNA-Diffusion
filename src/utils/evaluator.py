from collections import Counter
import math
import numpy


class Diversity:
    def __init__(self,sequences):
        self.sequences = sequences

    def __call__(self, sequences):
        return self.avg_entropy(self.sequences), self.uniqueness(self.sequences)

    def avg_entropy(self, sequences):
        seq_entropy = [self.entropy(seq) for seq in sequences]
        return sum(seq_entropy)/len(seq_entropy)

    def entropy(self, sequence):
        """
        large entropy -> random = diversity
        small entropy -> non-randomness
        """
        # Count each base's occurrences
        base_counts = Counter(sequence)

        # Calculate sequence length
        sequence_length = len(sequence)

        # Calculate entropy
        entropy = 0
        for base, count in base_counts.items():
            p_x = count / sequence_length
            if p_x > 0:  # Avoid log(0)
                entropy += - p_x * math.log2(p_x)
        return entropy

    def uniqueness(self,sequences):
        """
        ranging from 0 to 1
        large value indicates most of the sequences are unique
        small value indicates the most of sequences are the same
        """
        unique_sequences = set(sequences)
        return (len(unique_sequences) / len(sequences))

class FID:
    def __init__(ckpt_path=""):
        self.encoder = self.load_model(ckpt_path)

    def __call__(examples, samples):
        """
        encoder the examples and samples
        then compute the difference between examples and samples
        """
        examples_encode = self.encoder(examples)
        samples_encode = self.encoder(samples)
        # compute the distance between two set of vectors
        return self.calculate_fid(examples_encode,samples_encode)

    def load_model(self,model_path):
        return []

    def calculate_fid(self, act1, act2):
        from numpy import cov
        from numpy import trace
        from numpy import iscomplexobj
        from numpy.random import random
        from scipy.linalg import sqrtm
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = numpy.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if numpy.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid


def plot_motifs(GC,TATA,Init,CCAAT):
    # motif frequency is of shape n*2
    # draw the graph
    pass

def write_to_fasta(sequences, output_file):
    """
    Write a list of DNA sequences to a FASTA file.

    :param sequences: A list of DNA sequences.
    :param output_file: The name of the FASTA file to write to.
    """
    with open(output_file, 'a') as f:
        for i, seq in enumerate(sequences):
            # Using a generic header format: >sequence_1, >sequence_2, etc.
            f.write(f">sequence_{i+1}\n{seq}\n")

def check_diversity():
    for i in range(0,50,5):
        file_path = f'saved_models_unet50_vae90/epoch {i}_samples.pth'
        temp = torch.load(file_path)
        print(Diversity(temp)())
