from utils import DNAEncoder, GenomeExtractor, onehot_encoder

# -------------------test GenomeExtractor-------------------
# create a GenomeExtractor object
extractor = GenomeExtractor("human")

# get the sequence of 2000 letters from the human genome
sequenecs = extractor.extract("chr11", 35082742, 35084742)
# convert the sequence to onehot encoding
# shape = (1, 2000, 4) = (batch_size, sequence_length, neuleotide types)
encoded_sequence = onehot_encoder([sequenecs])

# --------------------test DNAEncoder--------------------
# create a DNAEncoder model
encoder = DNAEncoder(num_layers=5, target_width=64)
encoded = encoder(encoded_sequence)
print(f"input shape: {encoded_sequence.shape}")
# resulting shape = (1, 2000/2^5, 64) = (batch_size, sequence_length/2^num_layer, width)
print(f"output shape: {encoded.shape}")
# print(encoder)
