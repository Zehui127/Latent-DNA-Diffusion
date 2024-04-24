from kipoiseq import Interval
import logging
import pyfaidx
import os
import subprocess


class GenomeExtractor:
    """
    default supported organisms: human, mouse
    other organisms can be added by adding the organism name to _GENOME_NAMES
    from https://hgdownload.soe.ucsc.edu/downloads.html

    usage:
    >>> genome = GenomeExtractor("human")
    >>> genome.extract("chr11", 35082742, 35197430)

    get all the available chromosomes:
    >>> genome._chromosome_sizes
    Returns:
        an object that can extract DNA sequences from a genome
    """

    _GENOME_NAMES = {"human": "hg38", "mouse": "mm10"}
    _FASTA_PATH = ""

    def __init__(self, organism: str):
        fasta_file = self.get_fasta(organism)
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, chrom, start, end, **kwargs) -> str:
        """_summary_

        Args:
            chrom: chromosome name, e.g. 'chr11'
            start: starting position, e.g. 35082742
            end: ending position, e.g. 35197430

        Returns:
            the sequence of DNA in the given interval
        """
        # construct interval
        interval = Interval(chrom, start, end)
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(
            interval.chrom,
            max(interval.start, 0),
            min(interval.end, chromosome_length),
        )
        # pyfaidx wants a 1-based interval
        sequence = str(
            self.fasta.get_seq(
                trimmed_interval.chrom,
                trimmed_interval.start + 1,
                trimmed_interval.stop,
            ).seq
        ).upper()
        # Fill truncated values with N's.
        pad_upstream = "N" * max(-interval.start, 0)
        pad_downstream = "N" * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()

    def get_fasta(self, organism: str) -> str:
        """Get genome FASTA file for a given organism.
        If not present locally, will download the relevant
        file from http://hgdownload.cse.ucsc.edu/, and convert to .bgz
        Args:
            organism (str): human or mouse
        Returns:
            Fasta file path
        """
        path = os.path.join(
            self._FASTA_PATH,
            f"{self._GENOME_NAMES[organism]}.fa.bgz",
        )
        if not os.path.exists(path):
            gzpath = os.path.splitext(path)[0] + ".gz"
            if not os.path.exists(gzpath):
                fasta = (
                    "http://hgdownload.cse.ucsc.edu"
                    + f"/goldenPath/{self._GENOME_NAMES[organism]}"
                    + f"/bigZips/{self._GENOME_NAMES[organism]}.fa.gz"
                )
                print(fasta)
                logging.info(f"Downloading {fasta} to {gzpath}")
                process = subprocess.Popen(
                    f"curl -o {gzpath} {fasta}", shell=True
                )
                process.wait()
            logging.info(
                "Converting gz to bgz (this may take a couple minutes)"
            )
            bgrezip(gzpath)
        return os.path.splitext(path)[0]


def bgrezip(file_path: str):
    file, _ = os.path.splitext(file_path)
    logging.info(f"Rezipping {file_path}")
    process = subprocess.Popen(f"gunzip -c {file_path} > {file}", shell=True)
    process.wait()
    return file
