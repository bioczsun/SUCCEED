import gzip
import numpy as np
import pyBigWig as pbw


class SequenceFeature():
    def __init__(self,path):
        self.path = path
        self.load(path)
        
    '''
    Class to load the chromosome data
    '''
    def load(self, path):
        self.seq = self.read_seq(path)

    def get(self, start, end,n_uniform=True):
        seq = self.seq_to_npy(self.seq, start, end)
        # onehot_seq = self.onehot_encode(seq)
        onehot_seq = self.dna_1hot(seq,n_uniform=True)
        return onehot_seq

    def __len__(self):
        return len(self.seq)


    def read_seq(self, dna_dir):
        '''
        Transform fasta data to numpy array
        
        Args:
            dna_dir (str): Directory to DNA .fa path

        Returns:
            array: A numpy char array that contains DNA for a chromosome
        '''
        print(f'Reading sequence: {dna_dir}')
        with gzip.open(dna_dir, 'r') as f:
            seq = f.read().decode("utf-8")
        seq = seq[seq.find('\n'):]
        seq = seq.replace('\n', '').lower()
        return seq
        
    def seq_to_npy(self, seq, start, end):
        '''
        Transform fasta data to integer numpy array
        
        Args:
            dna_dir (str): Directory to DNA .fa path

        Returns:
            array: A numpy char array that contains DNA for a chromosome
        '''
        seq = seq[start : end]
        # en_dict = {'a' : 0, 't' : 1, 'c' : 2, 'g' : 3, 'n' : 4}
        # en_seq = [en_dict[ch] for ch in seq]
        # np_seq = np.array(en_seq, dtype = int)
        return seq
    
    def dna_1hot(self, seq, seq_len=None, n_uniform=False, n_sample=False):
        """ dna_1hot

        Args:
        seq:       nucleotide sequence.
        seq_len:   length to extend/trim sequences to.
        n_uniform: represent N's as 0.25, forcing float16,
        n_sample:  sample ACGT for N

        Returns:
        seq_code: length by nucleotides array representation.
        """
        if seq_len is None:
            seq_len = len(seq)
            seq_start = 0
        else:
            if seq_len <= len(seq):
                # trim the sequence
                seq_trim = (len(seq) - seq_len) // 2
                seq = seq[seq_trim:seq_trim + seq_len]
                seq_start = 0
            else:
                seq_start = (seq_len - len(seq)) // 2

        seq = seq.upper()

        # map nt's to a matrix len(seq)x4 of 0's and 1's.
        if n_uniform:
            seq_code = np.zeros((seq_len, 4), dtype='float16')
        else:
            seq_code = np.zeros((seq_len, 4), dtype='bool')
            
        for i in range(seq_len):
            if i >= seq_start and i - seq_start < len(seq):
                nt = seq[i - seq_start]
                if nt == 'A':
                    seq_code[i, 0] = 1
                elif nt == 'C':
                    seq_code[i, 1] = 1
                elif nt == 'G':
                    seq_code[i, 2] = 1
                elif nt == 'T':
                    seq_code[i, 3] = 1
                else:
                    # Set N or unknown bases to zero
                    seq_code[i, :] = 0

        return seq_code
    
    
class GenomicFeatureSingleThread():

    def __init__(self, path, norm):
        self.path = path
        self.load(path)
        self.norm = norm
        print(f'Feature path: {path} \n Normalization status: {norm}')

    def load(self, path):
        self.feature = self.read_feature(path)

    def get(self, chr_name, start, end):
        feature = self.feature_to_npy(chr_name, start, end)
        feature = np.nan_to_num(feature, 0) # Important! replace nan with 0
        if self.norm == 'log':
            feature = np.log(feature + 1)
        elif self.norm is None:
            feature = feature
        else:
            raise Exception(f'Norm type {self.norm} undefined')
        return feature

    def read_feature(self, path):
        '''
        read bigwig file
        '''
        bw_file = pbw.open(path)
        return bw_file

    def feature_to_npy(self, chr_name, start, end):
        signals = self.feature.values(chr_name, start, end)
        return np.array(signals)

    def length(self, chr_name):
        return self.feature.chroms(chr_name)

class GenomicFeature(GenomicFeatureSingleThread):

    def __init__(self, path, norm):
        self.path = path
        self.norm = norm
        print(f'Feature path: {path} \n Normalization status: {norm}')

    def load(self, path):
        raise Exception('Left blank')

    def feature_to_npy(self, chr_name, start, end):
        with pbw.open(self.path) as bw_file:
            signals = bw_file.values(chr_name, int(start), int(end))
        return np.array(signals)

    def length(self, chr_name):
        with pbw.open(self.path) as bw_file:
            length = bw_file.chroms(chr_name)
        return length