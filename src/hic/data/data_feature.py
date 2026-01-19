import gzip
import numpy as np
import pyBigWig as pbw
import h5py
import pandas as pd

class Feature():

    def __init__(self, **kwargs):
        self.load(**kwargs)
    
    def load(self):
        raise Exception('Not implemented')

    def get(self):
        raise Exception('Not implemented')

    def __len__(self):
        raise Exception('Not implemented')

class HiCFeature(Feature):

    def load(self, path = None):
        self.hic = self.load_hic(path)

    def get(self, start, window = 2097152, res = 10000):
        start_bin = int(start / res)
        range_bin = int(window / res)
        end_bin = start_bin + range_bin
        hic_mat = self.diag_to_mat(self.hic, start_bin, end_bin)
        return hic_mat

    def load_hic(self, path):
        print(f'Reading Hi-C: {path}')
        return dict(np.load(path))

    def diag_to_mat(self, ori_load, start, end):
        '''
        Only accessing 256 x 256 region max, two loops are okay
        '''
        square_len = end - start
        diag_load = {}
        for diag_i in range(square_len):
            diag_load[str(diag_i)] = ori_load[str(diag_i)][start : start + square_len - diag_i]
            diag_load[str(-diag_i)] = ori_load[str(-diag_i)][start : start + square_len - diag_i]
        start -= start
        end -= start

        diag_region = []
        for diag_i in range(square_len):
            diag_line = []
            for line_i in range(-1 * diag_i, -1 * diag_i + square_len):
                if line_i < 0:
                    diag_line.append(diag_load[str(line_i)][start + line_i + diag_i])
                else:
                    diag_line.append(diag_load[str(line_i)][start + diag_i])
            diag_region.append(diag_line)
        diag_region = np.array(diag_region).reshape(square_len, square_len)
        return diag_region

    def __len__(self):
        return len(self.hic['0'])

class GenomicFeatureSingleThread(Feature):

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

class SequenceFeature(Feature):

    def load(self, path = None):
        self.seq = self.read_seq(path)

    def get(self, start, end):
        seq = self.seq_to_npy(self.seq, start, end)
        # onehot_seq = self.onehot_encode(seq)
        onehot_seq = self.dna_1hot(seq)
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

    def onehot_encode(self, seq):
        ''' 
        encode integer dna array to onehot (n x 5)
        Args:
            seq (arr): Numpy array (n x 1) of dna encoded as 0-4 integers

        Returns:
            array: A numpy matrix (n x 5)
        '''
        seq_emb = np.zeros((len(seq), 5))
        seq_emb[np.arange(len(seq)), seq] = 1
        return seq_emb
    
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

class H5IntervalEmbeddingProvider:
    """
    Lazy loader that keeps precomputed sequence embeddings on disk and exposes
    helpers to fetch them according to intervals defined in a BED file.
    """

    def __init__(self, bed_path, h5_path, dataset_key='embeddings_tensor'):
        self.bed_path = bed_path
        self.h5_path = h5_path
        self.dataset_key = dataset_key
        self._h5_file = None
        self._dataset = None
        self.embedding_shape = None
        self.embedding_length = None
        self.embedding_dim = None

        self.interval_df = pd.read_csv(
            bed_path, sep='\t', names=['chr', 'start', 'end'], header=None
        )
        self.interval_lookup = {}
        for chr_name, sub_df in self.interval_df.groupby('chr', sort=False):
            intervals = sub_df[['start', 'end']].to_numpy(dtype=int)
            indices = sub_df.index.to_numpy()
            self.interval_lookup[chr_name] = {
                'intervals': intervals,
                'indices': indices
            }
        with h5py.File(self.h5_path, 'r') as h5_file:
            dataset = h5_file[self.dataset_key]
            self.embedding_shape = dataset.shape
            if dataset.ndim < 2:
                raise ValueError(f'Embedding dataset {self.dataset_key} must be at least 2D, found shape {dataset.shape}')
            if dataset.ndim == 2:
                self.embedding_length = 1
                self.embedding_dim = dataset.shape[1]
            else:
                self.embedding_length = dataset.shape[1]
                self.embedding_dim = dataset.shape[-1]

    def _get_dataset(self):
        if self._dataset is None:
            self._h5_file = h5py.File(self.h5_path, 'r')
            self._dataset = self._h5_file[self.dataset_key]
        return self._dataset

    def get_chr_intervals(self, chr_name):
        if chr_name not in self.interval_lookup:
            return np.empty((0, 2), dtype=int), np.array([], dtype=int)
        lookup = self.interval_lookup[chr_name]
        return lookup['intervals'], lookup['indices']

    def get_embedding(self, idx):
        dataset = self._get_dataset()
        return np.array(dataset[idx], dtype=np.float32)

    def close(self):
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None
            self._dataset = None

    def __del__(self):
        self.close()

if __name__ == '__main__':
    main()
