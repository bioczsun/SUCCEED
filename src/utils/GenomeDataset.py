import os 
import h5py
import glob
from intervaltree import Interval, IntervalTree
from torch.utils.data import Dataset
from utils.chromsome_dataset import SequenceFeature, GenomicFeature
import torch
import numpy as np
import random
import pysam
import pyBigWig



def dna_1hot(seq, seq_len=None, n_uniform=False, n_sample=False):
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

class GenomeDataset(Dataset):
    def __init__(self, contig_bed,
                 seq_dir,
                 feature_dir,
                 genome_assembly,
                 mode,
                 use_cache=True,
                 use_aug=True):
        self.contig_bed = contig_bed
        self.data_dir = seq_dir
        self.use_cache = use_cache
        if mode != 'train': self.use_aug = False
        if self.use_cache is False: 
            self.chr_names = self.get_chr_names(genome_assembly)
            self.chr_data_dict = self.load_chrs(self.chr_names)
            self.feat_data = self.get_chr_data(feature_dir)
        else:
            self.feat_seq,self.feat_data = self.get_chr_data(feature_dir)
    def __len__(self):
        if self.use_cache is False:
            return len(self.contig_bed)
        else:
            return len(self.feat_data)

    def __getitem__(self, idx):
        if self.use_cache is False:
            chrom,start,end = self.contig_bed[idx][0],int(self.contig_bed[idx][1]),int(self.contig_bed[idx][2])
            seq = self.chr_data_dict[chrom].get(start, end)
            feature = self.feat_data[idx]
            return seq, feature
        else:
            seq = self.feat_seq[idx]
            feature = self.feat_data[idx]
            return seq, feature
    
    def load_chrs(self, chr_names):
        '''
        Load all chromosomes in chr_names
        '''
        chr_data_dict = {}
        for chr_name in chr_names:
            chr_data_dict[chr_name] = SequenceFeature(os.path.join(self.data_dir,f"{chr_name}.fa.gz"))
        print(f'Loaded {len(chr_data_dict)} chromosomes')
        return chr_data_dict
    
    def get_chr_names(self, assembly):
        '''
        Get a list of all chromosome names. e.g. [chr1 , chr2, ...]
        '''
        print(f'Using Assembly: {assembly}')
        if assembly in ['hg38', 'hg19']:
            chrs = list(range(1, 23))
        elif assembly in ['mm10', 'mm9']:
            chrs = list(range(1, 20))
        else: raise Exception(f'Assembly {assembly} unknown')
        chrs.append('X')
        chrs.append('Y')
        chr_names = []
        for chr_num in chrs:
            chr_names.append(f'chr{chr_num}')
        return chr_names
    
    def get_chr_data(self, feature_dir):
        '''
        Load features from h5py file
        '''
        features = h5py.File(feature_dir, 'r',swmr=True)
        if self.use_cache is False:
            return features
        else:
            return features["sequences"],features["targets"]
        
        

class PretrainDataset(Dataset):
    def __init__(self, data, augment=False):
        self.sequence = data["sequence"]  # Original data shape (length, 4)
        self.target = data["target"]
        self.augment = augment  # Whether to apply data augmentation
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.sequence)

    def reverse_complement(self, seq):
        # Reverse complement: reverse the sequence and swap A-T, C-G
        n, d = seq.shape
        assert d == 4, 'must be one hot encoding with last dimension equal to 4'
        return torch.flip(torch.Tensor(seq), (-1, -2))

    def random_base(self, length):
        """Generate random one-hot encoded base sequence"""
        bases = np.eye(4)  # One-hot encoding of A, T, C, G
        random_bases = np.random.choice(4, length)  # Randomly choose bases
        return bases[random_bases]  # Shape (length, 4)

    def shift_sequence(self, seq, shift_amount):
        """Randomly shift the sequence and keep the same length"""

        if shift_amount > 0:
            # Right shift: pad with random bases on the left, remove bases from the right
            random_bases = self.random_base(shift_amount)  # (shift_amount, 4)
            shifted_seq = np.concatenate((random_bases, seq[:-shift_amount, :]), axis=0)
        else:
            # Left shift: pad with random bases on the right, remove bases from the left
            random_bases = self.random_base(-shift_amount)  # (-shift_amount, 4)
            shifted_seq = np.concatenate((seq[-shift_amount:, :], random_bases), axis=0)

        return shifted_seq

    def __getitem__(self, idx):
        random.seed(self.epoch * 100000 + idx)
        seq = self.sequence[idx]  # Original sequence (length, 4)
        target = self.target[idx]

        if self.augment:
            # If augmentation enabled, return three different variations of the same data
            aug_type = random.choice(['orig', 'rev', 'shift'])
            if aug_type == 'orig':
                return torch.Tensor(seq), torch.Tensor(target)
            elif aug_type == 'rev':
                rev_comp_seq = self.reverse_complement(seq)  # shape (length, 4)
                rev_target = np.flip(target, axis=1).copy()  # Reverse along the first target axis
                return torch.Tensor(rev_comp_seq), torch.Tensor(rev_target)
            elif aug_type == 'shift':
                shift_amount = random.choice([-3, -2, -1, 1, 2, 3])
                shifted_seq = self.shift_sequence(seq, shift_amount)
                return torch.Tensor(shifted_seq), torch.Tensor(target)
        else:
            # If no augmentation, return original sequence and target
            return torch.Tensor(seq), torch.Tensor(target)


class EpiDataset(torch.utils.data.Dataset):
    def __init__(self, contigs, targets, fasta, atac_path, atac_list,indexes_dict=None):
        self.contigs = contigs
        self.targets = targets
        self.fasta_path = fasta
        self.atac_path = atac_path
        self.atac_list = atac_list

        self._fasta = None
        self._atac_dict = None
        self.indexes_dict = indexes_dict

    @property
    def fasta(self):
        if self._fasta is None:
            self._fasta = pysam.FastaFile(self.fasta_path)
        return self._fasta

    @property
    def atac_dict(self):
        if self._atac_dict is None:
            atac_dict = {}
            for atac_file in self.atac_list:
                file_path = os.path.join(self.atac_path, atac_file)
                atac_bw = pyBigWig.open(file_path)
                atac_dict[atac_file.split('.')[0]] = atac_bw
            self._atac_dict = atac_dict
        return self._atac_dict
      
    def get_atac_bw(self,chrom, start, end, cell_type):
        atac_bw = self.atac_dict[cell_type]
        signals = atac_bw.values(chrom, int(start), int(end))
        signals = np.nan_to_num(signals, 0)
        return np.array(signals)
    def __len__(self):
        return len(self.contigs)

    def __getitem__(self, idx):
        seq_bed = self.contigs[idx]
        seq = self.fasta.fetch(seq_bed[0], int(seq_bed[1]), int(seq_bed[2]))
        cell_type = seq_bed[-1]
        seq = dna_1hot(seq)
        target = self.targets[idx]
        atac_signal = self.get_atac_bw(seq_bed[0], seq_bed[1], seq_bed[2], cell_type)
        if self.indexes_dict is not None:
            index = self.indexes_dict[cell_type]
            return torch.tensor(seq), torch.tensor(atac_signal), torch.tensor(target), torch.tensor(index)
        return torch.tensor(seq), torch.tensor(atac_signal), torch.tensor(target)
    
    def load_fasta(self,fasta_path):
        fasta_ = pysam.FastaFile(fasta_path)
        return fasta_
    
    def load_atac(self,atac_path=None, atac_list=None):
        atac_dict = {}
        for atac_file in atac_list:
            file_path = os.path.join(atac_path, atac_file)
            atac_bw = pyBigWig.open(file_path)
            atac_dict[atac_file.split('.')[0]] = atac_bw
            print(f"Load {file_path}...")
        return atac_dict


class HistonTFDataset(Dataset):
    def __init__(self, contig_bed,
                 seq_dir,
                 feature_dir,
                 genome_assembly,
                 atac_path,
                 atac_dict,
                 mode,
                 use_cache=False):
        self.contig_bed = contig_bed
        self.data_dir = seq_dir
        self.use_cache = use_cache
        self.atac_dict = atac_dict
        self.atac_data = self.load_features(atac_path,atac_dict)
        self.mode = mode
        if mode != 'train': self.use_aug = False
        self.chr_names = self.get_chr_names(genome_assembly)
        if self.use_cache is False and mode != "inference": 
            if mode == 'train': 
                self.chr_names.remove('chr2')
                self.chr_names.remove('chr10')
                self.chr_names.remove('chr21')
            elif mode == 'test': 
                self.chr_names = ['chr10','chr21']
            elif mode == 'valid': 
                self.chr_names = ['chr2']
            else:
                raise Exception(f'Unknown mode {mode}')
            self.chr_data_dict = self.load_chrs(self.chr_names)
            self.feat_data = self.get_chr_data(feature_dir)
        elif mode == "inference":
            # self.chr_names = ['chr10']
            self.chr_data_dict = self.load_chrs(self.chr_names)
        else:
            self.feat_seq,self.feat_data = self.get_chr_data(feature_dir)
    def __len__(self):
        if self.use_cache is False:
            return len(self.contig_bed)
        else:
            return len(self.feat_data)
        

    def __getitem__(self, idx):
        if self.use_cache is False and self.mode != "inference": 
            chrom,start,end,label = self.contig_bed[idx][0],int(self.contig_bed[idx][1]),int(self.contig_bed[idx][2]),self.contig_bed[idx][3]
            seq = self.chr_data_dict[chrom].get(start, end)
            atac_signal = self.atac_data[label].get(chrom,start,end)
            feature = self.feat_data[idx]
            if self.mode == "test":
                return seq,atac_signal, feature,label,chrom
            return seq,atac_signal, feature
        elif self.use_cache is False and self.mode == "inference":
            chrom,start,end,label = self.contig_bed[idx][0],int(self.contig_bed[idx][1]),int(self.contig_bed[idx][2]),self.contig_bed[idx][3]
            seq = self.chr_data_dict[chrom].get(start, end)
            atac_signal = self.atac_data[label].get(chrom,start,end)
            return seq,atac_signal,chrom,start,end
        else:
            seq = self.feat_seq[idx]
            feature = self.feat_data[idx]
            return seq, feature
    
    def load_chrs(self, chr_names):
        '''
        Load all chromosomes in chr_names
        '''
        chr_data_dict = {}
        for chr_name in chr_names:
            chr_data_dict[chr_name] = SequenceFeature(os.path.join(self.data_dir,f"{chr_name}.fa.gz"))
        print(f'Loaded {len(chr_data_dict)} chromosomes')
        return chr_data_dict
    
    def get_chr_names(self, assembly):
        '''
        Get a list of all chromosome names. e.g. [chr1 , chr2, ...]
        '''
        print(f'Using Assembly: {assembly}')
        if assembly in ['hg38', 'hg19']:
            chrs = list(range(1, 23))
        elif assembly in ['mm10', 'mm9']:
            chrs = list(range(1, 20))
        else: raise Exception(f'Assembly {assembly} unknown')
        chrs.append('X')
        chrs.append('Y')
        chr_names = []
        for chr_num in chrs:
            chr_names.append(f'chr{chr_num}')
        return chr_names
    
    def get_chr_data(self, feature_dir):
        '''
        Load features from h5py file
        '''
        features = h5py.File(feature_dir, 'r',swmr=True)
        if self.use_cache is False:
            return features["targets"]
        else:
            return features["sequences"],features["targets"]
        
    def load_features(self, root_dir, feat_dicts):
        '''
        Args:
            features: a list of dicts with 
                1. file name
                2. norm status
        Returns:
            feature_list: a list of genomic features (bigwig files)
        '''
        feat_list = {}
        for feat_item in feat_dicts:
            file_name = feat_item
            file_path = f'{root_dir}/{file_name}'
            norm = None #"log"
            feat_list[file_name.split('.')[0]] = GenomicFeature(file_path, norm)
        return feat_list


class ATACDenoisingDataset(torch.utils.data.Dataset):
    def __init__(self, contigs, fasta, noisy_file, clean_file, peak_file):
        self.contigs = contigs
        self.fasta_file = fasta
        self.noisy_file = noisy_file
        self.clean_file = clean_file
        self.chromosome_trees = self.build_chromosome_trees(peak_file)

        self._fasta = None
        self._noisy_bw = None
        self._clean_bw = None

    @property
    def fasta(self):
        if self._fasta is None:
            self._fasta = pysam.FastaFile(self.fasta_file)
        return self._fasta

    @property
    def atac_noisy(self):
        if self._noisy_bw is None:
            self._noisy_bw = pyBigWig.open(self.noisy_file)
        return self._noisy_bw

    @property
    def atac_clean(self):
        if self._clean_bw is None:
            self._clean_bw = pyBigWig.open(self.clean_file)
        return self._clean_bw

    def __len__(self):
        return len(self.contigs)

    def __getitem__(self, idx):
        contig = self.contigs[idx]
        seq, noisy_cov, clean_cov, labels = self.get_seq_targets(contig[0], contig[1], contig[2])
        seq_1hot = dna_1hot(seq)
        return (
            torch.tensor(seq_1hot, dtype=torch.float32),
            torch.tensor(clean_cov, dtype=torch.float32),
            torch.tensor(noisy_cov, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )

    def get_seq_targets(self, chrom, start, end):
        noisy_bw = self.atac_noisy
        clean_bw = self.atac_clean
        if chrom not in noisy_bw.chroms():
            raise ValueError(f"Chromosome {chrom} not found in BigWig file.")

        chrom_length = noisy_bw.chroms(chrom)
        if int(start) >= chrom_length or int(end) > chrom_length:
            raise ValueError(f"Coordinates {start}-{end} are out of bounds for chromosome {chrom}.")

        noisy_cov = noisy_bw.values(chrom, int(start), int(end))
        clean_cov = clean_bw.values(chrom, int(start), int(end))
        noisy_cov = np.nan_to_num(np.array(noisy_cov), nan=0)
        clean_cov = np.nan_to_num(np.array(clean_cov), nan=0)

        clean_cov = clean_cov.reshape(-1, 128).max(axis=1)

        query_points = np.arange(int(start), int(end), 128)
        label_ls = []
        for point in query_points:
            label = 0
            for interval in self.chromosome_trees[chrom][point : point + 128]:
                label = interval.data
            label_ls.append(label)

        seq = self.fasta.fetch(chrom, int(start), int(end))
        return seq, noisy_cov, clean_cov, label_ls

    @staticmethod
    def build_chromosome_trees(peak_file):
        chromosome_trees = {}
        with open(peak_file) as f:
            lines = f.readlines()[1:]
            for line in lines:
                chrom, start, end, *_ = line.strip().split()
                start = int(start)
                end = int(end)
                if chrom not in chromosome_trees:
                    chromosome_trees[chrom] = IntervalTree()
                chromosome_trees[chrom].add(Interval(start, end, 1))
        return chromosome_trees
    
class NoisyDirATACDataset(torch.utils.data.Dataset):
    def __init__(self, contigs, fasta, noisy_dir, clean_file, peak_file):

        self._dna_1hot = dna_1hot
        self.contigs = contigs
        self.fasta_file = fasta
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.bw")))
        if not self.noisy_files:
            raise ValueError(f"No .bw files found in noisy_dir: {noisy_dir}")
        self.clean_file = clean_file
        self.chromosome_trees = self.build_chromosome_trees(peak_file)

        self._fasta = None
        self._clean_bw = None
        self._noisy_bw_cache = {}

    @property
    def fasta(self):
        if self._fasta is None:
            self._fasta = pysam.FastaFile(self.fasta_file)
        return self._fasta

    @property
    def atac_clean(self):
        if self._clean_bw is None:
            self._clean_bw = pyBigWig.open(self.clean_file)
        return self._clean_bw

    def _get_random_noisy_bw(self):
        bw_path = random.choice(self.noisy_files)
        bw = self._noisy_bw_cache.get(bw_path)
        if bw is None:
            bw = pyBigWig.open(bw_path)
            self._noisy_bw_cache[bw_path] = bw
        return bw

    def __len__(self):
        return len(self.contigs)

    def __getitem__(self, idx):
        contig = self.contigs[idx]
        seq, noisy_cov, clean_cov, labels = self.get_seq_targets(contig[0], contig[1], contig[2])
        seq_1hot = self._dna_1hot(seq)
        return (
            torch.tensor(seq_1hot, dtype=torch.float32),
            torch.tensor(clean_cov, dtype=torch.float32),
            torch.tensor(noisy_cov, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )

    def get_seq_targets(self, chrom, start, end):
        noisy_bw = self._get_random_noisy_bw()
        clean_bw = self.atac_clean
        if chrom not in noisy_bw.chroms():
            raise ValueError(f"Chromosome {chrom} not found in noisy BigWig file.")

        chrom_length = noisy_bw.chroms(chrom)
        if int(start) >= chrom_length or int(end) > chrom_length:
            raise ValueError(f"Coordinates {start}-{end} are out of bounds for chromosome {chrom}.")

        noisy_cov = noisy_bw.values(chrom, int(start), int(end))
        clean_cov = clean_bw.values(chrom, int(start), int(end))
        noisy_cov = np.nan_to_num(np.array(noisy_cov), nan=0)
        clean_cov = np.nan_to_num(np.array(clean_cov), nan=0)

        clean_cov = clean_cov.reshape(-1, 128).max(axis=1)

        query_points = np.arange(int(start), int(end), 128)
        label_ls = []
        for point in query_points:
            label = 0
            for interval in self.chromosome_trees[chrom][point : point + 128]:
                label = interval.data
            label_ls.append(label)

        seq = self.fasta.fetch(chrom, int(start), int(end))
        return seq, noisy_cov, clean_cov, label_ls

    @staticmethod
    def build_chromosome_trees(peak_file):
        chromosome_trees = {}
        with open(peak_file) as f:
            lines = f.readlines()[1:]
            for line in lines:
                chrom, start, end, *_ = line.strip().split()
                start = int(start)
                end = int(end)
                if chrom not in chromosome_trees:
                    chromosome_trees[chrom] = IntervalTree()
                chromosome_trees[chrom].add(Interval(start, end, 1))
        return chromosome_trees
