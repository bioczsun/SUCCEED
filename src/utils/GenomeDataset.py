import os 
import h5py
from torch.utils.data import Dataset
from utils.chromsome_dataset import SequenceFeature, GenomicFeature
import torch
import numpy as np
import random

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
        self.sequence = data["sequences"]  # Original data shape (length, 4)
        self.target = data["targets"]
        self.augment = augment  # Whether to apply data augmentation

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
        seq = self.sequence[idx]  # Original sequence (length, 4)
        target = self.target[idx]

        if self.augment:
            # If augmentation enabled, return three different variations of the same data
            orig_seq = seq  
            rev_comp_seq = self.reverse_complement(seq)  # Reverse complement, shape (length, 4)
            shift_amount = random.choice([-3, -2, -1, 1, 2, 3])  # Random shift by 1–3 bp
            shifted_seq = self.shift_sequence(seq, shift_amount)  # Keep same shape
            return torch.Tensor(orig_seq), torch.Tensor(rev_comp_seq), torch.Tensor(shifted_seq), torch.Tensor(target)
        else:
            # If no augmentation, return original sequence and target
            return torch.Tensor(seq), torch.Tensor(target)
        

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
