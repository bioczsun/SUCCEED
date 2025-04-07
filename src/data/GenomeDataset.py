import os 
import h5py
from torch.utils.data import Dataset
from data.chromsome_dataset import SequenceFeature, GenomicFeature
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
        Get a list of all chr names. e.g. [chr1 , chr2, ...]
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
        load features h5py file
        '''
        features = h5py.File(feature_dir, 'r',swmr=True)
        if self.use_cache is False:
            return features
        else:
            return features["sequences"],features["targets"]
        
        

class EnformerDataset(Dataset):
    def __init__(self, data, augment=False):
        self.sequence = data["sequence"]  # 原始数据为 (length, 4)
        self.target = data["target"]
        self.augment = augment  # 是否启用数据增强

    def __len__(self):
        return len(self.sequence)

    def reverse_complement(self, seq):
        # 反向互补：反转序列并交换A-T，C-G
        n, d = seq.shape
        assert d == 4, 'must be one hot encoding with last dimension equal to 4'
        return torch.flip(torch.Tensor(seq), (-1, -2))

    def random_base(self, length):
        """生成随机的 one-hot 编码碱基序列"""
        bases = np.eye(4)  # A, T, C, G 的 one-hot 编码
        random_bases = np.random.choice(4, length)  # 随机选择碱基
        return bases[random_bases]  # 生成形状 (length, 4)

    def shift_sequence(self, seq, shift_amount):
        """随机位移序列，并保持长度一致"""

        if shift_amount > 0:
            # 右移，前面填充 shift_amount 个随机碱基，右边移除 shift_amount 个碱基
            random_bases = self.random_base(shift_amount)  # (shift_amount, 4)
            # 截取从 0 到 seq_length - shift_amount 的序列部分
            shifted_seq = np.concatenate((random_bases, seq[:-shift_amount, :]), axis=0)
        else:
            # 左移，后面填充 -shift_amount 个随机碱基，左边移除 -shift_amount 个碱基
            random_bases = self.random_base(-shift_amount)  # (shift_amount, 4)
            # 截取从 -shift_amount 到 seq_length 的序列部分
            shifted_seq = np.concatenate((seq[-shift_amount:, :], random_bases), axis=0)

            

        return shifted_seq

    def __getitem__(self, idx):
        seq = self.sequence[idx]  # 原始数据为 (length, 4)
        target = self.target[idx]

        if self.augment:
            # 如果启用增强，返回三种不同形式的同一条数据
            orig_seq = seq  # (2097512, 4)
            rev_comp_seq = self.reverse_complement(seq)  # 反向互补，保持形状 (length, 4)
            shift_amount = random.choice([-3, -2, -1, 1, 2, 3])  # 随机位移1-3个bp
            shifted_seq = self.shift_sequence(seq, shift_amount)  # 位移后保持形状 (length, 4)
            return torch.Tensor(orig_seq), torch.Tensor(rev_comp_seq), torch.Tensor(shifted_seq), torch.Tensor(target)
        else:
            # 如果不启用增强，直接返回原始序列和目标
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
        Get a list of all chr names. e.g. [chr1 , chr2, ...]
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
        load features h5py file
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
            norm = None#"log"
            feat_list[file_name.split('.')[0]] = GenomicFeature(file_path, norm)
        return feat_list
        

    