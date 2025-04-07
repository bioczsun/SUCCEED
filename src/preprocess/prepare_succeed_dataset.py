import pysam
import numpy as np
import argparse
import h5py
import os
from collections import namedtuple
from multiprocessing import Pool, cpu_count
from tqdm import tqdm  # 导入 tqdm 进度条库

# 添加命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='提取序列并保存为h5格式')
    parser.add_argument('--fasta', type=str, required=True, help='参考基因组路径')
    parser.add_argument('--bed', type=str, required=True, help='序列BED文件路径')
    parser.add_argument('--num_targets', type=int, required=True, help='目标特征数量')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录路径')
    return parser.parse_args()

# 读取文件，把fasta转换为one-hot编码
def onehot_seq(seq, seq_len=None, n_uniform=False, n_sample=False):
  """ dna_1hot

  Args:
   seq:    nucleotide sequence.
   seq_len:  length to extend/trim sequences to.
   n_uniform: represent N's as 0.25, forcing float16,
   n_sample: sample ACGT for N

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

# Define the mapping function for each sequence
def map_seq_to_targets(ti, contigs, fasta, targets):
    contig = contigs[ti]
    target = targets[:, ti, :]

    # Fetch and encode sequence
    seq = fasta.fetch(contig.chrom, int(contig.start), int(contig.end))
    one_hot_seq_array = onehot_seq(seq)

    return one_hot_seq_array, target, contig.name

# Define function to split sequences by type
def split_sequences(results, output_dir):
    train_seqs, train_targets = [], []
    valid_seqs, valid_targets = [], []
    test_seqs, test_targets = [], []

    for one_hot_seq, target, name in results:
        if name == "train":
            train_seqs.append(one_hot_seq)
            train_targets.append(target)
        elif name == "valid":
            valid_seqs.append(one_hot_seq)
            valid_targets.append(target)
        elif name == "test":
            test_seqs.append(one_hot_seq)
            test_targets.append(target)

    # 保存为h5格式
    with h5py.File(os.path.join(output_dir, "train_all.h5"), 'w') as f:
        f.create_dataset('sequence', data=np.stack(train_seqs))
        f.create_dataset('target', data=np.stack(train_targets))
    
    with h5py.File(os.path.join(output_dir, "valid_all.h5"), 'w') as f:
        f.create_dataset('sequence', data=np.stack(valid_seqs))
        f.create_dataset('target', data=np.stack(valid_targets))
    
    with h5py.File(os.path.join(output_dir, "test_all.h5"), 'w') as f:
        f.create_dataset('sequence', data=np.stack(test_seqs))
        f.create_dataset('target', data=np.stack(test_targets))

# Main function to handle processing with progress bar
if __name__ == "__main__":
    args = parse_args()
    
    # 创建输出目录(如果不存在)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化fasta文件和加载sequences.bed
    fasta = pysam.FastaFile(args.fasta)
    sequence_bed = open(args.bed).readlines()

    # 定义Contig命名元组并解析BED文件
    Contig = namedtuple('Contig', ['chrom', 'start', 'end', 'name'])
    contigs = [Contig(*line.strip().split('\t')[:4]) for line in sequence_bed]

    # 加载targets数据
    num_seqs = len(contigs)
    num_targets = args.num_targets
    targets_ls = [np.load(f"{os.path.dirname(args.bed)}/seqs_cov/{i}.h5.npy", allow_pickle=True) for i in range(num_targets)]
    targets = np.stack(targets_ls)

    # 使用tqdm的update方法实现进度条
    with tqdm(total=num_seqs) as pbar:
        results = []
        # 由于map_seq_to_targets需要更多参数，这里改为循环
        for i in range(num_seqs):
            result = map_seq_to_targets(i, contigs, fasta, targets)
            results.append(result)
            pbar.update(1)  # 每完成一个任务，更新进度条

    # 将结果进行分类和保存
    split_sequences(results, args.output_dir)