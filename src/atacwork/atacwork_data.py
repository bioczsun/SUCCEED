import argparse
import random
import heapq
import sys
import pysam
import pyBigWig
import subprocess
import tempfile
import collections
import numpy as np
import h5py
import os
from multiprocessing import Pool
from tqdm import tqdm

from intervaltree import Interval, IntervalTree


################################################################################
Contig = collections.namedtuple('Contig', ['chr', 'start', 'end'])
ModelSeq = collections.namedtuple('ModelSeq', ['chr', 'start', 'end', 'label'])

parser = argparse.ArgumentParser(description='ATACwork data')
parser.add_argument('--out_dir', type=str, default='data', help='data directory')
parser.add_argument('--name', type=str, default='name', help='data name directory')
parser.add_argument('--clean_file', type=str, help='Path to the clean BigWig file')
parser.add_argument('--noisy_file', type=str, help='Path to the noisy BigWig file')
parser.add_argument('--peak_file', type=str, help='Path to the peaks file')
parser.add_argument("--fasta_file", type=str, help="Path to the fasta file")
parser.add_argument("--gaps_file", type=str, help="Path to the gaps file")
parser.add_argument("--limit_bed", type=str, help="Path to the limit bed file")
parser.add_argument("--break_t", type=int,default=786432,help="Break large contigs")
parser.add_argument("--seed", type=int, default=1401, help="Random seed")
parser.add_argument("--restart", action="store_true")

params = parser.parse_args()

out_dir = params.out_dir
fasta_file = params.fasta_file
full_file = params.clean_file
sample_file = params.noisy_file
peak_file = params.peak_file
cell_name = params.name


random.seed(params.seed)
np.random.seed(params.seed)

##创建tree
chromosome_trees = {}
peaks_df = open(peak_file).readlines()[1:]
for line in peaks_df:
    line = line.strip().split()
    chrom = line[0]
    start = int(line[1])
    end = int(line[2])
    if chrom not in chromosome_trees:
        chromosome_trees[chrom] = IntervalTree()
    chromosome_trees[chrom].add(Interval(start, end, 1))
    

################################################################################

def main():
    # setup output directory
    if os.path.isdir(params.out_dir) and not params.restart:
        print('Remove output directory %s or use --restart option.' % params.out_dir)
        exit(1)

    if not os.path.isdir(params.out_dir):
        os.makedirs(params.out_dir)

    if params.restart:
        chrom_contigs = load_chromosomes(fasta_file)
        # remove gaps
        if params.gaps_file:
            chrom_contigs = split_contigs(chrom_contigs,
                                            params.gaps_file)

        # ditch the chromosomes for contigs
        contigs = []
        for chrom in chrom_contigs:
            contigs += [Contig(chrom, ctg_start, ctg_end)
                    for ctg_start, ctg_end in chrom_contigs[chrom]]

        # limit to a BED file
        if params.limit_bed is not None:
            contigs = limit_contigs(contigs, params.limit_bed)



        # filter for large enough
        seq_tlength = 131072
        contigs = [ctg for ctg in contigs if ctg.end - ctg.start >= seq_tlength]

        # break up large contigs
        if params.break_t is not None:
            contigs = break_large_contigs(contigs, params.break_t)

        # divide contigs into train, val, test
        train_contigs, valid_contigs, test_contigs = divide_contig_components(contigs)

        # rejoin broken contigs within set
        train_contigs = rejoin_large_contigs(train_contigs)
        valid_contigs = rejoin_large_contigs(valid_contigs)
        test_contigs = rejoin_large_contigs(test_contigs)


        ################################################################
        # define model sequences
        ################################################################

        # stride sequences across contig
        train_mseqs = contig_sequences(train_contigs, seq_tlength,
                                        65536, 'train')
        valid_mseqs = contig_sequences(valid_contigs, seq_tlength,
                                        65536, 'valid')
        test_mseqs = contig_sequences(test_contigs, seq_tlength,
                                        65536, 'test')

        all_mseqs = train_mseqs + valid_mseqs + test_mseqs

        # shuffle
        random.shuffle(all_mseqs)

        # 保存为bed文件
        with open("%s/sequence.bed"%out_dir,"w") as f:
            for mseq in all_mseqs:
                f.write("%s\t%d\t%d\t%s\n"%(mseq.chr,mseq.start,mseq.end,mseq.label))

        return all_mseqs


def get_seq_targets(contig, sample_file, full_file,chromosome_trees):
    
    sample_bw = pyBigWig.open(sample_file)
    full_bw = pyBigWig.open(full_file)
    if contig.chr not in sample_bw.chroms():
        raise ValueError(f"Chromosome {contig.chr} not found in BigWig file.")

    chrom_length = sample_bw.chroms(contig.chr)
    if int(contig.start) >= chrom_length or int(contig.end) > chrom_length:
        raise ValueError(f"Coordinates {contig.start}-{contig.end} are out of bounds for chromosome {contig.chr}.")

    # 获取覆盖率
    sample_coverage = sample_bw.values(contig.chr, int(contig.start), int(contig.end))
    full_coverage = full_bw.values(contig.chr, int(contig.start), int(contig.end))
    sample_coverage = np.nan_to_num(np.array(sample_coverage), nan=0)
    full_coverage = np.nan_to_num(np.array(full_coverage), nan=0)

    # 将数据调整到128分辨率
    sample_coverage = sample_coverage
    full_coverage = full_coverage.reshape(-1,128).max(axis=1)

    # 获取peak
    query_points = np.arange(int(contig.start), int(contig.end), 128)
    label_ls = []
    for point in query_points:
        chr = contig.chr
        label = 0
        for interval in chromosome_trees[chr][point:point+128]:
            label = interval.data
        label_ls.append(label)

    return sample_coverage, full_coverage, label_ls

def map_seq_to_targets(contig,fasta,sample_file,full_file):
    # Fetch and encode sequence
    seq = fasta.fetch(contig.chr, int(contig.start), int(contig.end))
    sample_coverage, full_coverage,peak_label = get_seq_targets(contig, sample_file, full_file,chromosome_trees)
    one_hot_seq_array = dna_1hot(seq)

    return one_hot_seq_array, sample_coverage, full_coverage,peak_label,contig.label


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

############################################################################################
def is_standard_chrom(chrom):
    if chrom.startswith("chr") and len(chrom) < 6:
        if chrom[3].isdigit() or chrom[3] == "X":
            return True
        else:
            return False
    else:
        return False

def load_chromosomes(genome_file):
  """ Load genome segments from either a FASTA file or
          chromosome length table. """

  # is genome_file FASTA or (chrom,start,end) table?
  file_fasta = (open(genome_file).readline()[0] == '>')

  chrom_segments = {}

  if file_fasta:
    fasta_open = pysam.FastaFile(genome_file)
    for i in range(len(fasta_open.references)):
      chrom_segments[fasta_open.references[i]] = [(0, fasta_open.lengths[i])]
    fasta_open.close()

  else:
    for line in open(genome_file):
      a = line.split()
      chrom_segments[a[0]] = [(0, int(a[1]))]

  # remove non-standard chromosomes
  for chrom in list(chrom_segments.keys()):
    if not is_standard_chrom(chrom):
      del chrom_segments[chrom]
  return chrom_segments


def split_contigs(chrom_segments, gaps_file):
  """ Split the assembly up into contigs defined by the gaps.

    Args:
      chrom_segments: dict mapping chromosome names to lists of (start,end)
      gaps_file: file specifying assembly gaps

    Returns:
      chrom_segments: same, with segments broken by the assembly gaps.
    """

  chrom_events = {}

  # add known segments
  for chrom in chrom_segments:
    if len(chrom_segments[chrom]) > 1:
      print(
          "I've made a terrible mistake...regarding the length of chrom_segments[%s]"
          % chrom,
          file=sys.stderr)
      exit(1)
    cstart, cend = chrom_segments[chrom][0]
    chrom_events.setdefault(chrom, []).append((cstart, 'Cstart'))
    chrom_events[chrom].append((cend, 'cend'))

  # add gaps
  for line in open(gaps_file):
    a = line.split()
    chrom = a[0]
    gstart = int(a[1])
    gend = int(a[2])

    # consider only if its in our genome
    if chrom in chrom_events:
      chrom_events[chrom].append((gstart, 'gstart'))
      chrom_events[chrom].append((gend, 'Gend'))

  for chrom in chrom_events:
    # sort
    chrom_events[chrom].sort()

    # read out segments
    chrom_segments[chrom] = []
    for i in range(len(chrom_events[chrom]) - 1):
      pos1, event1 = chrom_events[chrom][i]
      pos2, event2 = chrom_events[chrom][i + 1]

      event1 = event1.lower()
      event2 = event2.lower()

      shipit = False
      if event1 == 'cstart' and event2 == 'cend':
        shipit = True
      elif event1 == 'cstart' and event2 == 'gstart':
        shipit = True
      elif event1 == 'gend' and event2 == 'gstart':
        shipit = True
      elif event1 == 'gend' and event2 == 'cend':
        shipit = True
      elif event1 == 'gstart' and event2 == 'gend':
        pass
      else:
        print(
            "I'm confused by this event ordering: %s - %s" % (event1, event2),
            file=sys.stderr)
        exit(1)

      if shipit and pos1 < pos2:
        chrom_segments[chrom].append((pos1, pos2))

  # remove non-standard chromosomes
  for chrom in list(chrom_segments.keys()):
    if not is_standard_chrom(chrom):
      del chrom_segments[chrom]

  return chrom_segments


################################################################################
def limit_contigs(contigs, filter_bed):
  """ Limit to contigs overlapping the given BED.

    Args
     contigs: list of Contigs
     filter_bed: BED file to filter by

    Returns:
     fcontigs: list of Contigs
    """

  # print ctgments to BED
  ctg_fd, ctg_bed_file = tempfile.mkstemp()
  ctg_bed_out = open(ctg_bed_file, 'w')
  for ctg in contigs:
    print('%s\t%d\t%d' % (ctg.chr, ctg.start, ctg.end), file=ctg_bed_out)
  ctg_bed_out.close()

  # intersect w/ filter_bed
  fcontigs = []
  p = subprocess.Popen(
      'bedtools intersect -a %s -b %s' % (ctg_bed_file, filter_bed),
      shell=True,
      stdout=subprocess.PIPE)
  for line in p.stdout: # type: ignore
    a = line.decode('utf-8').split()
    chrom = a[0]
    ctg_start = int(a[1])
    ctg_end = int(a[2])
    fcontigs.append(Contig(chrom, ctg_start, ctg_end))

  p.communicate()

  os.close(ctg_fd)
  os.remove(ctg_bed_file)

  return fcontigs

################################################################################
def break_large_contigs(contigs, break_t, verbose=False):
  """Break large contigs in half until all contigs are under
     the size threshold."""

  # initialize a heapq of contigs and lengths
  contig_heapq = []
  for ctg in contigs:
    ctg_len = ctg.end - ctg.start
    heapq.heappush(contig_heapq, (-ctg_len, ctg))

  ctg_len = break_t + 1
  while ctg_len > break_t:

    # pop largest contig
    ctg_nlen, ctg = heapq.heappop(contig_heapq)
    ctg_len = -ctg_nlen

    # if too large
    if ctg_len > break_t:
      if verbose:
        print('Breaking %s:%d-%d (%d nt)' % (ctg.chr,ctg.start,ctg.end,ctg_len))

      # break in two
      ctg_mid = ctg.start + ctg_len//2

      try:
        ctg_left = Contig(ctg.genome, ctg.chr, ctg.start, ctg_mid) # type: ignore
        ctg_right = Contig(ctg.genome, ctg.chr, ctg_mid, ctg.end) # type: ignore
      except AttributeError:
        ctg_left = Contig(ctg.chr, ctg.start, ctg_mid)
        ctg_right = Contig(ctg.chr, ctg_mid, ctg.end)

      # add left
      ctg_left_len = ctg_left.end - ctg_left.start
      heapq.heappush(contig_heapq, (-ctg_left_len, ctg_left))

      # add right
      ctg_right_len = ctg_right.end - ctg_right.start
      heapq.heappush(contig_heapq, (-ctg_right_len, ctg_right))

  # return to list
  contigs = [len_ctg[1] for len_ctg in contig_heapq]

  return contigs

def divide_contig_components(contigs,test="chr10",val="chr2"):
    test_contigs = []
    val_contigs = []
    train_contigs = []
    for ctg in contigs:
        if ctg.chr == test:
            test_contigs.append(ctg)
        elif ctg.chr == val:
            val_contigs.append(ctg)
        else:
            train_contigs.append(ctg)
    return train_contigs,val_contigs,test_contigs

################################################################################
def contig_sequences(contigs, seq_length, stride=65536, label=None):
  ''' Break up a list of Contig's into a list of model length
       and stride sequence contigs.'''
  mseqs = []

  for ctg in contigs:
    seq_start = ctg.start
    seq_end = seq_start + seq_length

    while seq_end < ctg.end:
      # record sequence
      mseqs.append(ModelSeq(ctg.chr, seq_start, seq_end, label))

      # update
      seq_start += stride
      seq_end += stride

  return mseqs

################################################################################
def rejoin_large_contigs(contigs):
  """ Rejoin large contigs that were broken up before alignment comparison."""

  # split list by chromosome
  chr_contigs = {}
  for ctg in contigs:
    chr_contigs.setdefault(ctg.chr,[]).append(ctg)

  contigs = []
  for chrm in chr_contigs:
    # sort within chromosome
    chr_contigs[chrm].sort(key=lambda x: x.start)

    ctg_ongoing = chr_contigs[chrm][0]
    for i in range(1, len(chr_contigs[chrm])):
      ctg_this = chr_contigs[chrm][i]
      if ctg_ongoing.end == ctg_this.start:
        # join
        # ctg_ongoing.end = ctg_this.end
        ctg_ongoing = ctg_ongoing._replace(end=ctg_this.end)
      else:
        # conclude ongoing
        contigs.append(ctg_ongoing)

        # move to next
        ctg_ongoing = ctg_this

    # conclude final
    contigs.append(ctg_ongoing)

  return contigs

# Define function to split sequences by type
def split_sequences(results,output_dir):
    train_seqs, train_sample,train_full,train_label = [], [] ,[],[]
    valid_seqs, valid_sample,valid_full,valid_label = [], [], [],[]
    test_seqs, test_sample,test_full,test_label = [], [], [],[]
    for one_hot_seq, sample_coverage, full_coverage,label, name in results:
        if name == "valid":
            valid_seqs.append(one_hot_seq)
            valid_sample.append(sample_coverage)
            valid_full.append(full_coverage)
            valid_label.append(label)
        elif name == "test":
            test_seqs.append(one_hot_seq)
            test_sample.append(sample_coverage)
            test_full.append(full_coverage)
            test_label.append(label)
        else:
            train_seqs.append(one_hot_seq)
            train_sample.append(sample_coverage)
            train_full.append(full_coverage)
            train_label.append(label)

    # 保存训练集
    with h5py.File("%s/train_all_max_%s.h5"%(output_dir,cell_name), "w") as train_h5:
        train_h5.create_dataset("sequence", data=np.stack(train_seqs))
        train_h5.create_dataset("noisy", data=np.stack(train_sample))
        train_h5.create_dataset("clean", data=np.stack(train_full))
        train_h5.create_dataset("label", data=np.stack(train_label))

    # 保存验证集
    with h5py.File("%s/valid_all_max_%s.h5"%(output_dir,cell_name), "w") as valid_h5:
        valid_h5.create_dataset("sequence", data=np.stack(valid_seqs))
        valid_h5.create_dataset("noisy", data=np.stack(valid_sample))
        valid_h5.create_dataset("clean", data=np.stack(valid_full))
        valid_h5.create_dataset("label", data=np.stack(valid_label))

    # 保存测试集
    with h5py.File("%s/test_all_max_%s.h5"%(output_dir,cell_name), "w") as test_h5:
        test_h5.create_dataset("sequence", data=np.stack(test_seqs))
        test_h5.create_dataset("noisy", data=np.stack(test_sample))
        test_h5.create_dataset("clean", data=np.stack(test_full))
        test_h5.create_dataset("label", data=np.stack(test_label))

if __name__ == '__main__':
    all_mseqs = main()
    num_processes = 50
    fasta_open = pysam.FastaFile(fasta_file)
    # 假设 fasta, sample_file, full_file 是全局变量或者预先定义
    def map_seq_to_targets_partial(contig):
        return map_seq_to_targets(contig, fasta_open, sample_file, full_file)
    
    with Pool(num_processes) as pool:
        # 使用 tqdm 的 update 方法实现进度条
        with tqdm(total=len(all_mseqs)) as pbar: # type: ignore
            results = []
            for result in pool.imap(map_seq_to_targets_partial, all_mseqs): # type: ignore
                results.append(result)
                pbar.update(1)  # 每完成一个任务，更新进度条
            split_sequences(results,out_dir)