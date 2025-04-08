# SUCCEED
Deep learning methods based on DNA sequences have become an important tool for parsing multilevel regulatory mechanisms in genomes. These methods significantly improve the efficiency of identifying pathogenic genetic variants in genome-wide association studies (GWAS) by integrating artificial intelligence technologies with high-throughput functional genomic data. However, it remains challenging to precisely establish the mapping relationship between DNA sequences and complex epigenetic modifications. The Enformer model developed by David R. Kelley's team achieves accurate prediction from sequences to various epigenomic features by introducing long-range interaction mechanisms. This supervised learning framework innovatively combines the local feature extraction capabilities of one-dimensional convolutional neural networks with the global sequence modeling advantages of the Transformer architecture, achieving prediction accuracy significantly superior to traditional pure convolutional models and local window models.

We systematically evaluated the reproducibility of the Enformer model and its reusability in five different genomic tasks. To verify the model's reproducibility, our model retrained based on 6,389 epigenomic profiles demonstrated stable predictive performance and fine-tuning potential on new datasets and under different resolution conditions. Additionally, by developing new representation transfer strategies, we proved that the model can not only accurately predict cell-type-specific epigenetic modifications but also effectively improve the signal-to-noise ratio of chromatin accessibility data. Notably, when predicting cell-type-specific 3D chromatin interaction maps, Enformer significantly outperforms existing benchmark methods, and the model only requires a small number of scATAC-seq cells to accurately predict cell-type-specific 3D chromatin structures. These results suggest that the Enformer model, by integrating sequence features and epigenetic information, demonstrates excellent predictive capabilities in multiple genomics tasks. Our research lays a solid foundation for building a unified computational framework for multi-omics analysis.

## Dependencies and Installation
```shell
git clone https://github.com/bioczsun/SUCCEED.git
cd SUCCEED
conda create -n succeed python=3.11.2 basenji
conda activate succeed
conda install -c bioconda -c conda-forge tqdm einops==0.7.0 ucsc-bedgraphtobigwig
# Install Pytorch 2.5.1 with CUDA 12.1
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install torchmetrics==0.11.4
# Check if PyTorch is installed successfully and can access GPU
python -c "import torch; print(torch.cuda.is_available())"
```

# 1. Train from scratch
Next, we use a simple example on a small dataset (5 targets) to demonstrate how to train the SUCCEED model.
## Data Preparation
Prepare reference genome (hg38 and hg19)
```shell
cd SUCCEED
mkdir -p reference && wget -O - https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/hg38.fa.gz | gzip -d > reference/hg38.fa
mkdir -p reference && wget -O - https://hgdownload.cse.ucsc.edu/goldenpath/hg19/bigZips/hg19.fa.gz | gzip -d > reference/hg19.fa
```
Prepare a dataset that SUCCEED can read
```shell
conda activate succeed

# We referenced basenji's approach to dataset preparation, using hg38.fa as the reference genome
python src/preprocess/basenji_data.py -s 1 -g reference/hg38_gap.bed \
    -l 131072 --local -o "training/131k-128" -p 48 -t .1 -v .1 -w 128 \
    -b reference/hg38-blacklist.v2.bed reference/hg38.fa src/preprocess/target.txt

python src/preprocess/prepare_succeed_dataset.py \
    --fasta reference/hg38.fa \
    --bed result/training/131k-128/sequences.bed \
    --num_targets 5 \
    --output_dir result/training/131k-128
```
## Pre-training
```shell
python src/training/train.py \
    --project_dir /path/to/SUCCEED \
    --output_head_size 5 \
    --td result/training/131k-128/train_all.h5 \
    --vd result/training/131k-128/valid_all.h5 \
    --name 131k \
    --seed 1401 \
    --lr 0.001 \
    --batch 8 \
    --out_dir result/training/131k-128/model
```


# 2. Fine-tuning on new dataset
Next, we use the fully pre-trained model 131k-128bp on the complete dataset (6,389 targets) to fine-tune a new dataset with different resolution (we still use the small dataset with 5 targets for testing), using 524k-512bp as an example.
## Download pre-trained model
Coming soon
## Data Preparation for 524k-512bp model
```shell
python src/preprocess/basenji_data.py -s 1 -g reference/hg38_gap.bed \
    --break 7864320 -l 524288 --local -o "training/524k-512" -p 48 -t .1 -v .1 -w 512 \
    -b reference/hg38-blacklist.v2.bed reference/hg38.fa src/preprocess/target.txt

python src/preprocess/prepare_succeed_dataset.py \
    --fasta reference/hg38.fa \
    --bed result/training/524k-512/sequences.bed \
    --num_targets 5 \
    --output_dir result/training/524k-512
```
## Fine-tuning
```shell
python src/training/train.py \
    --project_dir /path/to/SUCCEED \
    --output_head_size 5 \
    --td result/training/524k-512/train_all.h5 \
    --vd result/training/524k-512/valid_all.h5 \
    --seed 1401 \
    --lr 0.001 \
    --batch 8 \
    --name 524k_131k_weight \
    --use_pth data/model/131k_corr_weight_10.10.1_best_network.pth \
    --out_dir result/training/524k-512/model
```

# 3. Epigenomic feature prediction (EFP)
## Data Preparation
First, we need to prepare a dataset with input length of 1,048,576 (1M) and resolution of 1024bp, for pre-training the 1M-1024bp SUCCEED model
```shell
python src/preprocess/basenji_data.py -s 1 -g reference/hg38_gap.bed \
    --break 7864320 -l 1048576 --local -o "training/1m-1024" -p 48 -t .1 -v .1 -w 1024 \
    -b reference/hg38-blacklist.v2.bed reference/hg38.fa src/preprocess/target.txt

python src/preprocess/prepare_succeed_dataset.py \
    --fasta reference/hg38.fa \
    --bed result/training/1m-1024/sequences.bed \
    --num_targets 5 \
    --output_dir result/training/1m-1024
```
Next, we prepare the dataset for the EFP task
```shell
Coming soon
```
## Pre-training
```shell
python src/training/train.py \
    --project_dir /path/to/SUCCEED \
    --output_head_size 5 \
    --td result/training/1m-1024/train_all.h5 \
    --vd result/training/1m-1024/valid_all.h5 \
    --name HistonTF \
    --out_dir result/training/1m-1024/model
```
## Fine-tuning
Then, we use the pre-trained model for fine-tuning on the EFP task. Here, instead of using the model fine-tuned on the example dataset, we use a model that has been pre-trained on 5000 targets (deduplicated with prediction targets)
```shell
python src/training/train_HistonTF.py \
    --project_dir /path/to/SUCCEED \
    --use_pth data/model/HistonTF_best_network.pth \
    --seed 1401 \
    --lr 0.001 \
    --batch 8 \
    --seq_dir reference/dna_sequence \
    --train_feature_dir data/EFP/train_target.h5 \
    --valid_feature_dir data/EFP/valid_target.h5 \
    --train_contig_bed data/EFP/train_bed.npy \
    --valid_contig_bed data/EFP/valid_bed.npy \
    --atac_path data/EFP/atac \
    --atac_dict "GM12878.bigWig,HepG2.bigWig,K562.bigWig,MCF-7.bigWig" \
    --name HistonTF \
    --output_head_size 46 \
    --genome_assembly hg38 \
    --out_dir result/EFP/4-cells/model
```

Testing on the test set
```shell
python src/inference/prediction_HistonTF.py \
    --project_dir /path/to/SUCCEED \
    --use_pth data/model/HistonTF_best_network.pth \
    --model result/EFP/4-cells/model/HistonTF_best_network.pth \
    --seed 1401 \
    --batch 8 \
    --seq_dir reference/dna_sequence \
    --feature_dir data/EFP/test_target.h5 \
    --contig_bed data/EFP/test_bed.npy  \
    --atac_path data/EFP/atac \
    --atac_dict "GM12878.bigWig,HepG2.bigWig,K562.bigWig,MCF-7.bigWig" \
    --name HistonTF \
    --out_dir result/EFP/4-cells/model/csv/logs/version_0
```

## Inference on the new dataset
```shell
python src/inference/inference_HistonTF.py \
    --project_dir /path/to/SUCCEED \
    --use_pth data/model/HistonTF_best_network.pth \
    --model result/EFP/4-cells/model/HistonTF_best_network.pth \
    --seed 1401 \
    --batch 8 \
    --seq_dir reference/dna_sequence \
    --atac_path data/EFP/atac \
    --atac_dict "A549.bigWig" \
    --name A549 \
    --contig_bed data/EFP/1m_epcot_sequences.bed  \ # You can substitute the contig_bed with your own dataset
    --out_dir result/EFP/4-cells/model/csv/logs/version_0
```

# 4. 表观基因组去噪与增强
## Data Preparation
We use the data from the AtacWorks paper to demonstrate the denoising and enhancement of chromatin accessibility data.

For bulk dataset:
```shell
# Training dataset on 4 cell types
for cell_type in "CD8-10" "Bcell-13" "CD4-9" "Nkcell-11"; do
    python src/atacwork/atacwork_data.py \
        --out_dir result/denoise/bulk/$cell_type \
        --clean_file data/denoise/bulk_blood_cell_denoising_experiments/200000_reads/train_data/clean_data/$cell_type.50000000.1.cutsites.smoothed.200.bw \
        --noisy_file data/denoise/bulk_blood_cell_denoising_experiments/200000_reads/train_data/noisy_data/$cell_type.200000.2.cutsites.smoothed.200.bw \
        --peak_file data/denoise/bulk_blood_cell_denoising_experiments/200000_reads/train_data/clean_data/$cell_type.50000000.1.cutsites.smoothed.200.3.narrowPeak \
        --fasta_file reference/hg19.fa \
        --gaps_file reference/hg19_gaps.bed \
        --name "$cell_type" \
        --restart
done

# Test dataset on erythroid
python src/atacwork/atacwork_data.py \
    --out_dir result/denoise/bulk/Erythro-15 \
    --clean_file data/denoise/bulk_blood_cell_denoising_experiments/200000_reads/test_data/erythroid_test_data/clean_data/Erythro-15.50000000.1.cutsites.smoothed.200.bw \
    --noisy_file data/denoise/bulk_blood_cell_denoising_experiments/200000_reads/test_data/erythroid_test_data/noisy_data/Erythro-15.200000.2.cutsites.smoothed.200.bw \
    --peak_file data/denoise/bulk_blood_cell_denoising_experiments/200000_reads/test_data/erythroid_test_data/clean_data/Erythro-15.50000000.1.cutsites.smoothed.200.3.narrowPeak \
    --fasta_file reference/hg19.fa \
    --gaps_file reference/hg19_gaps.bed \
    --name "Erythro-15" \
    --restart

# 制作peaks label bigwig文件
tail -n +2 data/denoise/bulk_blood_cell_denoising_experiments/200000_reads/test_data/erythroid_test_data/clean_data/Erythro-15.50000000.1.cutsites.smoothed.200.3.narrowPeak | awk -F'\t' '{print $1 "\t" $2 "\t" $3 "\t" 1}' > data/denoise/bulk_blood_cell_denoising_experiments/200000_reads/test_data/erythroid_test_data/clean_data/peaks.bed

bedGraphToBigWig data/denoise/bulk_blood_cell_denoising_experiments/200000_reads/test_data/erythroid_test_data/clean_data/peaks.bed reference/hg19.chrom.sizes data/denoise/bulk_blood_cell_denoising_experiments/200000_reads/test_data/erythroid_test_data/clean_data/peaks_label.bw
```

## Fine-tuning
For bulk dataset:
```shell
# Training on 4 cell types (Optional)
python src/atacwork/train_atacwork.py \
    --project_dir /path/to/SUCCEED \
    --name ATACwork \
    --use_pth data/model/131k_corr_weight_10.10.1_best_network.pth \
    --batch 32 \
    --lr 0.0001 \
    --dataset_dir result/denoise/bulk \
    --outpath result/denoise/bulk/train_model/ \
    --cell_types "CD8-10" "Bcell-13" "CD4-9" "Nkcell-11"

# Testing on erythroid
python src/atacwork/test_atacwork.py \
    --project_dir /home/hezj/projects/SUCCEED \
    --name test_result \
    --use_pth data/model/131k_corr_weight_10.10.1_best_network.pth \
    --model data/model/ATACwork_200000_best_network.pth \
    --batch 32 \
    --dataset_dir result/denoise/bulk \
    --outpath result/denoise/bulk/test_results/ \
    --cell_types "Erythro-15"
```