#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <BAM>"
    echo "Example: $0 /path/to/IMR-90_ATAC.bam"
    exit 1
fi

source activate succeed

input_bam=$1

if [ ! -f "$input_bam" ]; then
    echo "Error: $input_bam does not exist"
    exit 1
fi

dir=$(dirname "$input_bam")
name=$(basename "$input_bam" .bam)
output_prefix="$dir/$name"

temp_dir=$(mktemp -d)

echo "temp_dir: $temp_dir"

samtools view -h -f 2 -q 20 "$input_bam" | grep -v chrM | samtools sort -O bam -@ 40 -o "$temp_dir/last.bam"

bedtools genomecov -ibam "$temp_dir/last.bam" -bg > "$temp_dir/coverage.bedGraph"

count=$(samtools view -c -F 0x904 "$temp_dir/last.bam" -@ 40)

sort -k1,1 -k2,2n "$temp_dir/coverage.bedGraph" > "$temp_dir/sorted.bedGraph"
grep -E '^chr([1-9]|1[0-9]|2[0-2]|X)[[:space:]]' "$temp_dir/sorted.bedGraph" > "$temp_dir/filtered.bedGraph"

awk -v total=$count 'BEGIN{OFS="\t"} { $4 = sprintf("%.3f", ($4 * 1000000 / total)); print }' "$temp_dir/filtered.bedGraph" > "$temp_dir/cpm.bedGraph"

bedGraphToBigWig "$temp_dir/cpm.bedGraph" reference/hg38.chrom.size "$output_prefix.cpm.bw"
cp "$output_prefix.cpm.bw" "$dir/genomic_features/atac.bw"

rm -rf "$temp_dir"