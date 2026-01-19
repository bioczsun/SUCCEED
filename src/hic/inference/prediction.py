import argparse
import sys
from pathlib import Path

# Ensure `src/` is on sys.path so `import hic...` works when running this file directly.
# File location: <repo>/src/hic/inference/prediction.py
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import hic.inference.utils.inference_utils as infer
from hic.inference.utils import plot_utils

def main():
  
  parser = argparse.ArgumentParser(description='C.Origami Prediction Module.')
  parser.add_argument('--out', dest='output_path', default='outputs',
          help='output path for storing results (default: %(default)s)')
  parser.add_argument('--celltype', dest='celltype', 
                        help='Sample cell type for prediction, used for output separation')
  parser.add_argument('--chr', dest='chr_name', 
                        help='Chromosome for prediction', required=True)
  parser.add_argument('--start', dest='start', type=int,
                        help='Starting point for prediction (width is 2097152 bp which is the input window size)', required=True)
  parser.add_argument('--model', dest='model_path', 
                        help='Path to the model checkpoint', required=True)
  parser.add_argument('--seq', dest='seq_path', 
                        help='Path to the folder where the sequence .fa.gz files are stored', required=True)
  parser.add_argument('--ctcf', dest='ctcf_path',
                        default='',
                        help='Path to the folder where the CTCF ChIP-seq .bw files are stored (optional when using ATAC-only).')
  parser.add_argument('--atac', dest='atac_path', 
                        help='Path to the folder where the ATAC-seq .bw files are stored', required=True)
  parser.add_argument('--use-atac-only', dest='use_atac_only',
                        action='store_true',
                        help='Run inference with DNA sequence + ATAC only (no CTCF). Requires an ATAC-only trained checkpoint.')

  # Optional SUCCEED sequence encoder (used when succeed=True in ConvTransModel)
  parser.add_argument('--use-succeed-seq-model', dest='use_succeed_seq_model',
                        action='store_true',
                        help='Use a pretrained SUCCEED sequence encoder inside the Hi-C model (requires --succeed-ckpt).')
  parser.add_argument('--project-dir', dest='project_dir',
                        default=str(_REPO_ROOT),
                        help='Project root directory (should contain `src/`).')
  parser.add_argument('--succeed-ckpt', dest='succeed_ckpt',
                        default='',
                        help='Path to pretrained SUCCEED checkpoint (.pth) used as sequence encoder.')
  parser.add_argument('--succeed-output-heads', dest='succeed_output_heads', default=6389, type=int,
                        help='Output head size used when instantiating SUCCEED encoder (human head).')
  parser.add_argument('--succeed-stem-windows', dest='succeed_stem_windows', default=32, type=int,
                        help='SUCCEED config: stem_windows.')
  parser.add_argument('--succeed-pool-windows', dest='succeed_pool_windows', default=4, type=int,
                        help='SUCCEED config: pool_windows.')
  parser.add_argument('--succeed-max-seq-len', dest='succeed_max_seq_len', default=256, type=int,
                        help='SUCCEED config: max_seq_len.')
  parser.add_argument('--succeed-target-seq-len', dest='succeed_target_seq_len', default=256, type=int,
                        help='SUCCEED config: target_seq_len.')

  args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
  if (not args.use_atac_only) and (not args.ctcf_path):
      raise ValueError("`--ctcf` is required unless `--use-atac-only` is set.")
  single_prediction(args.output_path, args.celltype, 
                      args.chr_name, args.start,
                      args.model_path, 
                      args.seq_path, args.ctcf_path, args.atac_path,
                      use_atac_only=args.use_atac_only,
                      use_succeed_seq_model=args.use_succeed_seq_model,
                      project_dir=args.project_dir,
                      succeed_ckpt=args.succeed_ckpt)

def single_prediction(
    output_path,
    celltype,
    chr_name,
    start,
    model_path,
    seq_path,
    ctcf_path,
    atac_path,
    *,
    use_atac_only=False,
    use_succeed_seq_model=False,
    project_dir="",
    succeed_ckpt=""
):
    ctcf_path = ctcf_path if (ctcf_path and (not use_atac_only)) else None
    seq_region, ctcf_region, atac_region = infer.load_region(chr_name, 
            start, seq_path, ctcf_path, atac_path)
    pred = infer.prediction(seq_region, ctcf_region, atac_region, 
                                   model_path,
                                   use_atac_only=use_atac_only,
                                   use_succeed_seq_model=use_succeed_seq_model,
                                   project_dir=project_dir,
                                   succeed_ckpt=succeed_ckpt)
    plot = plot_utils.MatrixPlot(output_path, pred, 'prediction', celltype, 
                                 chr_name, start)
    plot.plot()

if __name__ == '__main__':
    main()
