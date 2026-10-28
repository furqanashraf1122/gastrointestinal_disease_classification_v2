# Corrected reproducibility package

## Evidence status

## Configured protocol
- Six manuscript backbones with ImageNet initialization required by the training CLI; failed downloads stop execution. Extraction initializes without pretrained weights only to load the saved trained checkpoint.
- Fixed training hyperparameters: Adam, learning rate 0.001, 100 epochs, weight decay 0.0001. No random search is performed. 
- Each CNN independently projects to 512 dimensions. ANOVA selection retains exactly 256 dimensions PER BRANCH for Kvasir and 384 PER BRANCH for GERD. These are preset counts, not validation-tuned counts; selected indices are learned on training data only. Do not call these totals across all six branches.
- Shared fixed weights actively used in A0/A1/A3/A4/A5 for both datasets: GoogLeNet .3, ResNet-50 .1, Inception-v3 .2, Xception .1, DenseNet-201 .2, SqueezeNet .1. No validation weight optimization in this protocol.  A2 uses 1/6 each.
- Projection/checkpoint selection uses validation accuracy. Train-only ANOVA ranking and LR fitting never use test labels. Weighted sampling balances training; augmentation is stochastic on the fly, not six saved copies of each image.

## Corrected Table 12 configurations
A0: full pipeline, ANOVA selection, shared fixed weights.
A1: A0 using networks trained without augmentation.
A2: A0 with equal probability weights.
A3: A0 retaining all 512 projected dimensions per branch.
A4: A0 with variance-based feature selection instead of ANOVA-based feature selection, retaining the same dataset-specific count.  Restoring ANOVA selection to A3 would reproduce A0 by definition. Different procedures may still yield identical metrics; do not force a numerical difference.
A5: A0 without learned projections; train-only scaled raw CNN features undergo the same preset selection count.

## Table 10
Averaging and concatenation are separate feature-level experiments with their own ANOVA selector and LR classifier. The preset count is applied to the fused representation (256/384), whereas A0 selects that count per branch and combines probabilities. Both Table 10 methods export independent probabilities and selected indices. A matching rounded score is possible and is not artificially changed.

## Run instructions
Install requirements into an environment with suitable PyTorch/CUDA support:

```bash
pip install -r requirements.txt
python -m unittest -v test_smoke test_revision
python run_repeated_experiments.py --dataset Kvasir --root /data/kvasir --manifest manifests/Kvasir_fixed_split.csv --out results/Kvasir
python run_repeated_experiments.py --dataset GERD --root /data/gerd --manifest /data/GERD_fixed_split.csv --out results/GERD
```

The Kvasir manifest preserves the supplied image assignments using portable class/filename paths. GERD images and the actual GERD split were not supplied: use the original GERD manifest, not a newly invented split. The generic split command partitions all available images and does not automatically reproduce the selected 6000-image Kvasir subset. DataLoader workers default to zero for portable operation.

For a single corrected run, train/extract augmented and non-augmented branches using the corresponding commands in run_repeated_experiments.py, then run:

```bash
python gi_ensemble.py ensemble --dataset Kvasir --features /runs/features --no-aug-features /runs/features_no_aug --out /runs/results
```

## Outputs and manuscript actions
Each seed exports ablation_metrics.json, table10_fusion_metrics.json, run_protocol.json, ensemble_weights.json, selected feature indices for every variant, test probability arrays, A0 test_predictions.csv, confusion matrices and classification reports, trained projection/scaler/selector/LR state in ensemble.pt, CNN checkpoints and validation histories. The runner retains manifests/configuration, commands/logs and environment/source hashes.

After all five runs, repeated_run_summary.json, per_seed_metrics.csv and mean_sample_sd.csv contain BOTH Table 10 and Table 12 results. Metrics are fractions; percentage values multiply by 100. SD is sample SD (ddof=1). Five training seeds share one fixed test set; they are not five independent test populations.

The author confirms that rerun results have been incorporated into the revised manuscript. Other methodology inconsistencies and the missing raw reproduction records remain separate issues. External Table 11 baseline results are not reproduced by this package. See ../RERUN_STATUS.md.
