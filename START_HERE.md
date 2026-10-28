
## Execute the ensemble experiments
Open a terminal inside 01_Ensemble_Ablations, then:

```bash
pip install -r requirements.txt
python -m unittest -v test_smoke test_revision
python run_repeated_experiments.py --dataset Kvasir --root /data/kvasir --manifest manifests/Kvasir_fixed_split.csv --out results/Kvasir
python run_repeated_experiments.py --dataset GERD --root /data/gerd --manifest /data/GERD_fixed_split.csv --out results/GERD
```


## Status

See RERUN_STATUS.md for the scope of this update.
