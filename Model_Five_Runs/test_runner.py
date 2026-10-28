import json
import tempfile
import unittest
from pathlib import Path
import run_five_runs as runner

class RunnerTests(unittest.TestCase):
    def test_original_manifest_relocation_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            df=runner.prepare_manifest(Path(__file__).with_name('kvasir_fixed_manifest.csv'),root,root/'manifest.csv',False)
            self.assertEqual(len(df),6000)
            self.assertEqual((df.split=='test').sum(),600)
            self.assertEqual(df.path.nunique(),6000)
    def test_summary_sample_sd(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            for seed,v in zip(runner.SEEDS,[.8,.85,.9,.95,1.]):
                folder=out/f'seed_{seed}';folder.mkdir()
                (folder/'test_metrics.json').write_text(json.dumps({'seed':seed,'test_images':600,**{m:v for m in runner.METRICS}}))
            summary=runner.summarize(out,runner.SEEDS)
            self.assertAlmostEqual(summary[0]['mean'],.9)
            self.assertAlmostEqual(summary[0]['sample_sd'],.07905694150420949)
            self.assertTrue((out/'per_seed_metrics.csv').is_file())
    def test_missing_run_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):runner.summarize(Path(d),runner.SEEDS)

if __name__=='__main__':unittest.main()
