#!/usr/bin/env python3
"""Endoscopy CNN training, feature fusion, evaluation and ablation reference code."""
import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")

# Ordered exactly as the final ensemble in manuscript Table 4.
MODEL_SPECS = {
    "googlenet": {"display_name": "GoogLeNet", "input_size": 224, "feature_dim": 1024},
    "resnet50": {"display_name": "ResNet-50", "input_size": 224, "feature_dim": 2048},
    "inception_v3": {"display_name": "Inception-v3", "input_size": 299, "feature_dim": 2048},
    "xception": {"display_name": "Xception", "input_size": 299, "feature_dim": 2048},
    "densenet201": {"display_name": "DenseNet-201", "input_size": 224, "feature_dim": 1920},
    "squeezenet1_1": {"display_name": "SqueezeNet", "input_size": 227, "feature_dim": 512},
}

def config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg["backbones"] != list(MODEL_SPECS):
        raise ValueError("Use the six updated models in the Table 4 order")
    for name, spec in MODEL_SPECS.items():
        if cfg["input_sizes"][name] != spec["input_size"]:
            raise ValueError(f"Input size for {name} must be {spec['input_size']}")
    candidates = list(cfg["fusion"]["fixed_feature_counts"].values())
    if not candidates or any(k < 1 or k > cfg["fusion"]["projection_dim"] for k in candidates):
        raise ValueError("Feature counts must fit within the projection dimension")
    weights = cfg["fusion"]["reported_weights"]
    if set(weights) != set(cfg["backbones"]) or any(v < 0 for v in weights.values()) or not np.isclose(sum(weights.values()), 1):
        raise ValueError("Shared weights must cover all backbones and sum to one")
    return cfg

def make_split(args):
    root = Path(args.root)
    rows = [
        {"path": p.relative_to(root).as_posix(), "label": p.parent.name}
        for p in sorted(root.glob("*/*"))
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    if not rows:
        raise ValueError("No class-folder image files found")
    df = pd.DataFrame(rows)
    indices = np.arange(len(df))
    train_val, test = train_test_split(indices, test_size=0.1, random_state=args.seed,
                                       stratify=df.label)
    train, val = train_test_split(train_val, test_size=2 / 9, random_state=args.seed,
                                  stratify=df.label.iloc[train_val])
    df["split"] = ""
    df.loc[train, "split"] = "train"
    df.loc[val, "split"] = "val"
    df.loc[test, "split"] = "test"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby(["split", "label"]).size().to_string())

def read_manifest(manifest, root):
    df = pd.read_csv(manifest, dtype=str)
    if "class_name" in df.columns:
        df["label"] = df["class_name"]
    if not {"path", "label", "split"}.issubset(df.columns):
        raise ValueError("Manifest must contain path,label,split")
    if "class_name" in df.columns:
        df["label"] = df["class_name"]
    df["split"] = df["split"].replace({"validation": "val"})
    if set(df.split) != {"train", "val", "test"}:
        raise ValueError("Expected train, val and test splits")
    if df.path.duplicated().any():
        raise ValueError("The same image path appears in multiple splits")
    if df[["path", "label", "split"]].isnull().any().any():
        raise ValueError("Missing manifest values")
    root = Path(root).resolve()
    for rel in df.path:
        target = (root / rel).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError(f"Image path missing or outside root: {rel}")
    labels = sorted(df.label.unique())
    for split in ("train", "val", "test"):
        if set(df[df.split == split].label) != set(labels):
            raise ValueError(f"Every class must occur in {split}")
    return df, {name: i for i, name in enumerate(labels)}

def backbone(name, nclasses, pretrained):
    import torch
    from torch import nn
    from torchvision import models
    if name == "xception":
        import timm
        feature_net = timm.create_model("xception", pretrained=pretrained,
                                        num_classes=0, global_pool="avg")
        dim = feature_net.num_features
    else:
        weights = "DEFAULT" if pretrained else None
        if name == "googlenet":
            feature_net = models.googlenet(
                weights=weights, aux_logits=pretrained, transform_input=False, init_weights=False)
            # Load pretrained auxiliary parameters first, then remove the unused heads.
            feature_net.aux_logits = False
            feature_net.aux1 = None
            feature_net.aux2 = None
            dim = feature_net.fc.in_features
            feature_net.fc = nn.Identity()
        elif name == "resnet50":
            feature_net = models.resnet50(weights=weights)
            dim = feature_net.fc.in_features
            feature_net.fc = nn.Identity()
        elif name == "inception_v3":
            feature_net = models.inception_v3(
                weights=weights, aux_logits=pretrained, transform_input=False, init_weights=False)
            feature_net.aux_logits = False
            feature_net.AuxLogits = None
            dim = feature_net.fc.in_features
            feature_net.fc = nn.Identity()
        elif name == "densenet201":
            feature_net = models.densenet201(weights=weights)
            dim = feature_net.classifier.in_features
            feature_net.classifier = nn.Identity()
        elif name == "squeezenet1_1":
            feature_net = models.squeezenet1_1(weights=weights).features
            dim = 512
        else:
            raise ValueError(f"Unknown backbone: {name}")

    if dim != MODEL_SPECS[name]["feature_dim"]:
        raise ValueError(f"Unexpected feature dimension for {name}: {dim}")

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = feature_net
            self.head = nn.Linear(dim, nclasses)

        def forward(self, x):
            z = self.features(x)
            if isinstance(z, tuple):
                z = z[0]
            if z.ndim > 2:
                z = torch.mean(z, dim=tuple(range(2, z.ndim)))
            return self.head(z), z

    return Network()

def transforms_for(size, augmented):
    from torchvision import transforms as T
    base = [T.Resize((size, size))]
    if augmented:
        base += [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.15),
        ]
    return T.Compose(base + [
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def loader(df, root, labels, size, split, batch, workers, augmented=False, balance_training=False):
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    class Images(Dataset):
        def __init__(self):
            self.rows = df[df.split == split].reset_index(drop=True)
            self.tfm = transforms_for(size, augmented)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row = self.rows.iloc[idx]
            with Image.open(Path(root) / row.path) as image:
                x = self.tfm(image.convert("RGB"))
            return x, labels[row.label], row.path

    images = Images()
    sampler = None
    if balance_training and split == "train":
        frequencies = images.rows.label.value_counts()
        weights = [1.0 / frequencies[label] for label in images.rows.label]
        sampler = WeightedRandomSampler(weights, num_samples=len(images), replacement=True)
    return DataLoader(images, batch_size=batch, shuffle=False, sampler=sampler,
                      num_workers=workers, pin_memory=True)

def train_all(args):
    import torch
    from torch import nn
    cfg = config(args.config)
    run_seed = cfg["seed"] if args.run_seed is None else args.run_seed
    seed_all(run_seed)
    df, labels = read_manifest(args.manifest, args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opts = cfg["train"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "experiment.json", {"config": cfg, "run_seed": run_seed, "labels": labels,
               "manifest": str(Path(args.manifest).resolve()),
               "augmentation": not args.no_augment, "pretrained": bool(args.pretrained),
               "training_procedure": "fixed_hyperparameters_no_random_search"})
    for name in cfg["backbones"]:
        size = cfg["input_sizes"][name]
        model = backbone(name, len(labels), args.pretrained).to(device)
        train_dl = loader(df, args.root, labels, size, "train", opts["batch_size"],
                          opts["num_workers"], augmented=not args.no_augment, balance_training=True)
        val_dl = loader(df, args.root, labels, size, "val", opts["batch_size"],
                        opts["num_workers"])
        optimizers = {"adam": torch.optim.Adam, "sgdm": torch.optim.SGD,
                      "rmsprop": torch.optim.RMSprop}
        method = opts["optimizer"].lower()
        kw = {"momentum": 0.9} if method == "sgdm" else {}
        optim = optimizers[method](model.parameters(), lr=opts["learning_rate"],
                                   weight_decay=opts["weight_decay"], **kw)
        best = -1.0
        history = []
        for epoch in range(opts["epochs"]):
            model.train()
            for x, y, _ in train_dl:
                x, y = x.to(device), y.to(device)
                optim.zero_grad(set_to_none=True)
                logits, _ = model(x)
                loss = nn.functional.cross_entropy(logits, y)
                loss.backward()
                optim.step()
            model.eval()
            actual, predicted = [], []
            with torch.no_grad():
                for x, y, _ in val_dl:
                    logits, _ = model(x.to(device))
                    actual.extend(y.tolist())
                    predicted.extend(logits.argmax(1).cpu().tolist())
            acc = accuracy_score(actual, predicted)
            print(f"{name} epoch={epoch + 1} val_accuracy={acc:.4f}", flush=True)
            history.append({"epoch": epoch + 1, "validation_accuracy": float(acc)})
            pd.DataFrame(history).to_csv(out / f"{name}_training_history.csv", index=False)
            if acc > best:
                best = acc
                torch.save({"state": model.state_dict(), "labels": labels,
                            "backbone": name, "config": cfg}, out / f"{name}.pt")

def extract_all(args):
    import torch
    cfg = config(args.config)
    run_seed = cfg["seed"] if args.run_seed is None else args.run_seed
    seed_all(run_seed)
    df, labels = read_manifest(args.manifest, args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in cfg["backbones"]:
        checkpoint = torch.load(Path(args.models) / f"{name}.pt", map_location="cpu",
                                weights_only=False)
        if checkpoint["labels"] != labels:
            raise ValueError(f"Checkpoint label mapping differs for {name}")
        model = backbone(name, len(labels), pretrained=False).to(device)
        model.load_state_dict(checkpoint["state"])
        model.eval()
        for split in ("train", "val", "test"):
            dl = loader(df, args.root, labels, cfg["input_sizes"][name], split,
                        cfg["train"]["batch_size"], cfg["train"]["num_workers"])
            feats, probs, targets, paths = [], [], [], []
            with torch.no_grad():
                for x, y, path in dl:
                    logits, z = model(x.to(device))
                    feats.append(z.cpu().numpy())
                    probs.append(logits.softmax(1).cpu().numpy())
                    targets.extend(y.tolist())
                    paths.extend(path)
            np.savez_compressed(out / f"{name}_{split}.npz",
                                features=np.concatenate(feats),
                                probabilities=np.concatenate(probs),
                                y=np.array(targets), paths=np.array(paths))
    save_json(out / "metadata.json", {"config": cfg, "run_seed": run_seed, "labels": labels,
                                       "manifest": str(Path(args.manifest).resolve())})

def load_features(folder, names):
    result = {}
    for split in ("train", "val", "test"):
        per_model = [np.load(Path(folder) / f"{name}_{split}.npz") for name in names]
        first_paths = per_model[0]["paths"]
        first_y = per_model[0]["y"]
        for x in per_model[1:]:
            if not np.array_equal(first_paths, x["paths"]) or not np.array_equal(first_y, x["y"]):
                raise ValueError("Model features do not align by image and label")
        result[split] = {
            "X": [x["features"].astype(np.float32) for x in per_model],
            "p": [x["probabilities"].astype(np.float64) for x in per_model],
            "y": first_y, "paths": first_paths,
        }
        for x in per_model:
            x.close()
    return result

def metrics(y, probs):
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
    }

def full_prob(clf, X, nclasses):
    result = np.zeros((len(X), nclasses), dtype=np.float64)
    result[:, clf.classes_.astype(int)] = clf.predict_proba(X)
    return result

def train_projections(splits, cfg):
    """Learn a separate supervised 512D projection for each CNN branch.

    Each projection has its own training-only classification head. No feature
    averaging, concatenation, or cross-branch loss is used to train projections.
    """
    import torch
    from torch import nn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scalers = [StandardScaler().fit(X) for X in splits["train"]["X"]]
    tensors = {}
    for part in ("train", "val", "test"):
        tensors[part] = [
            torch.tensor(s.transform(X), dtype=torch.float32, device=device)
            for s, X in zip(scalers, splits[part]["X"])
        ]
    fcfg = cfg["fusion"]
    D = fcfg["projection_dim"]
    projectors = nn.ModuleList([nn.Linear(X.shape[1], D) for X in tensors["train"]]).to(device)
    heads = nn.ModuleList([nn.Linear(D, len(set(splits["train"]["y"])))
                           for _ in tensors["train"]]).to(device)
    optimizers = [torch.optim.Adam(list(P.parameters()) + list(H.parameters()),
                                   lr=fcfg["projection_lr"])
                  for P, H in zip(projectors, heads)]
    y = torch.tensor(splits["train"]["y"], dtype=torch.long, device=device)
    for P, H, optim, X_train, X_val in zip(
            projectors, heads, optimizers, tensors["train"], tensors["val"]):
        best, best_state = -1.0, None
        for _ in range(fcfg["projection_epochs"]):
            P.train()
            H.train()
            order = torch.randperm(len(y), device=device)
            for ids in order.split(256):
                loss = nn.functional.cross_entropy(H(P(X_train[ids])), y[ids])
                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()
            P.eval()
            H.eval()
            with torch.no_grad():
                score = accuracy_score(
                    splits["val"]["y"], H(P(X_val)).argmax(1).cpu().numpy())
            if score > best:
                best = score
                best_state = ({k: v.detach().cpu().clone() for k, v in P.state_dict().items()},
                              {k: v.detach().cpu().clone() for k, v in H.state_dict().items()})
        P.load_state_dict(best_state[0])
        H.load_state_dict(best_state[1])
    output = {}
    with torch.no_grad():
        for part in tensors:
            output[part] = [P(X).cpu().numpy() for P, X in zip(projectors, tensors[part])]
    return output, scalers, projectors

def evaluate_feature_fusion(projected, splits, feature_counts, out=None):
    """Generate feature-averaging/concatenation results for Table 10.

    Feature selection and candidate-count selection use training/validation
    only. The held-out test split is used once for final metrics.
    """
    strategies = {
        "feature_averaging": {
            split: np.mean(projected[split], axis=0) for split in projected
        },
        "feature_concatenation": {
            split: np.concatenate(projected[split], axis=1) for split in projected
        },
    }
    output = {}
    y_train, y_val, y_test = (splits[s]["y"] for s in ("train", "val", "test"))
    nclasses = len(np.unique(y_train))
    for name, matrices in strategies.items():
        X_train, X_val, X_test = (matrices[s] for s in ("train", "val", "test"))
        best_score, best_k = -1.0, None
        candidates = sorted(set(min(int(k), X_train.shape[1]) for k in feature_counts))
        for k in candidates:
            selector = SelectKBest(f_classif, k=k).fit(X_train, y_train)
            clf = fit_classifier(selector.transform(X_train), y_train)
            val_probs = full_prob(clf, selector.transform(X_val), nclasses)
            score = accuracy_score(y_val, val_probs.argmax(1))
            if score > best_score:
                best_score, best_k = score, k
        selector = SelectKBest(f_classif, k=best_k).fit(X_train, y_train)
        clf = fit_classifier(selector.transform(X_train), y_train)
        test_probs = full_prob(clf, selector.transform(X_test), nclasses)
        if out is not None:
            np.savez_compressed(Path(out) / f"{name}_test_probabilities.npz", probabilities=test_probs, y=y_test, paths=splits["test"]["paths"])
            save_json(Path(out) / f"{name}_selected_features.json", selector.get_support(indices=True).tolist())
        output[name] = {
            "selected_features": int(best_k),
            "validation_accuracy": float(best_score),
            **metrics(y_test, test_probs),
        }
    return output

def fit_classifier(X, y):
    return LogisticRegression(max_iter=1500, class_weight="balanced").fit(X, y)

def best_model_probs(splits):
    val_scores = [accuracy_score(splits["val"]["y"], p.argmax(1))
                  for p in splits["val"]["p"]]
    idx = int(np.argmax(val_scores))
    return idx, splits["test"]["p"][idx]

def variance_scores(X, y):
    """Training-only unsupervised ranking for the A4 control."""
    return np.var(X, axis=0)

def fit_branch_models(train_branches, val_branches, test_branches, y_train, y_val,
                     feature_counts, nclasses, select_features=True):
    """Fit train-only branch selectors at the configured dataset count and LR heads."""
    models, selected_counts, val_probabilities, test_probabilities = [], [], [], []
    for X_train, X_val, X_test in zip(train_branches, val_branches, test_branches):
        if select_features:
            if any(int(k) > X_train.shape[1] for k in feature_counts):
                raise ValueError("Requested fixed feature count exceeds branch width")
            candidates = sorted(set(int(k) for k in feature_counts))
            best_score, selector = -1.0, None
            for k in candidates:
                candidate = SelectKBest(variance_scores if select_features == "variance" else f_classif, k=k).fit(X_train, y_train)
                clf = fit_classifier(candidate.transform(X_train), y_train)
                probs = full_prob(clf, candidate.transform(X_val), nclasses)
                score = accuracy_score(y_val, probs.argmax(1))
                if score > best_score:
                    best_score, selector = score, candidate
        else:
            selector = None
        train_selected = selector.transform(X_train) if selector else X_train
        val_selected = selector.transform(X_val) if selector else X_val
        test_selected = selector.transform(X_test) if selector else X_test
        clf = fit_classifier(train_selected, y_train)
        models.append((selector, clf))
        selected_counts.append(int(train_selected.shape[1]))
        val_probabilities.append(full_prob(clf, val_selected, nclasses))
        test_probabilities.append(full_prob(clf, test_selected, nclasses))
    return (models, selected_counts, np.stack(val_probabilities),
            np.stack(test_probabilities))

def branch_ensemble(train_branches, val_branches, test_branches, y_train, y_val,
                    feature_counts, nclasses, weight_step, select_features=True,
                    fixed_weights=None):
    models, counts, val_probs, test_probs = fit_branch_models(
        train_branches, val_branches, test_branches, y_train, y_val,
        feature_counts, nclasses, select_features)
    if fixed_weights is None:
        weights, val_score = select_weights(val_probs, y_val, weight_step)
        source = "validation_grid_search"
    else:
        weights = np.asarray(fixed_weights, dtype=float)
        if len(weights) != len(val_probs) or np.any(weights < 0) or not np.isclose(weights.sum(), 1):
            raise ValueError("Fixed ensemble weights must be nonnegative and sum to one")
        val_score = accuracy_score(y_val, np.tensordot(weights, val_probs, axes=(0, 0)).argmax(1))
        source = "fixed_config_weights"
    return models, counts, weights, val_score, np.tensordot(weights, test_probs, axes=(0, 0)), source

def select_weights(val_probs, y_val, step):
    """Joint simplex-grid search using validation predictions only."""
    if step <= 0 or step > 1:
        raise ValueError("weight_step must be in (0, 1]")
    units = round(1 / step)
    if not np.isclose(units * step, 1):
        raise ValueError("weight_step must divide 1 exactly")
    best, weights = -1.0, None
    for cuts in itertools.combinations_with_replacement(range(len(val_probs)), units):
        w = np.bincount(cuts, minlength=len(val_probs)) / units
        pred = np.tensordot(w, val_probs, axes=(0, 0)).argmax(1)
        score = accuracy_score(y_val, pred)
        if score > best:
            best, weights = score, w
    return weights, best

def run_ensemble(args):
    import torch
    cfg = config(args.config)
    run_seed = cfg["seed"] if args.run_seed is None else args.run_seed
    seed_all(run_seed)
    names = cfg["backbones"]
    cfg["fusion"]["feature_counts"] = [cfg["fusion"]["fixed_feature_counts"][args.dataset]]
    shared_weights = [cfg["fusion"]["reported_weights"][name] for name in names]
    splits = load_features(args.features, names)
    nclasses = len(set(splits["train"]["y"]))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    projected, scalers, projectors = train_projections(splits, cfg)
    table10 = evaluate_feature_fusion(projected, splits, cfg["fusion"]["feature_counts"], out=out)
    save_json(out / "table10_fusion_metrics.json", {"run_seed": run_seed, **table10})
    y_train, y_val, y_test = (splits[s]["y"] for s in ("train", "val", "test"))
    fc = cfg["fusion"]["feature_counts"]
    proposed = branch_ensemble(projected["train"], projected["val"], projected["test"],
                               y_train, y_val, fc, nclasses,
                               cfg["fusion"]["weight_step"], fixed_weights=shared_weights)
    models, counts, weights, val_score, probabilities, weight_source = proposed
    results["A0"] = {
        "configuration": "Complete proposed pipeline",
        "selected_features_per_branch": dict(zip(names, counts)),
        "weights": dict(zip(names, weights.tolist())),
        "weight_source": weight_source,
        "validation_accuracy": float(val_score),
        **metrics(y_test, probabilities),
    }

    # Table 12 A1: rerun the complete pipeline using CNN features extracted
    # from models trained without training-data augmentation.
    if args.no_aug_features:
        no_aug = load_features(args.no_aug_features, names)
        for part in ("train", "val", "test"):
            if not np.array_equal(no_aug[part]["paths"], splits[part]["paths"]):
                raise ValueError("No-augmentation run uses a different image split/order")
        seed_all(run_seed)
        no_aug_projected, _, _ = train_projections(no_aug, cfg)
        a1 = branch_ensemble(no_aug_projected["train"], no_aug_projected["val"],
                             no_aug_projected["test"], y_train, y_val, fc, nclasses,
                             cfg["fusion"]["weight_step"], fixed_weights=shared_weights)
        results["A1"] = {"configuration": "A0 without training-data augmentation",
                         "selected_features_per_branch": dict(zip(names, a1[1])),
                         "weights": dict(zip(names, a1[2].tolist())),
                         "weight_source": a1[5], "validation_accuracy": float(a1[3]),
                         **metrics(y_test, a1[4])}
    else:
        results["A1"] = {"configuration": "A0 without training-data augmentation",
                         "status": "not_run; provide --no-aug-features"}

    # Table 12 A2: same branch models and selectors as A0, with equal weights.
    equal = np.full(len(names), 1.0 / len(names))
    a2 = branch_ensemble(projected["train"], projected["val"], projected["test"],
                         y_train, y_val, fc, nclasses, cfg["fusion"]["weight_step"],
                         fixed_weights=equal)
    results["A2"] = {"configuration": "A0 with equal probability weights",
                     "selected_features_per_branch": dict(zip(names, a2[1])),
                     "weights": dict(zip(names, a2[2].tolist())),
                     "weight_source": a2[5], "validation_accuracy": float(a2[3]),
                     **metrics(y_test, a2[4])}

    # Table 12 A3: remove branch-specific feature selection; preserve the
    # projections, six classifiers, and shared fixed probability fusion.
    a3 = branch_ensemble(projected["train"], projected["val"], projected["test"],
                         y_train, y_val, fc, nclasses, cfg["fusion"]["weight_step"],
                         select_features=False, fixed_weights=shared_weights)
    results["A3"] = {"configuration": "A0 without branch-specific feature selection",
                     "selected_features_per_branch": dict(zip(names, a3[1])),
                     "weights": dict(zip(names, a3[2].tolist())),
                     "weight_source": a3[5], "validation_accuracy": float(a3[3]),
                     **metrics(y_test, a3[4])}

    # A4 is a distinct control: train-only variance ranking replaces ANOVA.
    a4 = branch_ensemble(projected["train"], projected["val"], projected["test"],
                         y_train, y_val, fc, nclasses, cfg["fusion"]["weight_step"],
                         select_features="variance", fixed_weights=shared_weights)
    results["A4"] = {"configuration": "A0 with variance ranking instead of ANOVA feature selection",
                     "selected_features_per_branch": dict(zip(names, a4[1])),
                     "weights": dict(zip(names, a4[2].tolist())),
                     "weight_source": a4[5], "validation_accuracy": float(a4[3]),
                     **metrics(y_test, a4[4])}

    # Table 12 A5: remove learned projections, while retaining independent
    # per-CNN selectors, LR heads, and shared fixed probability fusion.
    raw_scalers = [StandardScaler().fit(X) for X in splits["train"]["X"]]
    raw = {part: [scaler.transform(X) for scaler, X in zip(raw_scalers, splits[part]["X"])]
           for part in ("train", "val", "test")}
    a5 = branch_ensemble(raw["train"], raw["val"], raw["test"], y_train, y_val,
                         fc, nclasses, cfg["fusion"]["weight_step"], fixed_weights=shared_weights)
    results["A5"] = {"configuration": "A0 without learned projections",
                     "selected_features_per_branch": dict(zip(names, a5[1])),
                     "weights": dict(zip(names, a5[2].tolist())),
                     "weight_source": a5[5], "validation_accuracy": float(a5[3]),
                     **metrics(y_test, a5[4])}

    # Persist audit evidence for every independently evaluated procedure.
    variants = {"A0": proposed, "A2": a2, "A3": a3, "A4": a4, "A5": a5}
    if args.no_aug_features:
        variants["A1"] = a1
    for stage, result in variants.items():
        np.savez_compressed(out / f"{stage}_test_probabilities.npz",
                            probabilities=result[4], y=y_test, paths=splits["test"]["paths"])
        pd.DataFrame(confusion_matrix(y_test, result[4].argmax(1))).to_csv(out / f"{stage}_confusion_matrix.csv", index=False)
        save_json(out / f"{stage}_classification_report.json", classification_report(y_test, result[4].argmax(1), output_dict=True, zero_division=0))
        save_json(out / f"{stage}_selected_features.json", {
            name: (selector.get_support(indices=True).tolist() if selector is not None
                   else list(range(clf.n_features_in_)))
            for name, (selector, clf) in zip(names, result[0])})
    save_json(out / "ensemble_weights.json", {"dataset": args.dataset, "run_seed": run_seed,
              "source": "fixed_shared_config", "weights": dict(zip(names, weights.tolist()))})
    save_json(out / "run_protocol.json", {"dataset": args.dataset, "run_seed": run_seed,
              "fixed_features_per_branch": fc[0], "test_n": len(y_test),
              "metric_units": "fractions", "f1_average": "macro", "config": cfg})
    torch.save({"projectors": [P.state_dict() for P in projectors],
                "scalers": scalers,
                "selectors": [selector for selector, _ in models],
                "heads": [head for _, head in models], "config": cfg, "weights": weights},
               out / "ensemble.pt")
    save_json(out / "ablation_metrics.json", {"run_seed": run_seed, **results})
    pd.DataFrame({"path": splits["test"]["paths"], "true": splits["test"]["y"],
                  "predicted": probabilities.argmax(1)}).to_csv(out / "test_predictions.csv", index=False)
    print(json.dumps(results, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    split = sub.add_parser("split", help="Create an image-stratified 70/20/10 manifest")
    split.add_argument("--root", required=True)
    split.add_argument("--out", required=True)
    split.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train-all", help="Fine-tune and checkpoint six backbones")
    extract = sub.add_parser("extract-all", help="Extract features and probabilities")
    ensemble = sub.add_parser("ensemble", help="Fit fusion with fixed shared weights and run ablations")
    for p in (train, extract):
        p.add_argument("--root", required=True)
        p.add_argument("--manifest", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--config", default="config.json")
    train.add_argument("--pretrained", action="store_true", default=True, help="Required ImageNet initialization; download failures stop the run")
    train.add_argument("--no-augment", action="store_true")
    train.add_argument("--run-seed", type=int, help="Override config seed for an independent run")
    extract.add_argument("--models", required=True)
    extract.add_argument("--run-seed", type=int, help="Seed and record this feature-extraction run")
    ensemble.add_argument("--dataset", required=True, choices=["Kvasir", "GERD"])
    ensemble.add_argument("--features", required=True)
    ensemble.add_argument("--out", required=True)
    ensemble.add_argument("--config", default="config.json")
    ensemble.add_argument("--no-aug-features")
    ensemble.add_argument("--run-seed", type=int, help="Override config seed for an independent run")
    args = parser.parse_args()
    {"split": make_split, "train-all": train_all,
     "extract-all": extract_all, "ensemble": run_ensemble}[args.command](args)

if __name__ == "__main__":
    main()
