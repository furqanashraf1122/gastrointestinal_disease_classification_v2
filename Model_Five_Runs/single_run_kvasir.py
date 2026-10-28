"""CLI conversion of the supplied notebook Kvasir cell; ResNet-50 only."""
import argparse

def main(args):
    """
    Local PyTorch Kvasir 8-class classification
    ==========================================
    Dataset: D:\\Rizwan new\\archive\\kvasir-dataset-v2

    Uses a reproducible 6,000-image subset.
    Training:   4,200 images (70%)
    Validation: 1,200 images (20%)
    Testing:      600 images (10%)

    The original dataset is never deleted or modified.
    """

    from pathlib import Path
    import random
    import json
    import copy
    import time

    import numpy as np
    import pandas as pd
    from PIL import Image, UnidentifiedImageError

    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models

    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        classification_report,
        confusion_matrix,
    )

    # ============================================================
    # 1. CONFIGURATION
    # ============================================================

    DATA_ROOT = Path(args.root).resolve()
    OUTPUT_DIR = Path(args.out).resolve()
    MANIFEST_PATH = Path(args.manifest).resolve()

    SEED = args.seed
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    EPOCHS = args.epochs
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 0
    USE_PRETRAINED = True

    # Folder name: number of images used from that class.
    CLASS_COUNTS = {
        "dyed-lifted-polyps": 1000,
        "dyed-resection-margins": 500,
        "esophagitis": 1000,
        "normal-cecum": 500,
        "normal-pylorus": 500,
        "normal-z-line": 500,
        "polyps": 1000,
        "ulcerative-colitis": 1000,
    }

    IMAGE_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".bmp",
        ".tif", ".tiff", ".webp"
    }

    SPLIT_COUNTS = {
        1000: {"train": 700, "validation": 200, "test": 100},
        500: {"train": 350, "validation": 100, "test": 50},
    }

    CLASS_NAMES = list(CLASS_COUNTS.keys())
    CLASS_TO_LABEL = {
        name: index for index, name in enumerate(CLASS_NAMES)
    }
    NUM_CLASSES = len(CLASS_NAMES)

    # ============================================================
    # 2. REPRODUCIBILITY AND DEVICE
    # ============================================================

    def set_seed(seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("KVASIR GASTROINTESTINAL DISEASE CLASSIFICATION")
    print("=" * 65)
    print("Dataset root:", DATA_ROOT)
    print("Output directory:", OUTPUT_DIR)
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Number of classes:", NUM_CLASSES)
    print("Target images:", sum(CLASS_COUNTS.values()))

    # ============================================================
    # 3. FIND IMAGES RECURSIVELY
    # ============================================================

    def find_images(folder: Path):
        """
        Recursively discover images inside a class folder.
        Supports common image formats.
        """
        return sorted(
            path
            for path in folder.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    # ============================================================
    # 4. CREATE THE FIXED 6,000-IMAGE MANIFEST
    # ============================================================

    def create_manifest():
        """
        Selects the target number of images from each class and
        creates a stratified 70/20/10 split.

        Sampling is performed with a fixed random seed.
        Source images are not copied, moved, or deleted.
        """

        if not DATA_ROOT.is_dir():
            raise FileNotFoundError(
                f"Dataset directory does not exist:\n{DATA_ROOT}\n"
                "Please correct DATA_ROOT at the top of this script."
            )

        rows = []
        rng = random.Random(SEED)

        print("\nChecking dataset folders...")

        for class_name, target_count in CLASS_COUNTS.items():

            class_dir = DATA_ROOT / class_name

            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing class folder:\n{class_dir}"
                )

            images = find_images(class_dir)

            print(
                f"{class_name:28s} "
                f"Found: {len(images):5d} | "
                f"Required: {target_count}"
            )

            if len(images) < target_count:
                raise ValueError(
                    f"Class '{class_name}' has only {len(images)} images, "
                    f"but {target_count} are required."
                )

            # Randomly select the target number of images.
            rng.shuffle(images)
            selected = images[:target_count]

            # Split the selected images within this class.
            split_counts = SPLIT_COUNTS[target_count]

            train_end = split_counts["train"]
            val_end = (
                split_counts["train"]
                + split_counts["validation"]
            )

            assignments = []

            assignments.extend(
                ("train", p)
                for p in selected[:train_end]
            )

            assignments.extend(
                ("validation", p)
                for p in selected[train_end:val_end]
            )

            assignments.extend(
                ("test", p)
                for p in selected[val_end:]
            )

            for split_name, image_path in assignments:
                rows.append({
                    "path": str(image_path.resolve()),
                    "class_name": class_name,
                    "label": CLASS_TO_LABEL[class_name],
                    "split": split_name,
                })

        df = pd.DataFrame(rows)

        # Validate total image count.
        expected_total = sum(CLASS_COUNTS.values())

        if len(df) != expected_total:
            raise RuntimeError(
                f"Expected {expected_total} images, found {len(df)}."
            )

        # Check duplicate file paths.
        if df["path"].duplicated().any():
            raise RuntimeError(
                "Duplicate image paths detected in the manifest."
            )

        # Validate every class/split count.
        expected_split_counts = []

        for class_name, total in CLASS_COUNTS.items():
            for split_name, count in SPLIT_COUNTS[total].items():
                expected_split_counts.append(
                    (class_name, split_name, count)
                )

        observed = df.groupby(
            ["class_name", "split"]
        ).size().to_dict()

        for class_name, split_name, expected_count in expected_split_counts:
            actual_count = observed.get(
                (class_name, split_name), 0
            )

            if actual_count != expected_count:
                raise RuntimeError(
                    f"Incorrect split for {class_name}/{split_name}: "
                    f"{actual_count} instead of {expected_count}"
                )

        df.to_csv(MANIFEST_PATH, index=False)

        print("\nManifest saved to:")
        print(MANIFEST_PATH)

        return df

    def load_or_create_manifest():
        """
        Reuses the existing manifest so subsequent runs use the
        same image selection and split.
        """

        if not MANIFEST_PATH.exists():
            raise FileNotFoundError("Use run_five_runs.py to prepare the fixed manifest first")

        print("\nLoading existing manifest:")
        print(MANIFEST_PATH)

        df = pd.read_csv(MANIFEST_PATH)

        required_columns = {
            "path", "class_name", "label", "split"
        }

        if not required_columns.issubset(df.columns):
            raise ValueError(
                "Manifest is missing required columns. "
                "Back it up and remove it to generate a new manifest."
            )

        expected_total = sum(CLASS_COUNTS.values())

        if len(df) != expected_total:
            raise ValueError(
                f"Manifest has {len(df)} rows, "
                f"but {expected_total} are expected. "
                "Back it up and remove it to regenerate."
            )

        # Ensure paths still exist.
        missing_paths = [
            path
            for path in df["path"].astype(str)
            if not Path(path).is_file()
        ]

        if missing_paths:
            raise FileNotFoundError(
                f"{len(missing_paths)} manifest images cannot be found. "
                "Check the dataset location or regenerate the manifest."
            )

        # Validate folder-to-label mapping.
        for row in df.itertuples(index=False):
            image_path = Path(row.path).resolve()

            try:
                relative_path = image_path.relative_to(
                    DATA_ROOT.resolve()
                )
            except ValueError as exc:
                raise ValueError(
                    f"Image path is outside DATA_ROOT: {image_path}"
                ) from exc

            actual_class = relative_path.parts[0]

            if actual_class != row.class_name:
                raise ValueError(
                    f"Folder/label mismatch for {image_path}: "
                    f"manifest class is '{row.class_name}', "
                    f"folder class is '{actual_class}'."
                )

            if CLASS_TO_LABEL.get(row.class_name) != int(row.label):
                raise ValueError(
                    f"Incorrect numeric label for {image_path}"
                )

            if row.split not in {"train", "validation", "test"}:
                raise ValueError(
                    f"Invalid split '{row.split}' in manifest."
                )

        return df

    manifest = load_or_create_manifest()

    print("\nSplit totals:")
    print(
        manifest["split"].value_counts().reindex(
            ["train", "validation", "test"],
            fill_value=0
        )
    )

    print("\nPer-class split counts:")
    print(
        pd.crosstab(
            manifest["class_name"],
            manifest["split"]
        ).reindex(
            index=CLASS_NAMES,
            columns=["train", "validation", "test"],
            fill_value=0
        )
    )

    # ============================================================
    # 5. IMAGE TRANSFORMS
    # ============================================================

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.08
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # ============================================================
    # 6. PYTORCH DATASET
    # ============================================================

    class KvasirManifestDataset(Dataset):

        def __init__(self, dataframe, transform=None):
            self.dataframe = dataframe.reset_index(
                drop=True
            ).copy()

            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, index):
            row = self.dataframe.iloc[index]

            image_path = row["path"]
            label = int(row["label"])

            try:
                with Image.open(image_path) as image:
                    image = image.convert("RGB")

            except (UnidentifiedImageError, OSError) as exc:
                raise RuntimeError(
                    f"Unable to read image: {image_path}"
                ) from exc

            if self.transform is not None:
                image = self.transform(image)

            return image, label, image_path

    # Separate the three datasets.
    train_df = manifest[
        manifest["split"] == "train"
    ].copy()

    val_df = manifest[
        manifest["split"] == "validation"
    ].copy()

    test_df = manifest[
        manifest["split"] == "test"
    ].copy()

    train_dataset = KvasirManifestDataset(
        train_df, train_transform
    )

    val_dataset = KvasirManifestDataset(
        val_df, eval_transform
    )

    test_dataset = KvasirManifestDataset(
        test_df, eval_transform
    )

    loader_settings = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_settings
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_settings
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_settings
    )

    print("\nDataset sizes:")
    print("Training:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Testing:", len(test_dataset))

    # ============================================================
    # 7. CREATE RESNET-50
    # ============================================================

    print("\nLoading ResNet-50...")

    if USE_PRETRAINED:
        try:
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)

        except Exception as exc:
            raise RuntimeError(
                "Could not load ImageNet pretrained ResNet-50 weights. "
                "Check your internet connection and try again, or set "
                "USE_PRETRAINED = False if random initialization is intended."
            ) from exc

    else:
        model = models.resnet50(weights=None)

    # Replace the original 1,000-class output layer with 8 classes.
    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    model = model.to(device)

    # ============================================================
    # 8. LOSS, OPTIMIZER, SCHEDULER AND MIXED PRECISION
    # ============================================================

    # In this dataset, four classes have 700 training images and
    # four classes have 350 training images. Use class weights.
    train_counts = (
        train_df["label"]
        .value_counts()
        .reindex(range(NUM_CLASSES), fill_value=0)
        .to_numpy()
    )

    if np.any(train_counts == 0):
        raise ValueError(
            "At least one class has no training images."
        )

    class_weights = (
        len(train_df)
        / (NUM_CLASSES * torch.tensor(
            train_counts, dtype=torch.float32
        ))
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    # Mixed precision is enabled on CUDA and disabled on CPU.
    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(device.type == "cuda")
        )
    except (AttributeError, TypeError):
        # Compatibility with older PyTorch versions.
        scaler = torch.cuda.amp.GradScaler(
            enabled=(device.type == "cuda")
        )

    # ============================================================
    # 9. TRAINING AND VALIDATION FUNCTIONS
    # ============================================================

    def run_epoch(loader, training):
        model.train(training)

        total_loss = 0.0
        all_labels = []
        all_predictions = []

        for images, labels, _paths in loader:

            images = images.to(
                device, non_blocking=True
            )

            labels = labels.to(
                device, non_blocking=True
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(training):

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=(device.type == "cuda")
                ):
                    logits = model(images)
                    loss = criterion(logits, labels)

                if training:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            total_loss += loss.item() * labels.size(0)

            predictions = logits.argmax(dim=1)

            all_labels.extend(
                labels.detach().cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.detach().cpu().numpy().tolist()
            )

        average_loss = total_loss / len(loader.dataset)

        accuracy = accuracy_score(
            all_labels,
            all_predictions
        )

        return average_loss, accuracy

    # ============================================================
    # 10. MODEL TRAINING
    # ============================================================

    best_val_accuracy = -1.0
    best_state = None
    history = []

    start_time = time.time()

    print("\nStarting model training...")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_accuracy = run_epoch(
            train_loader, training=True
        )

        val_loss, val_accuracy = run_epoch(
            val_loader, training=False
        )

        scheduler.step(val_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "learning_rate": current_lr,
        }

        history.append(record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_accuracy:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # Select the checkpoint using validation accuracy only.
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_state = copy.deepcopy(
                model.state_dict()
            )

            checkpoint = {
                "model_state_dict": best_state,
                "class_names": CLASS_NAMES,
                "class_counts": CLASS_COUNTS,
                "image_size": IMAGE_SIZE,
                "seed": SEED,
                "best_validation_accuracy": best_val_accuracy,
            }

            torch.save(
                checkpoint,
                OUTPUT_DIR / "best_resnet50_kvasir.pth"
            )

            print("  Saved new best validation checkpoint.")

    # Save training history.
    history_df = pd.DataFrame(history)

    history_df.to_csv(
        OUTPUT_DIR / "training_history.csv",
        index=False
    )

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a valid model checkpoint."
        )

    # Restore the best validation checkpoint.
    model.load_state_dict(best_state)

    print("\nTraining complete.")
    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    # ============================================================
    # 11. FINAL TEST EVALUATION
    # ============================================================

    # Test data is not used for model selection.
    # Evaluate it once after selecting the best validation checkpoint.

    model.eval()

    y_true = []
    y_pred = []
    y_prob = []
    test_paths = []

    with torch.no_grad():

        for images, labels, paths in test_loader:

            images = images.to(
                device, non_blocking=True
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):
                logits = model(images)

            probabilities = torch.softmax(
                logits, dim=1
            ).cpu().numpy()

            predictions = probabilities.argmax(axis=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(predictions.tolist())
            y_prob.extend(probabilities.tolist())
            test_paths.extend(list(paths))

    # ============================================================
    # 12. CALCULATE METRICS
    # ============================================================

    test_accuracy = accuracy_score(
        y_true, y_pred
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )
    )

    elapsed_minutes = (
        time.time() - start_time
    ) / 60.0

    summary = {
        "test_images": len(y_true),
        "accuracy": float(test_accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "best_validation_accuracy": float(best_val_accuracy),
        "elapsed_minutes": float(elapsed_minutes),
        "device": str(device),
        "seed": SEED,
    }

    with open(
        OUTPUT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    # ============================================================
    # 13. CLASSIFICATION REPORT
    # ============================================================

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    # ============================================================
    # 14. CONFUSION MATRIX
    # ============================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    cm_df = pd.DataFrame(
        cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    cm_df.to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )

    # ============================================================
    # 15. SAVE TEST PREDICTIONS AND CLASS PROBABILITIES
    # ============================================================

    predictions_df = pd.DataFrame({
        "path": test_paths,
        "true_label": y_true,
        "predicted_label": y_pred,
        "true_class": [
            CLASS_NAMES[index] for index in y_true
        ],
        "predicted_class": [
            CLASS_NAMES[index] for index in y_pred
        ],
    })

    probability_array = np.asarray(y_prob)

    for index, class_name in enumerate(CLASS_NAMES):
        predictions_df[f"prob_{class_name}"] = (
            probability_array[:, index]
        )

    predictions_df.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False
    )

    # ============================================================
    # 16. PRINT FINAL RESULTS
    # ============================================================

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS")
    print("=" * 65)

    print(json.dumps(summary, indent=4))

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("\nConfusion matrix:")
    print(cm_df)

    print("\nSaved output files:")
    for output_file in sorted(OUTPUT_DIR.iterdir()):
        if output_file.is_file():
            print(" -", output_file.name)

    print("\nAll results saved to:")
    print(OUTPUT_DIR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    main(parser.parse_args())
