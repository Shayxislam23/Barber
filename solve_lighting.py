from __future__ import annotations

from io import BytesIO
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageFile
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path("work")
ZIP_PATH = ROOT / "lighting.zip"
PUBLIC_LINK = "https://cloud.mail.ru/public/GCsv/1BXmZPEBj"
WEBLINK_SUFFIX = "GCsv/1BXmZPEBj"
ROOT.mkdir(exist_ok=True)


def download_dataset() -> None:
    print("Resolving Cloud Mail download server...", flush=True)
    dispatcher = requests.get(
        "https://cloud.mail.ru/api/v2/dispatcher",
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    dispatcher.raise_for_status()
    prefix = dispatcher.json()["body"]["weblink_get"][0]["url"].rstrip("/")
    direct_url = f"{prefix}/{WEBLINK_SUFFIX}"

    print("Downloading the image archive...", flush=True)
    with requests.get(
        direct_url,
        stream=True,
        allow_redirects=True,
        timeout=(60, 1200),
        headers={"User-Agent": "Mozilla/5.0", "Referer": PUBLIC_LINK},
    ) as response:
        response.raise_for_status()
        downloaded = 0
        with ZIP_PATH.open("wb") as output:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded // (100 * 1024 * 1024) != (downloaded - len(chunk)) // (100 * 1024 * 1024):
                    print(f"Downloaded {downloaded / 1024**2:.0f} MB", flush=True)

    if ZIP_PATH.stat().st_size < 100_000_000 or not zipfile.is_zipfile(ZIP_PATH):
        raise RuntimeError(f"Invalid archive: {ZIP_PATH.stat().st_size} bytes")
    print(f"Archive ready: {ZIP_PATH.stat().st_size / 1024**2:.1f} MB", flush=True)


if not ZIP_PATH.exists() or not zipfile.is_zipfile(ZIP_PATH):
    download_dataset()


def normalized_stem(member: str) -> str:
    return PurePosixPath(member).stem


def find_csv_member(members: list[str], filename: str) -> str | None:
    candidates = [m for m in members if PurePosixPath(m).name.lower() == filename.lower()]
    if not candidates:
        return None
    # Prefer CSVs near the archive root rather than unrelated nested metadata.
    return min(candidates, key=lambda item: (len(PurePosixPath(item).parts), len(item)))


def read_image(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    with archive.open(member) as source:
        raw = source.read()
    with Image.open(BytesIO(raw)) as image:
        image = image.convert("RGB").resize((160, 120), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def extract_features(rgb: np.ndarray) -> np.ndarray:
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    value = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = (value - minimum) / (value + 1e-6)

    features: list[float] = []
    percentiles = [0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 98, 99, 100]
    features.extend(np.percentile(gray, percentiles).tolist())
    features.extend([
        float(gray.mean()), float(gray.std()), float(np.median(gray)),
        float(gray.min()), float(gray.max()), float(np.ptp(gray)),
    ])

    histogram, _ = np.histogram(gray, bins=64, range=(0, 1), density=True)
    features.extend(histogram.tolist())

    for threshold in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        features.append(float((gray <= threshold).mean()))
    for threshold in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99]:
        features.append(float((gray >= threshold).mean()))

    for channel_index in range(3):
        channel = rgb[..., channel_index]
        features.extend([
            float(channel.mean()), float(channel.std()), float(channel.min()), float(channel.max()),
            *np.percentile(channel, [1, 5, 10, 25, 50, 75, 90, 95, 99]).tolist(),
        ])

    for channel in (value, saturation):
        features.extend([
            float(channel.mean()), float(channel.std()), float(channel.min()), float(channel.max()),
            *np.percentile(channel, [1, 5, 10, 25, 50, 75, 90, 95, 99]).tolist(),
        ])

    height, width = gray.shape
    for grid_size in (3, 4):
        for row in range(grid_size):
            for column in range(grid_size):
                block = gray[
                    row * height // grid_size:(row + 1) * height // grid_size,
                    column * width // grid_size:(column + 1) * width // grid_size,
                ]
                features.extend([
                    float(block.mean()), float(block.std()),
                    *np.percentile(block, [10, 50, 90]).tolist(),
                ])

    gradient_x = np.diff(gray, axis=1, append=gray[:, -1:])
    gradient_y = np.diff(gray, axis=0, append=gray[-1:, :])
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    probabilities = histogram / (histogram.sum() + 1e-12)
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())
    p10, p25, p75, p90 = np.percentile(gray, [10, 25, 75, 90])

    features.extend([
        float(gradient.mean()), float(gradient.std()),
        float(np.percentile(gradient, 90)), float(np.percentile(gradient, 99)),
        entropy,
        float(gray.mean() / (gray.std() + 1e-6)),
        float(p90 - p10), float(p75 - p25),
        float(value.mean() - saturation.mean()),
        float((gray < 0.10).mean() - (gray > 0.90).mean()),
    ])
    return np.asarray(features, dtype=np.float32)


with zipfile.ZipFile(ZIP_PATH) as archive:
    members = [info.filename for info in archive.infolist() if not info.is_dir()]
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_members = [m for m in members if PurePosixPath(m).suffix.lower() in image_extensions]
    print(f"Images in archive: {len(image_members)}", flush=True)

    train_csv_member = find_csv_member(members, "train.csv")
    test_csv_member = find_csv_member(members, "test.csv")

    train_members: list[str] = []
    test_members: list[str] = []
    labels: np.ndarray
    test_ids: list[str]

    if train_csv_member and test_csv_member:
        print("Using train.csv and test.csv from the archive", flush=True)
        with archive.open(train_csv_member) as source:
            train_df = pd.read_csv(source, dtype={"id": str})
        with archive.open(test_csv_member) as source:
            test_df = pd.read_csv(source, dtype={"id": str})

        # Prefer images located under train/test paths if stems overlap.
        by_stem: dict[str, list[str]] = {}
        for member in image_members:
            by_stem.setdefault(normalized_stem(member), []).append(member)

        def select_member(item_id: str, split: str) -> str:
            candidates = by_stem.get(str(item_id), [])
            if not candidates:
                raise KeyError(f"Image id not found: {item_id}")
            split_matches = [m for m in candidates if split in [p.lower() for p in PurePosixPath(m).parts]]
            return split_matches[0] if split_matches else candidates[0]

        train_ids = train_df["id"].astype(str).tolist()
        test_ids = test_df["id"].astype(str).tolist()
        labels = train_df["label"].astype(int).to_numpy()
        train_members = [select_member(item_id, "train") for item_id in train_ids]
        test_members = [select_member(item_id, "test") for item_id in test_ids]
    else:
        print("CSV metadata not found inside archive; inferring split from folders", flush=True)
        train_pairs: list[tuple[str, int]] = []
        test_members = []
        class_map = {"0": 0, "1": 1, "2": 2, "dark": 0, "normal": 1, "bright": 2}
        for member in image_members:
            parts = [part.lower() for part in PurePosixPath(member).parts]
            parent = PurePosixPath(member).parent.name.lower()
            if "test" in parts:
                test_members.append(member)
            elif parent in class_map:
                train_pairs.append((member, class_map[parent]))
        train_pairs.sort(key=lambda item: normalized_stem(item[0]))
        test_members.sort(key=normalized_stem)
        train_members = [member for member, _ in train_pairs]
        labels = np.asarray([label for _, label in train_pairs], dtype=int)
        test_ids = [normalized_stem(member) for member in test_members]

    if not train_members or not test_members:
        raise RuntimeError(f"Could not identify train/test images: train={len(train_members)}, test={len(test_members)}")
    print(f"Train: {len(train_members)}; test: {len(test_members)}", flush=True)

    print("Extracting train image features...", flush=True)
    train_features = []
    for index, member in enumerate(train_members, start=1):
        train_features.append(extract_features(read_image(archive, member)))
        if index % 250 == 0:
            print(f"Train features: {index}/{len(train_members)}", flush=True)

    print("Extracting test image features...", flush=True)
    test_features = []
    for index, member in enumerate(test_members, start=1):
        test_features.append(extract_features(read_image(archive, member)))
        if index % 100 == 0:
            print(f"Test features: {index}/{len(test_members)}", flush=True)

X_train = np.vstack(train_features)
X_test = np.vstack(test_features)
print("Feature matrices:", X_train.shape, X_test.shape, flush=True)

extra_trees = ExtraTreesClassifier(
    n_estimators=450,
    max_features=0.85,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=11,
    n_jobs=-1,
)
random_forest = RandomForestClassifier(
    n_estimators=350,
    max_features=0.80,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=12,
    n_jobs=-1,
)
hist_gradient = HistGradientBoostingClassifier(
    max_iter=220,
    learning_rate=0.05,
    max_leaf_nodes=31,
    l2_regularization=2.0,
    random_state=13,
)
svc = make_pipeline(
    StandardScaler(),
    SVC(C=10, gamma="scale", probability=True, class_weight="balanced", random_state=14),
)
ensemble = VotingClassifier(
    estimators=[
        ("extra", extra_trees),
        ("rf", random_forest),
        ("hgb", hist_gradient),
        ("svc", svc),
    ],
    voting="soft",
    weights=[4, 3, 2, 2],
    n_jobs=-1,
)

print("Training ensemble...", flush=True)
ensemble.fit(X_train, labels)
predictions = ensemble.predict(X_test).astype(int)

submission = pd.DataFrame({"id": test_ids, "label": predictions})
submission.to_csv("submission_lighting.csv", index=False)
print(submission.head(), flush=True)
print("Class counts:", submission["label"].value_counts().sort_index().to_dict(), flush=True)
print("Saved submission_lighting.csv", flush=True)
