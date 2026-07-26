from __future__ import annotations

import csv
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO, TextIOWrapper
from pathlib import Path, PurePosixPath

import numpy as np
import requests
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path("work")
ZIP_PATH = ROOT / "lighting.zip"
PUBLIC_LINK = "https://cloud.mail.ru/public/GCsv/1BXmZPEBj"
WEBLINK_SUFFIX = "GCsv/1BXmZPEBj"
ROOT.mkdir(exist_ok=True)


def download_dataset() -> None:
    dispatcher = requests.get(
        "https://cloud.mail.ru/api/v2/dispatcher",
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    dispatcher.raise_for_status()
    prefix = dispatcher.json()["body"]["weblink_get"][0]["url"].rstrip("/")
    direct_url = f"{prefix}/{WEBLINK_SUFFIX}"
    print("Downloading image archive...", flush=True)
    with requests.get(
        direct_url,
        stream=True,
        allow_redirects=True,
        timeout=(60, 1200),
        headers={"User-Agent": "Mozilla/5.0", "Referer": PUBLIC_LINK},
    ) as response:
        response.raise_for_status()
        with ZIP_PATH.open("wb") as output:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    output.write(chunk)
    if ZIP_PATH.stat().st_size < 100_000_000 or not zipfile.is_zipfile(ZIP_PATH):
        raise RuntimeError("Dataset download is invalid")
    print(f"Archive downloaded: {ZIP_PATH.stat().st_size / 1024**2:.1f} MB", flush=True)


if not ZIP_PATH.exists() or not zipfile.is_zipfile(ZIP_PATH):
    download_dataset()


def csv_rows(archive: zipfile.ZipFile, member: str) -> list[dict[str, str]]:
    with archive.open(member) as raw:
        with TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
            return list(csv.DictReader(text))


def best_csv(members: list[str], filename: str) -> str | None:
    matches = [m for m in members if PurePosixPath(m).name.lower() == filename]
    return min(matches, key=lambda m: (len(PurePosixPath(m).parts), len(m))) if matches else None


with zipfile.ZipFile(ZIP_PATH) as archive:
    members = [item.filename for item in archive.infolist() if not item.is_dir()]
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    images = [m for m in members if PurePosixPath(m).suffix.lower() in extensions]
    train_csv = best_csv(members, "train.csv")
    test_csv = best_csv(members, "test.csv")

    if train_csv and test_csv:
        train_meta = csv_rows(archive, train_csv)
        test_meta = csv_rows(archive, test_csv)
        by_stem: dict[str, list[str]] = {}
        for member in images:
            by_stem.setdefault(PurePosixPath(member).stem, []).append(member)

        def choose(item_id: str, split: str) -> str:
            candidates = by_stem.get(str(item_id), [])
            if not candidates:
                raise KeyError(f"Missing image id {item_id}")
            preferred = [m for m in candidates if split in [p.lower() for p in PurePosixPath(m).parts]]
            return preferred[0] if preferred else candidates[0]

        train_members = [choose(row["id"], "train") for row in train_meta]
        labels = np.asarray([int(row["label"]) for row in train_meta], dtype=np.int64)
        test_ids = [str(row["id"]) for row in test_meta]
        test_members = [choose(item_id, "test") for item_id in test_ids]
    else:
        class_map = {"0": 0, "1": 1, "2": 2, "dark": 0, "normal": 1, "bright": 2}
        train_pairs: list[tuple[str, int]] = []
        test_members: list[str] = []
        for member in images:
            path = PurePosixPath(member)
            parts = [p.lower() for p in path.parts]
            parent = path.parent.name.lower()
            if "test" in parts:
                test_members.append(member)
            elif parent in class_map:
                train_pairs.append((member, class_map[parent]))
        train_pairs.sort(key=lambda pair: PurePosixPath(pair[0]).stem)
        test_members.sort(key=lambda m: PurePosixPath(m).stem)
        train_members = [m for m, _ in train_pairs]
        labels = np.asarray([label for _, label in train_pairs], dtype=np.int64)
        test_ids = [PurePosixPath(m).stem for m in test_members]

print(f"Images found: train={len(train_members)}, test={len(test_members)}", flush=True)
if len(train_members) < 100 or len(test_members) < 10:
    raise RuntimeError("Train/test image split was not identified")

_WORKER_ARCHIVE: zipfile.ZipFile | None = None


def init_worker(zip_name: str) -> None:
    global _WORKER_ARCHIVE
    _WORKER_ARCHIVE = zipfile.ZipFile(zip_name)


def image_features(member: str) -> np.ndarray:
    if _WORKER_ARCHIVE is None:
        raise RuntimeError("Worker archive is not initialized")
    with _WORKER_ARCHIVE.open(member) as source:
        raw = source.read()
    with Image.open(BytesIO(raw)) as image:
        image.draft("RGB", (64, 64))
        image = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
        rgb = np.asarray(image, dtype=np.float32) / 255.0

    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    value = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = (value - minimum) / (value + 1e-6)
    qs = np.percentile(gray, [1, 5, 10, 20, 25, 40, 50, 60, 75, 80, 90, 95, 99])
    features = [
        gray.mean(), gray.std(), *qs.tolist(),
        (gray < 0.05).mean(), (gray < 0.10).mean(), (gray < 0.20).mean(),
        (gray > 0.80).mean(), (gray > 0.90).mean(), (gray > 0.95).mean(),
        value.mean(), value.std(), saturation.mean(), saturation.std(),
    ]
    for channel in range(3):
        ch = rgb[..., channel]
        features.extend([ch.mean(), ch.std(), *np.percentile(ch, [10, 50, 90]).tolist()])
    # Coarse spatial illumination pattern.
    for row in range(3):
        for col in range(3):
            block = gray[row * 64 // 3:(row + 1) * 64 // 3, col * 64 // 3:(col + 1) * 64 // 3]
            features.extend([block.mean(), block.std()])
    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    gradient = np.sqrt(gx * gx + gy * gy)
    features.extend([gradient.mean(), gradient.std(), np.percentile(gradient, 90)])
    return np.asarray(features, dtype=np.float32)


def feature_matrix(items: list[str], name: str) -> np.ndarray:
    workers = max(1, min(4, os.cpu_count() or 2))
    print(f"Computing {name} features with {workers} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(str(ZIP_PATH),)) as pool:
        rows = list(pool.map(image_features, items, chunksize=12))
    return np.vstack(rows)


X = feature_matrix(train_members, "train")
Xt = feature_matrix(test_members, "test")
print("Feature matrices:", X.shape, Xt.shape, flush=True)

# Standardize using train statistics.
mean = X.mean(axis=0)
std = X.std(axis=0)
std[std < 1e-6] = 1.0
Z = (X - mean) / std
Zt = (Xt - mean) / std
classes = np.asarray([0, 1, 2], dtype=np.int64)

# 1) Regularized linear discriminant classifier.
class_means = np.vstack([Z[labels == cls].mean(axis=0) for cls in classes])
centered = np.vstack([Z[labels == cls] - class_means[cls] for cls in classes])
covariance = centered.T @ centered / max(1, len(Z) - len(classes))
covariance += np.eye(covariance.shape[0], dtype=np.float32) * 0.35
inverse_covariance = np.linalg.pinv(covariance)
lda_scores = np.column_stack([
    -0.5 * np.einsum("ij,jk,ik->i", Zt - class_means[cls], inverse_covariance, Zt - class_means[cls])
    for cls in classes
])

# 2) Diagonal Gaussian classifier.
variances = np.vstack([Z[labels == cls].var(axis=0) + 0.20 for cls in classes])
nb_scores = np.column_stack([
    -0.5 * (((Zt - class_means[cls]) ** 2 / variances[cls]).sum(axis=1) + np.log(variances[cls]).sum())
    for cls in classes
])

# 3) Weighted k-nearest neighbours, vectorized for the 300 test rows.
distances = ((Zt[:, None, :] - Z[None, :, :]) ** 2).mean(axis=2)
k = min(21, len(Z))
nearest = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
knn_scores = np.zeros((len(Zt), 3), dtype=np.float64)
for row_index in range(len(Zt)):
    indices = nearest[row_index]
    weights = 1.0 / (distances[row_index, indices] + 1e-5)
    for cls in classes:
        knn_scores[row_index, cls] = weights[labels[indices] == cls].sum()

# 4) Ordered exposure thresholds optimized on training data.
exposure_train = 0.45 * X[:, 0] + 0.25 * X[:, 8] + 0.20 * X[:, 12] + 0.10 * X[:, 21]
exposure_test = 0.45 * Xt[:, 0] + 0.25 * Xt[:, 8] + 0.20 * Xt[:, 12] + 0.10 * Xt[:, 21]
order = np.argsort(exposure_train)
sorted_labels = labels[order]
n = len(sorted_labels)
cumulative = np.zeros((3, n + 1), dtype=np.int64)
for cls in classes:
    cumulative[cls, 1:] = np.cumsum(sorted_labels == cls)
best_correct = -1
best_i, best_j = n // 3, 2 * n // 3
for i in range(1, n - 1):
    correct_zero = cumulative[0, i]
    for j in range(i + 1, n):
        correct = correct_zero + (cumulative[1, j] - cumulative[1, i]) + (cumulative[2, n] - cumulative[2, j])
        if correct > best_correct:
            best_correct, best_i, best_j = int(correct), i, j
sorted_exposure = exposure_train[order]
t1 = float((sorted_exposure[best_i - 1] + sorted_exposure[best_i]) / 2)
t2 = float((sorted_exposure[best_j - 1] + sorted_exposure[best_j]) / 2)
threshold_predictions = np.where(exposure_test < t1, 0, np.where(exposure_test < t2, 1, 2))
print(f"Exposure threshold train accuracy: {best_correct / n:.4f}", flush=True)

# Normalize score matrices and combine with the ordered brightness decision.
def softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponent = np.exp(np.clip(shifted, -40, 40))
    return exponent / exponent.sum(axis=1, keepdims=True)

combined = 0.36 * softmax(lda_scores) + 0.29 * softmax(nb_scores) + 0.25 * (knn_scores / (knn_scores.sum(axis=1, keepdims=True) + 1e-12))
combined[np.arange(len(combined)), threshold_predictions] += 0.10
predictions = combined.argmax(axis=1).astype(int)

with Path("submission_lighting.csv").open("w", encoding="utf-8", newline="") as output:
    writer = csv.writer(output)
    writer.writerow(["id", "label"])
    writer.writerows(zip(test_ids, predictions.tolist()))

counts = {int(cls): int((predictions == cls).sum()) for cls in classes}
print("Class counts:", counts, flush=True)
print("Saved submission_lighting.csv", flush=True)
