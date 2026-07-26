from pathlib import Path
import subprocess
import sys
import zipfile

# Install into the active virtual environment created by build.sh.
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "requests", "pillow", "numpy", "pandas", "scikit-learn"
])

import requests
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path("work")
ZIP_PATH = ROOT / "lighting.zip"
EXTRACT_DIR = ROOT / "data"
PUBLIC_LINK = "https://cloud.mail.ru/public/GCsv/1BXmZPEBj"
WEBLINK_SUFFIX = "GCsv/1BXmZPEBj"
ROOT.mkdir(exist_ok=True)


def download_dataset():
    print("Resolving Cloud Mail download server...", flush=True)
    dispatcher = requests.get(
        "https://cloud.mail.ru/api/v2/dispatcher",
        timeout=60,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    dispatcher.raise_for_status()
    prefix = dispatcher.json()["body"]["weblink_get"][0]["url"].rstrip("/")
    direct_url = f"{prefix}/{WEBLINK_SUFFIX}"
    print("Downloading dataset from:", direct_url, flush=True)

    with requests.get(
        direct_url,
        stream=True,
        allow_redirects=True,
        timeout=(60, 900),
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": PUBLIC_LINK,
        },
    ) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with ZIP_PATH.open("wb") as output:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                output.write(chunk)
                downloaded += len(chunk)
                if downloaded % (100 * 1024 * 1024) < 4 * 1024 * 1024:
                    print(f"Downloaded {downloaded / 1024**2:.0f} MB", flush=True)
        print("Downloaded bytes:", downloaded, "expected:", total, flush=True)

    if ZIP_PATH.stat().st_size < 100_000_000 or not zipfile.is_zipfile(ZIP_PATH):
        raise RuntimeError(
            f"Downloaded file is not the expected ZIP: {ZIP_PATH.stat().st_size} bytes"
        )


if not ZIP_PATH.exists():
    download_dataset()

if not EXTRACT_DIR.exists() or not any(EXTRACT_DIR.rglob("*.png")):
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    print("Extracting dataset...", flush=True)
    with zipfile.ZipFile(ZIP_PATH) as archive:
        archive.extractall(EXTRACT_DIR)

extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
image_paths = [
    path for path in EXTRACT_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in extensions
]
image_by_id = {path.stem: path for path in image_paths}
print("Images found:", len(image_paths), flush=True)

train_csvs = [p for p in EXTRACT_DIR.rglob("train.csv") if p.is_file()]
test_csvs = [p for p in EXTRACT_DIR.rglob("test.csv") if p.is_file()]

if train_csvs and test_csvs:
    train_df = pd.read_csv(train_csvs[0])
    test_df = pd.read_csv(test_csvs[0])
    train_ids = train_df["id"].astype(str).tolist()
    labels = train_df["label"].astype(int).to_numpy()
    test_ids = test_df["id"].astype(str).tolist()
    train_paths = [image_by_id[item_id] for item_id in train_ids]
    test_paths = [image_by_id[item_id] for item_id in test_ids]
else:
    train_pairs = []
    test_paths = []
    for path in image_paths:
        parts_lower = [part.lower() for part in path.parts]
        parent = path.parent.name.lower()
        if parent in {"0", "1", "2"}:
            train_pairs.append((path, int(parent)))
        elif "test" in parts_lower:
            test_paths.append(path)
    train_pairs.sort(key=lambda item: item[0].stem)
    test_paths.sort(key=lambda path: path.stem)
    train_paths = [item[0] for item in train_pairs]
    labels = np.asarray([item[1] for item in train_pairs], dtype=int)
    test_ids = [path.stem for path in test_paths]

if not train_paths or not test_paths:
    raise RuntimeError(
        f"Could not identify train/test images: train={len(train_paths)}, test={len(test_paths)}"
    )

print("Train:", len(train_paths), "Test:", len(test_paths), flush=True)


def extract_features(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((160, 120))
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    value = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    saturation = (value - minimum) / (value + 1e-6)

    features = []
    percentiles = [0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 98, 99, 100]
    features.extend(np.percentile(gray, percentiles).tolist())
    features.extend([
        gray.mean(), gray.std(), np.median(gray),
        gray.min(), gray.max(), gray.max() - gray.min(),
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
            channel.mean(), channel.std(), channel.min(), channel.max(),
            *np.percentile(channel, [1, 5, 10, 25, 50, 75, 90, 95, 99]).tolist(),
        ])

    for channel in (value, saturation):
        features.extend([
            channel.mean(), channel.std(), channel.min(), channel.max(),
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
                    block.mean(), block.std(),
                    *np.percentile(block, [10, 50, 90]).tolist(),
                ])

    gradient_x = np.diff(gray, axis=1, append=gray[:, -1:])
    gradient_y = np.diff(gray, axis=0, append=gray[-1:, :])
    gradient = np.sqrt(gradient_x**2 + gradient_y**2)
    probabilities = histogram / (histogram.sum() + 1e-12)
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())

    p10, p25, p75, p90 = np.percentile(gray, [10, 25, 75, 90])
    features.extend([
        gradient.mean(), gradient.std(),
        np.percentile(gradient, 90), np.percentile(gradient, 99),
        entropy,
        gray.mean() / (gray.std() + 1e-6),
        p90 - p10,
        p75 - p25,
        value.mean() - saturation.mean(),
        float((gray < 0.10).mean() - (gray > 0.90).mean()),
    ])

    return np.asarray(features, dtype=np.float32)


print("Extracting train features...", flush=True)
X_train = np.vstack([extract_features(path) for path in train_paths])
print("Extracting test features...", flush=True)
X_test = np.vstack([extract_features(path) for path in test_paths])
print("Feature matrices:", X_train.shape, X_test.shape, flush=True)

extra_trees = ExtraTreesClassifier(
    n_estimators=700,
    max_features=0.85,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=11,
    n_jobs=-1,
)
random_forest = RandomForestClassifier(
    n_estimators=500,
    max_features=0.80,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=12,
    n_jobs=-1,
)
hist_gradient = HistGradientBoostingClassifier(
    max_iter=300,
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
