from pathlib import Path
import zipfile
import subprocess
import sys

subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q',
    'gdown', 'pillow', 'numpy', 'pandas', 'scikit-learn', 'opencv-python-headless'
])

import gdown
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

FILE_ID = '1a1CqcUCqULZM9w6Rjc5cJT8rvG8bKpH9'
ROOT = Path('work')
ZIP = ROOT / 'lighting.zip'
EXTRACT = ROOT / 'data'
ROOT.mkdir(exist_ok=True)

if not ZIP.exists():
    print('Downloading dataset...')
    result = gdown.download(id=FILE_ID, output=str(ZIP), quiet=False, fuzzy=True)
    if not result or not ZIP.exists():
        raise RuntimeError('Dataset download failed')

if not EXTRACT.exists():
    EXTRACT.mkdir(parents=True, exist_ok=True)
    print('Extracting dataset...')
    with zipfile.ZipFile(ZIP) as zf:
        zf.extractall(EXTRACT)

exts = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
all_images = [p for p in EXTRACT.rglob('*') if p.is_file() and p.suffix.lower() in exts]
print('Images found:', len(all_images))

train_items = []
test_items = []
for p in all_images:
    parts = [x.lower() for x in p.parts]
    parent = p.parent.name
    if parent in {'0', '1', '2'}:
        train_items.append((p, int(parent)))
    elif 'test' in parts:
        test_items.append(p)

if not train_items or not test_items:
    raise RuntimeError(f'Could not identify train/test images. train={len(train_items)}, test={len(test_items)}')

train_items.sort(key=lambda x: x[0].stem)
test_items.sort(key=lambda p: p.stem)
print('Train:', len(train_items), 'Test:', len(test_items))


def features(path: Path):
    img = Image.open(path).convert('RGB').resize((192, 128))
    a = np.asarray(img, dtype=np.float32) / 255.0
    gray = 0.2126*a[...,0] + 0.7152*a[...,1] + 0.0722*a[...,2]
    hsv = cv2.cvtColor((a*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    sat = hsv[...,1] / 255.0
    val = hsv[...,2] / 255.0
    f = []
    qs = [0,1,2,5,10,15,20,25,30,40,50,60,70,75,80,85,90,95,98,99,100]
    f += np.percentile(gray, qs).tolist()
    f += [gray.mean(), gray.std(), np.median(gray), gray.min(), gray.max(), gray.max()-gray.min()]
    hist, _ = np.histogram(gray, bins=64, range=(0,1), density=True)
    f += hist.tolist()
    for t in [0.01,0.02,0.03,0.05,0.08,0.1,0.15,0.2,0.25,0.3,0.4,0.5]:
        f.append(float((gray <= t).mean()))
    for t in [0.5,0.6,0.7,0.75,0.8,0.85,0.9,0.92,0.95,0.97,0.99]:
        f.append(float((gray >= t).mean()))
    for c in range(3):
        ch = a[...,c]
        f += [ch.mean(), ch.std(), ch.min(), ch.max()] + np.percentile(ch,[1,5,10,25,50,75,90,95,99]).tolist()
    for ch in [sat, val]:
        f += [ch.mean(), ch.std(), ch.min(), ch.max()] + np.percentile(ch,[1,5,10,25,50,75,90,95,99]).tolist()
    h,w = gray.shape
    for gy in range(4):
        for gx in range(4):
            b = gray[gy*h//4:(gy+1)*h//4, gx*w//4:(gx+1)*w//4]
            f += [b.mean(), b.std()] + np.percentile(b,[10,50,90]).tolist()
    for gy in range(3):
        for gx in range(3):
            b = gray[gy*h//3:(gy+1)*h//3, gx*w//3:(gx+1)*w//3]
            f += [b.mean(), np.percentile(b,25), np.percentile(b,75)]
    gx = np.diff(gray, axis=1, append=gray[:,-1:])
    gy = np.diff(gray, axis=0, append=gray[-1:,:])
    grad = np.sqrt(gx*gx + gy*gy)
    lap = cv2.Laplacian((gray*255).astype(np.uint8), cv2.CV_32F)
    probs = hist/(hist.sum()+1e-12)
    entropy = float(-(probs*np.log(probs+1e-12)).sum())
    f += [grad.mean(), grad.std(), np.percentile(grad,90), np.percentile(grad,99), lap.var(), entropy]
    f += [
        gray.mean()/(gray.std()+1e-6),
        np.percentile(gray,90)-np.percentile(gray,10),
        np.percentile(gray,75)-np.percentile(gray,25),
        val.mean()-sat.mean(),
        float((gray < 0.1).mean() - (gray > 0.9).mean()),
    ]
    return np.asarray(f, dtype=np.float32)

X = np.vstack([features(p) for p,_ in train_items])
y = np.array([label for _,label in train_items])
Xt = np.vstack([features(p) for p in test_items])
print('Feature shapes:', X.shape, Xt.shape)

models = {
    'extra': ExtraTreesClassifier(n_estimators=1000, max_features=0.85, min_samples_leaf=1, class_weight='balanced', random_state=11, n_jobs=-1),
    'rf': RandomForestClassifier(n_estimators=700, max_features=0.8, min_samples_leaf=1, class_weight='balanced', random_state=12, n_jobs=-1),
    'hgb': HistGradientBoostingClassifier(max_iter=350, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=2.0, random_state=13),
    'svc': make_pipeline(StandardScaler(), SVC(C=12, gamma='scale', probability=True, class_weight='balanced', random_state=14)),
}

cv = StratifiedKFold(5, shuffle=True, random_state=42)
scores = {}
for name, model in models.items():
    s = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    scores[name] = s.mean()
    print(name, 'CV accuracy', s.mean(), '+/-', s.std())

weights = [max(scores[n]-0.33, 0.01) for n in ['extra','rf','hgb','svc']]
ensemble = VotingClassifier(
    estimators=[('extra',models['extra']),('rf',models['rf']),('hgb',models['hgb']),('svc',models['svc'])],
    voting='soft', weights=weights, n_jobs=-1
)
ensemble.fit(X,y)
pred = ensemble.predict(Xt).astype(int)

submission = pd.DataFrame({'id':[p.stem for p in test_items], 'label':pred})
submission = submission.sort_values('id').reset_index(drop=True)
submission.to_csv('submission_lighting.csv', index=False)
print(submission.head())
print('Class counts:', submission['label'].value_counts().sort_index().to_dict())
print('Saved submission_lighting.csv')
