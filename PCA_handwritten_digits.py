import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# how many images to keep per digit (None = use the whole dataset)
PER_DIGIT = 20

plt.rcParams.update({'figure.dpi': 110, 'axes.titlesize': 11, 'axes.grid': False})

digits = load_digits()
if PER_DIGIT is None:
    keep = np.arange(len(digits.target))
else:
    keep = np.concatenate([np.where(digits.target == d)[0][:PER_DIGIT] for d in range(10)])

X = digits.data[keep]        # (n, 64) data matrix, one image per row
imgs = digits.images[keep]   # (n, 8, 8)
y = digits.target[keep]
n, p = X.shape

print(f"data matrix X: {n} images x {p} pixels")
print(f"images per digit: {np.bincount(y)}")


# =========================================================
# Lab 1: images as vectors, class means, variance
# =========================================================
print("\n--- Lab 1 ---")

# a single image is an 8x8 array; flattened it is a vector in R^64
print("one image:", imgs[0].shape, "-> vector:", X[0].shape)

# collect samples, means and deviations first so the colour scale fits the data
rows = []
for d in range(10):
    flat = X[y == d]
    mean_flat = flat.mean(axis=0)
    best = np.argmin(np.linalg.norm(flat - mean_flat, axis=1))  # closest to class mean
    sample = imgs[y == d][best]
    mean_img = mean_flat.reshape(8, 8)
    rows.append((sample, mean_img, sample - mean_img))
vmax = max(np.abs(r[2]).max() for r in rows)

row_names = ["Closest sample\nto the mean", "Class mean", "Sample minus\nclass mean"]
fig, axes = plt.subplots(3, 10, figsize=(14, 5.6), constrained_layout=True)
for d, (sample, mean_img, dev) in enumerate(rows):
    axes[0, d].imshow(sample, cmap='gray', vmin=0, vmax=16)
    axes[1, d].imshow(mean_img, cmap='gray', vmin=0, vmax=16)
    im_dev = axes[2, d].imshow(dev, cmap='coolwarm', vmin=-vmax, vmax=vmax)
    axes[0, d].set_title(f"{d}", fontsize=13, fontweight='bold')

for r in range(3):
    for d in range(10):
        axes[r, d].set_xticks([])
        axes[r, d].set_yticks([])
        for spine in axes[r, d].spines.values():
            spine.set_visible(False)
    axes[r, 0].set_ylabel(row_names[r], fontsize=11, fontweight='bold', rotation=0,
                          ha='right', va='center', labelpad=12)

fig.colorbar(im_dev, ax=axes[2, :], shrink=0.9, pad=0.01, aspect=30, label='pixel difference')
fig.suptitle("Representative samples, class means and deviations", fontsize=14, fontweight='bold')
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(X.var(axis=0).reshape(8, 8), cmap='hot')
plt.colorbar(label='Variance')
plt.title("Variance of each pixel")
plt.axis('off')
plt.show()

plt.figure(figsize=(12, 3.5))
sevens = np.where(y == 7)[0][:10]
for i, idx in enumerate(sevens):
    plt.subplot(1, len(sevens), i + 1)
    plt.imshow(imgs[idx], cmap='gray')
    plt.title(f"#{i + 1}", fontsize=9)
    plt.axis('off')
plt.suptitle("Different handwriting styles for the digit 7", fontsize=12)
plt.show()


# =========================================================
# Lab 2: centering, rank, dependence between pixels
# =========================================================
print("\n--- Lab 2 ---")

mean_vec = X.mean(axis=0)
Xc = X - mean_vec
print("column means of Xc are ~0:", np.allclose(Xc.mean(axis=0), 0))

rank_X = np.linalg.matrix_rank(X)
rank_Xc = np.linalg.matrix_rank(Xc)
print(f"rank(X)  = {rank_X} / {p}")
print(f"rank(Xc) = {rank_Xc} / {p}")

# pixels that never change are all-zero columns after centering,
# so they make the columns linearly dependent and lower the rank
const_pixels = np.where(X.std(axis=0) == 0)[0]
print(f"constant pixels ({len(const_pixels)}): {const_pixels.tolist()}")

cov = Xc.T @ Xc / (n - 1)
print("matches np.cov:", np.allclose(cov, np.cov(X, rowvar=False)))

# correlation between pixels (constant pixels are left out, they have no variance)
live = np.setdiff1d(np.arange(p), const_pixels)
corr = np.corrcoef(X[:, live], rowvar=False)

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
cmax = np.abs(cov).max()
im0 = ax[0].imshow(cov, cmap='coolwarm', vmin=-cmax, vmax=cmax)
ax[0].set_title("Covariance matrix (64 x 64)")
ax[0].set_xlabel("pixel index")
ax[0].set_ylabel("pixel index")
fig.colorbar(im0, ax=ax[0], fraction=0.046)
im1 = ax[1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
ax[1].set_title(f"Correlation ({len(live)} non-constant pixels)")
fig.colorbar(im1, ax=ax[1], fraction=0.046)
plt.tight_layout()
plt.show()

# neighbouring pixels are strongly correlated, which is why the data
# can be compressed so well
off_diag = corr[~np.eye(len(live), dtype=bool)]
print(f"mean |correlation| between different pixels: {np.abs(off_diag).mean():.3f}")


# =========================================================
# Lab 3: eigenvectors of the covariance matrix, PCA, 2D projection
# =========================================================
print("\n--- Lab 3 ---")

# cov is symmetric, so eigh gives real eigenvalues (ascending order)
vals, vecs = np.linalg.eigh(cov)
vals = np.clip(vals[::-1], 0, None)   # tiny negative values are rounding noise
vecs = vecs[:, ::-1]

# check the definition: C v = lambda v
print("C v = lambda v holds:", np.allclose(cov @ vecs, vecs * vals))

explained = vals / vals.sum()
cum_var = np.cumsum(explained)
k90 = int(np.argmax(cum_var >= 0.90)) + 1
print(f"PC1 explains {explained[0]*100:.1f}%, PC2 explains {explained[1]*100:.1f}%")
print(f"components needed for 90% of the variance: {k90}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
ax[0].bar(range(1, 21), explained[:20] * 100)
ax[0].set_title("Variance explained by the first 20 components")
ax[0].set_xlabel("component")
ax[0].set_ylabel("variance (%)")
ax[1].plot(range(1, p + 1), cum_var * 100, marker='o', markersize=3)
ax[1].axhline(90, color='r', linestyle='--', label='90%')
ax[1].axvline(k90, color='gray', linestyle=':')
ax[1].set_title("Cumulative explained variance")
ax[1].set_xlabel("number of components")
ax[1].set_ylabel("cumulative variance (%)")
ax[1].grid(alpha=0.5)
ax[1].legend()
plt.tight_layout()
plt.show()

# the first eigenvectors, shown as 8x8 images
fig, axes = plt.subplots(1, 8, figsize=(15, 2.8), constrained_layout=True)
for i in range(8):
    m = np.abs(vecs[:, i]).max()
    axes[i].imshow(vecs[:, i].reshape(8, 8), cmap='coolwarm', vmin=-m, vmax=m)
    axes[i].set_title(f"PC{i + 1}", fontsize=10)
    axes[i].axis('off')
fig.suptitle("First eigenvectors (principal directions)", fontsize=13, fontweight='bold')
plt.show()

# 2D projection
Z2 = Xc @ vecs[:, :2]
plt.figure(figsize=(10, 8))
sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=y, cmap='tab10', alpha=0.8, s=35)
plt.colorbar(sc, label='digit')
plt.title("Images projected onto PC1 and PC2")
plt.xlabel(f"PC1 ({explained[0]*100:.1f}% of variance)")
plt.ylabel(f"PC2 ({explained[1]*100:.1f}% of variance)")
plt.grid(alpha=0.5)
plt.show()


# =========================================================
# Lab 4: quadratic forms, SVD, reconstruction error
# =========================================================
print("\n--- Lab 4 ---")

# quadratic form: the variance of the data along a unit vector v is v^T C v
def quad_form(v):
    return v @ cov @ v

v1 = vecs[:, 0]
rng = np.random.default_rng(0)
r = rng.normal(size=p)
r /= np.linalg.norm(r)
print(f"v1^T C v1        = {quad_form(v1):.4f}  (lambda_1 = {vals[0]:.4f})")
print(f"variance of Xc@v1 = {np.var(Xc @ v1, ddof=1):.4f}")
print(f"random unit vector: {quad_form(r):.4f}  (never above lambda_1)")

# SVD of the centered matrix and its link with PCA
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
print("eigenvalues of C == S^2/(n-1):", np.allclose(vals[:len(S)], S**2 / (n - 1)))
print("right singular vectors == eigenvectors (up to sign):",
      np.allclose(np.abs(Vt[:10]), np.abs(vecs[:, :10].T), atol=1e-6))

def svd_recon(k):
    return (U[:, :k] * S[:k]) @ Vt[:k] + mean_vec

def pca_recon(k):
    W = vecs[:, :k]
    return (Xc @ W) @ W.T + mean_vec

print("SVD and PCA reconstructions agree:",
      all(np.allclose(svd_recon(k), pca_recon(k)) for k in (1, 4, 16, 32)))

# reconstruct a few digits with different numbers of components
ks = [1, 4, 16, 32, 64]
show_digits = [7, 3, 9]
fig, axes = plt.subplots(len(show_digits), len(ks) + 1, figsize=(13, 6.5), constrained_layout=True)
print("\ndigit |  k | MSE")
for row, d in enumerate(show_digits):
    idx = np.where(y == d)[0][0]
    axes[row, 0].imshow(imgs[idx], cmap='gray', vmin=0, vmax=16)
    axes[row, 0].set_title(f"original {d}", fontsize=10, fontweight='bold')
    axes[row, 0].axis('off')
    for col, k in enumerate(ks):
        rec = svd_recon(k)[idx]
        mse = np.mean((X[idx] - rec) ** 2)
        print(f"  {d}   | {k:2d} | {mse:.4f}")
        axes[row, col + 1].imshow(rec.reshape(8, 8), cmap='gray', vmin=0, vmax=16)
        axes[row, col + 1].set_title(f"k = {k}\nMSE {mse:.2f}", fontsize=9)
        axes[row, col + 1].axis('off')
fig.suptitle("Reconstruction with k singular values", fontsize=13, fontweight='bold')
plt.show()

# average error over all images, computed directly and with Eckart-Young
k_range = [1, 2, 4, 8, 16, 24, 32, 48, 64]
direct = [np.mean((X - svd_recon(k)) ** 2) for k in k_range]
theory = [np.sum(S[k:] ** 2) / X.size for k in k_range]
print("\nk  | average MSE")
for k, e in zip(k_range, direct):
    print(f"{k:2d} | {e:.4f}")
print("direct MSE matches Eckart-Young formula:", np.allclose(direct, theory))

fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
ax.plot(k_range, direct, marker='o', color='purple', linewidth=2, label='measured error')
ax.scatter(k_range, theory, marker='x', s=70, color='orange', zorder=3,
           label='Eckart-Young formula')
ax.set_title("Average reconstruction error vs. number of components",
             fontsize=12, fontweight='bold')
ax.set_xlabel("k (components kept)")
ax.set_ylabel("average MSE per pixel")
ax.set_xticks(k_range)
ax.grid(alpha=0.4)
ax.legend()
plt.show()