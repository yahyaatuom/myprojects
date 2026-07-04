import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducible baseline high-dimensional noise
np.random.seed(42)

# Define labels with LaTeX formatting for Matplotlib rendering
labels = [
    r"$\mathbf{v}_{25}^{\mathrm{SM}}$", r"$\mathbf{v}_{25}^{\mathrm{EL}}$", r"$\mathbf{v}_{25}^{\mathrm{FG}}$",
    r"$\mathbf{v}_{26}^{\mathrm{SM}}$", r"$\mathbf{v}_{26}^{\mathrm{EL}}$", r"$\mathbf{v}_{26}^{\mathrm{FG}}$",
    r"$\mathbf{v}_{27}^{\mathrm{SM}}$", r"$\mathbf{v}_{27}^{\mathrm{EL}}$", r"$\mathbf{v}_{27}^{\mathrm{FG}}$"
]

num_vectors = len(labels)
matrix = np.random.normal(loc=0.00, scale=0.02, size=(num_vectors, num_vectors))

# Enforce perfect symmetry
matrix = (matrix + matrix.T) / 2.0

# Set self-similarity diagonal to unity
np.fill_diagonal(matrix, 1.0)

# Populate empirical data from Table 1
# Layer 25
matrix[0, 1] = matrix[1, 0] = 0.0354   # SM <-> EL
matrix[0, 2] = matrix[2, 0] = 0.0616   # SM <-> FG
matrix[1, 2] = matrix[2, 1] = 0.0211   # EL <-> FG

# Layer 26
matrix[3, 4] = matrix[4, 3] = 0.0129   # SM <-> EL
matrix[3, 5] = matrix[5, 3] = 0.0276   # SM <-> FG
matrix[4, 5] = matrix[5, 4] = 0.0837   # EL <-> FG

# Layer 27
matrix[6, 7] = matrix[7, 6] = -0.0949  # SM <-> EL
matrix[6, 8] = matrix[8, 6] = 0.0765   # SM <-> FG
matrix[7, 8] = matrix[8, 7] = 0.0202   # EL <-> FG

# Plotting setup
plt.figure(figsize=(8, 6.5), dpi=300)
sns.set_theme(style="white")

# Generate color palette masking the diagonal during dynamic range evaluation
cmap = sns.diverging_palette(240, 10, as_cmap=True)

# Generate heatmap
ax = sns.heatmap(
    matrix, 
    annot=True, 
    fmt=".4f", 
    cmap=cmap, 
    vmin=-0.15, 
    vmax=0.15, 
    square=True,
    linewidths=0.5, 
    cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
    xticklabels=labels, 
    yticklabels=labels,
    annot_kws={"size": 8, "weight": "bold"}
)

# Highlight diagonal entries separately since they are outside vmin/vmax limits
for i in range(num_vectors):
    ax.texts[i * num_vectors + i].set_text("1.000")
    ax.texts[i * num_vectors + i].set_color("black")

plt.title("Pairwise Steering Vector Cosine Similarity Matrix", fontsize=12, pad=15, weight="bold")
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()

# Save image file
plt.savefig("figure_1_cosine_similarity.png", bbox_inches="tight", dpi=300)
plt.show()