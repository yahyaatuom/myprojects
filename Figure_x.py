import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Initialize canvas with standard publication aspect ratio
fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
ax.set_xlim(0, 11)
ax.set_ylim(0, 5)
ax.axis("off")

# Define reusable layout styles
box_style = dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#495057", lw=1.5)
formula_style = dict(size=10, fontname="DejaVu Sans", weight="medium", color="#212529")
title_style = dict(size=11, fontname="DejaVu Sans", weight="bold", color="#1a1d20")

# -------------------------------------------------------------------------
# STAGE 1: Contrastive Extraction
# -------------------------------------------------------------------------
ax.text(0.5, 4.2, "Phase 1: Contrastive Extraction", **title_style)
rect1 = patches.FancyBboxPatch((0.5, 1.2), 2.5, 2.5, boxstyle="round,pad=0.1", 
                               facecolor="#e8f0fe", edgecolor="#1a73e8", lw=1.5)
ax.add_patch(rect1)

ax.text(0.7, 3.2, "Prompt Matrix Pairs:\n  • Base Context ($P_{base}$)\n  • Perturbed Context ($P_{pert}$)", size=9, verticalalignment="top")
ax.text(0.7, 2.0, r"$\Delta \mathbf{h}_L = \mathbf{h}_L^{base} - \mathbf{h}_L^{pert}$", **formula_style)
ax.text(0.7, 1.4, r"$\mathbf{v}_L = \Delta \mathbf{h}_L / \|\Delta \mathbf{h}_L\|_2$", **formula_style)

# Directional Arrow 1 -> 2
ax.annotate("", xy=(3.5, 2.45), xytext=(3.1, 2.45),
            xycoords="data", textcoords="data",
            arrowprops=dict(arrowstyle="->", lw=2, color="#495057"))
ax.text(3.15, 2.6, r"$\mathbf{v}_L$", size=10, weight="bold", color="#1a73e8")

# -------------------------------------------------------------------------
# STAGE 2: Orthogonal Complement Projection
# -------------------------------------------------------------------------
ax.text(3.9, 4.2, "Phase 2: Orthogonal Projection", **title_style)
rect2 = patches.FancyBboxPatch((3.9, 1.2), 3.0, 2.5, boxstyle="round,pad=0.1", 
                               facecolor="#fef7e0", edgecolor="#f9ab00", lw=1.5)
ax.add_patch(rect2)

ax.text(4.1, 3.2, "Residual Stream Interception:\nIntercept intermediate state $\\mathbf{h}_L^{(t+k)}$\nat forward runtime step.", size=9, verticalalignment="top")
ax.text(4.1, 2.0, r"$\mathbf{p} = \langle \mathbf{h}_L^{(t+k)}, \mathbf{v}_L \rangle \mathbf{v}_L$", **formula_style)
ax.text(4.1, 1.4, r"$\mathbf{h}_{orthogonal} = \mathbf{h}_L^{(t+k)} - \mathbf{p}$", **formula_style)

# Directional Arrow 2 -> 3
ax.annotate("", xy=(7.4, 2.45), xytext=(7.0, 2.45),
            xycoords="data", textcoords="data",
            arrowprops=dict(arrowstyle="->", lw=2, color="#495057"))
ax.text(7.03, 2.6, r"$\mathbf{h}_{orth}$", size=9, weight="bold", color="#f9ab00")

# -------------------------------------------------------------------------
# STAGE 3: Scaled Injection
# -------------------------------------------------------------------------
ax.text(7.8, 4.2, "Phase 3: Scaled Injection", **title_style)
rect3 = patches.FancyBboxPatch((7.8, 1.2), 2.7, 2.5, boxstyle="round,pad=0.1", 
                               facecolor="#e6f4ea", edgecolor="#137333", lw=1.5)
ax.add_patch(rect3)

ax.text(8.0, 3.2, "Manifold Steering Execution:\nApply policy coefficient $\\alpha$\nalong targeted axis.", size=9, verticalalignment="top")
ax.text(8.0, 1.7, r"$\mathbf{h}_{L, steered}^{(t+k)} = \mathbf{h}_{orthogonal} + \alpha \mathbf{v}_L$", **formula_style)

# Final Output Arrow
ax.annotate("", xy=(11.0, 2.45), xytext=(10.6, 2.45),
            xycoords="data", textcoords="data",
            arrowprops=dict(arrowstyle="->", lw=2, color="#137333"))
ax.text(10.65, 2.7, "Modified\nLogits", size=8, weight="bold", color="#137333")

plt.tight_layout()
plt.savefig("figure_2_pipeline_diagram.png", bbox_inches="tight", dpi=300)
plt.show()