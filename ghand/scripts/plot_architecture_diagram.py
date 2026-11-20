import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# ---------- Color palette ----------
COLOR_ACTIVE   = '#2E8B57'   # green - active path
COLOR_DISABLED = '#C0392B'   # red - disabled path
COLOR_NEUTRAL  = '#4A4A4A'   # dark grey - neutral boxes
COLOR_AUX      = '#7F8C8D'   # light grey - auxiliary/background
COLOR_BG       = '#F5F5F5'   # light background box

# ---------- Helper: rounded box ----------
def box(x, y, w, h, text, facecolor='white', edgecolor=COLOR_NEUTRAL,
        textcolor='black', fontsize=12, bold=False, linewidth=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.05,rounding_size=0.15",
                       linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, color=textcolor,
            fontweight='bold' if bold else 'normal', wrap=True)
    return (x + w/2, y + h/2)  # return center

# ---------- Helper: arrow ----------
def arrow(start, end, color=COLOR_NEUTRAL, style='-|>', lw=2,
          linestyle='-', alpha=1.0):
    a = FancyArrowPatch(start, end, arrowstyle=style,
                        mutation_scale=20, color=color,
                        linewidth=lw, linestyle=linestyle, alpha=alpha,
                        shrinkA=2, shrinkB=2)
    ax.add_patch(a)

# ============================================================
# TITLE
# ============================================================
ax.text(7, 7.6, "How Our Model Trains", ha='center', va='center',
        fontsize=18, fontweight='bold', color=COLOR_NEUTRAL)

# ============================================================
# LEFT: Training data
# ============================================================
data_center = box(0.3, 3.3, 2.2, 1.4,
                  "Training Data\n\n400,000 robot\ndemonstrations",
                  facecolor=COLOR_BG, edgecolor=COLOR_NEUTRAL,
                  fontsize=11)

# ============================================================
# CENTER: The Policy (neural network)
# ============================================================
policy_center = box(3.3, 3.3, 2.8, 1.4,
                    "Policy\n(Neural Network)\n\nDecides what\nactions to take",
                    facecolor='#EAF2F8', edgecolor='#2874A6',
                    fontsize=11, bold=False)

# ============================================================
# RIGHT TOP: Active path — BC loss
# ============================================================
bc_center = box(7.3, 4.9, 3.2, 1.6,
                "BEHAVIOR CLONING LOSS\n(ACTIVE)\n\nTrains the policy to\ncopy the demonstrations",
                facecolor='#D5F5E3', edgecolor=COLOR_ACTIVE,
                fontsize=11, linewidth=3)

# ============================================================
# RIGHT BOTTOM: Disabled path — Q-loss + MMD
# ============================================================
bear_center = box(7.3, 2.3, 3.2, 1.8,
                  "BEAR POLICY IMPROVEMENT\n(DISABLED)\n\nWould try to improve the\npolicy using critics +\nMMD constraint",
                  facecolor='#FADBD8', edgecolor=COLOR_DISABLED,
                  fontsize=11, linewidth=3)

# ============================================================
# FAR RIGHT: Actor loss box
# ============================================================
loss_center = box(11.2, 3.3, 2.5, 1.6,
                  "ACTOR LOSS\n\nWhat actually\nupdates the policy",
                  facecolor=COLOR_BG, edgecolor=COLOR_NEUTRAL,
                  fontsize=11, bold=True)

# ============================================================
# BOTTOM: Auxiliary networks (still trained, don't affect policy)
# ============================================================
aux_center = box(3.3, 0.5, 6.0, 1.3,
                 "Also trained in the background:\nCritics (4)   •   Behavior Model (CVAE)   •   Constraint Weight (α)\n\nThese do NOT influence the policy",
                 facecolor='#F2F3F4', edgecolor=COLOR_AUX,
                 fontsize=10, linewidth=1.5)

# ============================================================
# ARROWS
# ============================================================

# Data → Policy
arrow((2.5, 4.0), (3.3, 4.0), color=COLOR_NEUTRAL, lw=2.5)

# Policy → BC loss (ACTIVE, thick green)
arrow((6.1, 4.3), (7.3, 5.3), color=COLOR_ACTIVE, lw=3.5)

# Policy → BEAR improvement (DISABLED, red dashed)
arrow((6.1, 3.7), (7.3, 2.9), color=COLOR_DISABLED, lw=2.5, linestyle='--')

# BC loss → Actor loss (ACTIVE green)
arrow((10.5, 5.6), (11.3, 4.5), color=COLOR_ACTIVE, lw=3.5)

# BEAR → Actor loss (DISABLED, red dashed with X)
arrow((10.5, 2.9), (11.3, 3.7), color=COLOR_DISABLED, lw=2.5, linestyle='--')

# Draw red X over the disabled arrow
ax.plot([10.75, 11.05], [3.05, 3.45], color=COLOR_DISABLED, lw=3)
ax.plot([10.75, 11.05], [3.45, 3.05], color=COLOR_DISABLED, lw=3)

# Data → Aux networks (grey)
arrow((1.4, 3.3), (1.4, 1.8), color=COLOR_AUX, lw=1.5, linestyle=':')
arrow((1.4, 1.15), (3.3, 1.15), color=COLOR_AUX, lw=1.5, linestyle=':')

# ============================================================
# X-MARKS on disabled paths (extra visual cue)
# ============================================================
# Red X on the disabled actor path (already drawn above on arrow to loss)
# Add small red X near the disabled box
ax.text(7.9, 3.2, "✕", fontsize=28, color=COLOR_DISABLED,
        ha='center', va='center', fontweight='bold')

# ============================================================
# LEGEND
# ============================================================
legend_elements = [
    mpatches.Patch(facecolor='#D5F5E3', edgecolor=COLOR_ACTIVE,
                   label='Active training path'),
    mpatches.Patch(facecolor='#FADBD8', edgecolor=COLOR_DISABLED,
                   label='Disabled (would harm performance)'),
    mpatches.Patch(facecolor='#F2F3F4', edgecolor=COLOR_AUX,
                   label='Trained but does not affect policy'),
]
ax.legend(handles=legend_elements, loc='lower right',
          fontsize=10, frameon=True, bbox_to_anchor=(1.0, 0.0))

# ============================================================
# CAPTION
# ============================================================
ax.text(7, -0.15,
        "Transparency note: we keep the full BEAR architecture, but disable the parts "
        "that destabilised training.\nThe policy is trained purely by Behavior Cloning.",
        ha='center', va='top', fontsize=10, style='italic', color=COLOR_NEUTRAL)

plt.tight_layout()
plt.savefig('bear_architecture_diagram.png', dpi=200, bbox_inches='tight',
            facecolor='white')
