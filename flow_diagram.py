"""
Flow Diagram Generator for the Multi-Agent NL-to-Code System
Run: python3 flow_diagram.py
Saves: flow_diagram.png
"""

import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
except ImportError:
    install("matplotlib")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def draw_box(ax, x, y, w, h, label, sublabel="", color="#4A90D9", text_color="white", radius=0.3):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=color, edgecolor="#2C3E50", linewidth=1.5, zorder=3
    )
    ax.add_patch(box)
    ax.text(x, y + (0.12 if sublabel else 0), label,
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=text_color, zorder=4)
    if sublabel:
        ax.text(x, y - 0.18, sublabel,
                ha="center", va="center", fontsize=7, color=text_color, zorder=4, style="italic")


def arrow(ax, x1, y1, x2, y2, label="", color="#2C3E50"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5), zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.08, my, label, fontsize=7, color="#555", va="center", zorder=5)


fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")
ax.set_facecolor("#F8F9FA")
fig.patch.set_facecolor("#F8F9FA")

ax.text(7, 9.6, "Multi-Agent Natural Language to Code System",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#2C3E50")
ax.text(7, 9.25, "LangGraph + Claude (Anthropic)  •  Scaler x IITR Hackathon",
        ha="center", va="center", fontsize=9, color="#7F8C8D")

# ── Nodes ──────────────────────────────────────────────────────────────────────

# User
draw_box(ax, 7, 8.5, 2.8, 0.55, "User / Natural Language Request", color="#ECF0F1", text_color="#2C3E50")

# Gradio UI
draw_box(ax, 7, 7.6, 3.0, 0.55, "Gradio Chat Interface", sublabel="(+10 bonus points)", color="#8E44AD")

# Router Agent
draw_box(ax, 7, 6.6, 3.2, 0.6, "Router Agent", sublabel="Classify: Python | SQL  &  task type", color="#E74C3C")

# Requirements Agent
draw_box(ax, 7, 5.6, 3.2, 0.6, "Requirements Agent", sublabel="Extract: name, params, edge cases", color="#E67E22")

# Python Code Agent
draw_box(ax, 3.5, 4.4, 2.8, 0.65, "Python Code Agent", sublabel="Generate Python function/class", color="#27AE60")

# SQL Code Agent
draw_box(ax, 10.5, 4.4, 2.8, 0.65, "SQL Code Agent", sublabel="Generate SQL query + wrapper", color="#16A085")

# Test Gen Agent
draw_box(ax, 7, 3.3, 3.2, 0.6, "Test Generation Agent", sublabel="pytest unit tests + assert cases", color="#2980B9")

# Code Executor Agent
draw_box(ax, 7, 2.2, 3.2, 0.6, "Code Executor Agent", sublabel="Run code & tests in subprocess sandbox", color="#8E44AD")

# Refinement Agent
draw_box(ax, 11.5, 2.2, 2.6, 0.6, "Refinement Agent", sublabel="Fix code on failure (max 2x)", color="#C0392B")

# Response Agent
draw_box(ax, 7, 1.1, 3.2, 0.6, "Response Agent", sublabel="Format final output with results", color="#2C3E50")

# Final Output
draw_box(ax, 7, 0.25, 3.5, 0.4, "Final Response to User", color="#ECF0F1", text_color="#2C3E50")

# ── Arrows ─────────────────────────────────────────────────────────────────────

arrow(ax, 7, 8.22, 7, 7.88)                          # user → gradio
arrow(ax, 7, 7.32, 7, 6.9)                            # gradio → router
arrow(ax, 7, 6.3, 7, 5.9)                             # router → requirements

# requirements → python (fork left)
arrow(ax, 5.4, 5.3, 4.1, 4.73, "Python", "#27AE60")
# requirements → sql (fork right)
arrow(ax, 8.6, 5.3, 9.9, 4.73, "SQL", "#16A085")

# python + sql → test gen
arrow(ax, 3.7, 4.07, 5.4, 3.6, "", "#27AE60")
arrow(ax, 10.3, 4.07, 8.6, 3.6, "", "#16A085")

arrow(ax, 7, 3.0, 7, 2.5)                             # test gen → executor
arrow(ax, 7, 1.9, 7, 1.4)                             # executor → response (pass)

# executor → refinement (fail)
ax.annotate("", xy=(11.5, 2.5), xytext=(8.6, 2.2),
            arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.5,
                            connectionstyle="arc3,rad=-0.3"), zorder=2)
ax.text(10.0, 2.55, "FAIL", fontsize=7.5, color="#C0392B", fontweight="bold")

# refinement → test gen (retry loop)
ax.annotate("", xy=(8.6, 3.3), xytext=(11.5, 2.5),
            arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.5,
                            connectionstyle="arc3,rad=-0.3"), zorder=2)
ax.text(10.6, 3.1, "retry", fontsize=7, color="#C0392B", style="italic")

# executor → response pass label
ax.text(7.45, 1.65, "PASS", fontsize=7.5, color="#27AE60", fontweight="bold")

arrow(ax, 7, 0.8, 7, 0.45)                            # response → output

# ── Legend ─────────────────────────────────────────────────────────────────────

legend_items = [
    mpatches.Patch(color="#E74C3C", label="Router Agent"),
    mpatches.Patch(color="#E67E22", label="Requirements Agent"),
    mpatches.Patch(color="#27AE60", label="Python Code Agent"),
    mpatches.Patch(color="#16A085", label="SQL Code Agent"),
    mpatches.Patch(color="#2980B9", label="Test Generation Agent"),
    mpatches.Patch(color="#8E44AD", label="Code Executor / Gradio"),
    mpatches.Patch(color="#C0392B", label="Refinement Agent"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=7.5,
          framealpha=0.9, ncol=2, bbox_to_anchor=(0.01, 0.01))

plt.tight_layout()
plt.savefig("flow_diagram.png", dpi=150, bbox_inches="tight", facecolor="#F8F9FA")
print("Saved: flow_diagram.png")
plt.show()
