"""
Flow Diagram — Python Code Generator + Test Case Generator
Architecture: LangGraph + Groq (SimpleAgent pattern)
Run: python3 flow_diagram.py  →  saves flow_diagram.png
"""

import subprocess, sys

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib"])
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch


# ── Helpers ───────────────────────────────────────────────────────────────────

def draw_box(ax, x, y, w, h, label, sublabel="", color="#4A90D9",
             text_color="white", radius=0.25, fontsize=9):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=color, edgecolor="#2C3E50", linewidth=1.5, zorder=3,
    )
    ax.add_patch(box)
    offset = 0.13 if sublabel else 0
    ax.text(x, y + offset, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=text_color, zorder=4)
    if sublabel:
        ax.text(x, y - 0.17, sublabel, ha="center", va="center",
                fontsize=7, color=text_color, zorder=4, style="italic")


def straight_arrow(ax, x1, y1, x2, y2, label="", label_dx=0.12,
                   color="#2C3E50", lw=1.6):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw), zorder=2)
    if label:
        ax.text((x1 + x2) / 2 + label_dx, (y1 + y2) / 2,
                label, fontsize=7.5, color=color, fontweight="bold",
                va="center", zorder=5)


def curved_arrow(ax, x1, y1, x2, y2, label="", rad=0.35,
                 color="#C0392B", lw=1.6, label_offset=(0, 0)):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"), zorder=2)
    if label:
        ax.text((x1 + x2) / 2 + label_offset[0],
                (y1 + y2) / 2 + label_offset[1],
                label, fontsize=7.5, color=color, fontweight="bold",
                va="center", zorder=5)


# ── Canvas ────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(13, 11))
ax.set_xlim(0, 13)
ax.set_ylim(0, 11)
ax.axis("off")
ax.set_facecolor("#F4F6F8")
fig.patch.set_facecolor("#F4F6F8")

# Title
ax.text(6.5, 10.65, "Python Code Generator + Test Case Generator",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#2C3E50")
ax.text(6.5, 10.3, "LangGraph  •  SimpleAgent Pattern  •  Groq  •  Scaler x IITR Hackathon",
        ha="center", va="center", fontsize=8.5, color="#7F8C8D")

# ── Nodes (top → bottom, centred at x=6.5) ───────────────────────────────────

# 1. User
draw_box(ax, 6.5, 9.7, 3.2, 0.5, "User Request",
         sublabel="Natural language description", color="#BDC3C7", text_color="#2C3E50")

# 2. Gradio
draw_box(ax, 6.5, 8.75, 3.4, 0.55, "Gradio Chat Interface",
         sublabel="Chat UI  •  localhost:7860", color="#8E44AD")

# 3. Requirements Extractor  (LangGraph node — plain LLM call)
draw_box(ax, 6.5, 7.7, 3.6, 0.65, "Requirements Extractor",
         sublabel="LangGraph Node  •  LLM extracts structured spec", color="#E67E22")

# 4. Code Generator Agent  (SimpleAgent)
draw_box(ax, 6.5, 6.5, 3.6, 0.75, "Code Generator Agent",
         sublabel="SimpleAgent  •  No tools  •  Generates Python code", color="#27AE60")

# 5. Test Generator Agent  (SimpleAgent)
draw_box(ax, 6.5, 5.3, 3.6, 0.75, "Test Generator Agent",
         sublabel="SimpleAgent  •  No tools  •  Writes pytest test cases", color="#2980B9")

# 6. Code Executor  (LangGraph node using RunCode Tool)
draw_box(ax, 6.5, 4.0, 3.6, 0.75, "Code Executor Node",
         sublabel="Uses  RunCode Tool  •  Subprocess sandbox", color="#8E44AD")

# RunCode Tool badge (small box hanging off the executor)
draw_box(ax, 10.5, 4.0, 2.2, 0.5, "RunCode Tool",
         sublabel="LangChain Tool", color="#6C3483", fontsize=8)

# 7. Response Node
draw_box(ax, 6.5, 2.7, 3.6, 0.65, "Response Node",
         sublabel="Formats code + tests + execution results", color="#2C3E50")

# 8. Final output
draw_box(ax, 6.5, 1.7, 3.2, 0.5, "Final Response to User",
         color="#BDC3C7", text_color="#2C3E50")

# ── Arrows ────────────────────────────────────────────────────────────────────

straight_arrow(ax, 6.5, 9.45, 6.5, 9.03)              # user → gradio
straight_arrow(ax, 6.5, 8.48, 6.5, 8.03)              # gradio → requirements
straight_arrow(ax, 6.5, 7.37, 6.5, 6.88)              # requirements → code gen
straight_arrow(ax, 6.5, 6.12, 6.5, 5.68)              # code gen → test gen
straight_arrow(ax, 6.5, 4.93, 6.5, 4.38)              # test gen → executor
straight_arrow(ax, 6.5, 3.63, 6.5, 3.03,              # executor → response (PASS)
               label="PASS", label_dx=0.15, color="#27AE60")
straight_arrow(ax, 6.5, 2.37, 6.5, 1.95)              # response → final output

# RunCode Tool connector (dashed line from executor to tool badge)
ax.annotate("", xy=(9.38, 4.0), xytext=(8.3, 4.0),
            arrowprops=dict(arrowstyle="-|>", color="#6C3483", lw=1.4,
                            linestyle="dashed"), zorder=2)
ax.text(8.84, 4.18, "calls", fontsize=7, color="#6C3483", ha="center", zorder=5)

# Retry loop: executor → code gen (FAIL, curved, right side)
curved_arrow(ax, 8.3, 4.0, 8.3, 6.5,
             label="FAIL\n(retry < 3)", rad=-0.0,
             color="#C0392B", label_offset=(0.55, -1.2))

# vertical line on right side for retry
ax.plot([8.3, 8.3], [4.0, 6.5], color="#C0392B", lw=1.6, zorder=1, linestyle="--")
ax.annotate("", xy=(8.3, 6.5), xytext=(8.3, 4.0),
            arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.6), zorder=2)

# Max retries label on executor → response path
ax.text(5.2, 3.35, "max retries\nreached", fontsize=7, color="#C0392B",
        ha="center", va="center", style="italic", zorder=5)

# ── Legend ────────────────────────────────────────────────────────────────────

legend_items = [
    mpatches.Patch(color="#E67E22", label="LangGraph Node (plain LLM)"),
    mpatches.Patch(color="#27AE60", label="Code Generator Agent (SimpleAgent)"),
    mpatches.Patch(color="#2980B9", label="Test Generator Agent (SimpleAgent)"),
    mpatches.Patch(color="#8E44AD", label="Code Executor Node"),
    mpatches.Patch(color="#6C3483", label="RunCode Tool (LangChain Tool)"),
    mpatches.Patch(color="#2C3E50", label="Response Node"),
    mpatches.Patch(color="#C0392B", label="Retry loop (max 3)"),
]
ax.legend(handles=legend_items, loc="lower left", fontsize=7.5,
          framealpha=0.95, ncol=2, bbox_to_anchor=(0.0, 0.0))

plt.tight_layout()
plt.savefig("flow_diagram.png", dpi=150, bbox_inches="tight", facecolor="#F4F6F8")
print("Saved: flow_diagram.png")
plt.show()
