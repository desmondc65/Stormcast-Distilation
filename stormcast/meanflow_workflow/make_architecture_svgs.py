"""Generate the two detailed architecture figures (SVG) for the MeanFlow workflow.

Figure 1: regression_model.svg  — Stage 1 StormCastUNet F_theta (frozen mean).
Figure 2: meanflow_model.svg    — Stage 2 MeanFlowPrecond u_theta (trainable head).

All layer names, channel counts and parameter numbers are the measured values
from instantiating the canonical 192x96 networks (see
../meanflow_architecture.md, "Full layer inventory"). Regenerate PNGs with
render_pngs.sh in this directory.
"""

import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# tiny SVG builder
# ----------------------------------------------------------------------------


class SVG:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.body = []

    def rect(self, x, y, w, h, fill, stroke, rx=8, sw=1.6, dash=None, opacity=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.body.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d} '
            f'opacity="{opacity}"/>'
        )

    def text(self, x, y, s, size=11.5, fill="#263238", weight="normal",
             anchor="start", style="", family="Helvetica, Arial, sans-serif"):
        st = f' font-style="{style}"' if style else ""
        self.body.append(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
            f'fill="{fill}" font-weight="{weight}" text-anchor="{anchor}"{st}>{s}</text>'
        )

    def lines_in_box(self, x, y, w, lines, size=11.5, pad=9, lh=15.5,
                     fill="#263238", anchor="middle"):
        tx = x + w / 2 if anchor == "middle" else x + pad
        for i, ln in enumerate(lines):
            wt = "bold" if isinstance(ln, tuple) and ln[1] == "b" else "normal"
            s = ln[0] if isinstance(ln, tuple) else ln
            self.text(tx, y + pad + (i + 0.72) * lh, s, size=size, fill=fill,
                      weight=wt, anchor=anchor)

    def path(self, d, stroke, sw=1.6, dash=None, marker=None, fill="none"):
        m = f' marker-end="url(#{marker})"' if marker else ""
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.body.append(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}"{da}{m}/>'
        )

    def arrow(self, x1, y1, x2, y2, stroke, sw=1.6, dash=None, marker="arrDark"):
        self.path(f"M {x1} {y1} L {x2} {y2}", stroke, sw, dash, marker)

    def circle(self, cx, cy, r, fill, stroke, sw=1.6):
        self.body.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def render(self):
        defs = """<defs>
<marker id="arrDark" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#37474F"/></marker>
<marker id="arrGreen" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#1B5E20"/></marker>
<marker id="arrGrey" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#90A4AE"/></marker>
<marker id="arrAmber" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#F57F17"/></marker>
</defs>"""
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
            f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">\n{defs}\n'
            f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="#FFFFFF"/>\n'
            + "\n".join(self.body)
            + "\n</svg>\n"
        )


# ----------------------------------------------------------------------------
# shared U-Net drawing (5 levels + bottleneck), staircase-U layout
# ----------------------------------------------------------------------------

ENC_X0, DEC_X0 = 400, 1150     # L0 x of encoder box / decoder box left edge
STAIR = 35                     # staircase shift per level
BW_E, BW_D = 310, 330          # box widths
Y0, PITCH, BH = 235, 102, 76   # first level y, level pitch, level box height
RES = ["192×96", "96×48", "48×24", "24×12", "12×6"]


def enc_x(i):
    return ENC_X0 + STAIR * i


def dec_x(i):
    return DEC_X0 - STAIR * i


def lvl_y(i):
    return Y0 + PITCH * i


def draw_unet(S, pal, enc_l0_lines, enc_sums, dec_sums, arrmk):
    box_fill, box_stroke, txt = pal["fill"], pal["stroke"], pal["text"]

    enc_lines = [
        enc_l0_lines,
        ["↓2× UNetBlock 128→128 (378 k)",
         "UNetBlock 128→256 · then 256→256 ×3",
         ("Σ 5.37 M", "b")],
        ["↓2× UNetBlock 256→256 (1.38 M)",
         "UNetBlock 256→256 ×4 (1.31 M each)",
         ("Σ 6.63 M", "b")],
        ["↓2× UNetBlock 256→256 (1.38 M)",
         "UNetBlock 256→256 ×4 (1.31 M each)",
         ("Σ 6.63 M", "b")],
        ["↓2× UNetBlock 256→256 (1.38 M)",
         "UNetBlock 256→256 ×4 (1.31 M each)",
         ("Σ 6.63 M", "b")],
    ]
    dec_lines = [
        ["↑2× UNetBlock 256→256 (1.38 M)",
         "UNetBlock 384→128 ×1 · 256→128 ×4",
         "head: GroupNorm+SiLU → Conv 3×3 128→4 (0-init)",
         ("Σ 4.26 M", "b")],
        ["↑2× UNetBlock 256→256 (1.38 M)",
         "UNetBlock 512→256 ×4 · 384→256 ×1",
         ("Σ 11.22 M", "b")],
        ["↑2× UNetBlock 256→256 (1.38 M)",
         "UNetBlock 512→256 ×5 (2.03 M each)",
         ("Σ 11.55 M", "b")],
        ["↑2× UNetBlock 256→256 (1.38 M)",
         "UNetBlock 512→256 ×5 (2.03 M each)",
         ("Σ 11.55 M", "b")],
        ["from bottleneck ↓",
         "UNetBlock 512→256 ×5 (2.03 M each)",
         ("Σ 13.06 M incl. in0+in1", "b")],
    ]

    # column headers
    S.text(enc_x(0) + BW_E / 2, Y0 - 14, "ENCODER  enc.*", size=13,
           weight="bold", fill=box_stroke, anchor="middle")
    S.text(dec_x(0) + BW_D / 2, Y0 - 14, "DECODER  dec.*", size=13,
           weight="bold", fill=box_stroke, anchor="middle")

    for i in range(5):
        y = lvl_y(i)
        # encoder box
        S.rect(enc_x(i), y, BW_E, BH, box_fill, box_stroke)
        S.lines_in_box(enc_x(i), y, BW_E, enc_lines[i], size=11, fill=txt)
        # decoder box
        S.rect(dec_x(i), y, BW_D, BH, box_fill, box_stroke)
        S.lines_in_box(dec_x(i), y, BW_D, dec_lines[i], size=11, fill=txt)
        # channel tag above the encoder box (keeps the left flank clear)
        ch = "128 ch" if i == 0 else "256 ch"
        S.text(enc_x(i) + 2, y - 6, f"{RES[i]} · {ch}", size=10.5,
               fill="#78909C", anchor="start")
        # skip connection
        sx, dxx = enc_x(i) + BW_E, dec_x(i)
        my = y + BH / 2
        S.arrow(sx, my, dxx - 3, my, "#90A4AE", sw=1.4, dash="6 4",
                marker="arrGrey")
        midx = (sx + dxx) / 2
        S.text(midx, my - 7, f"{RES[i]}", size=10.5, fill="#78909C",
               anchor="middle", style="italic")
        S.text(midx, my + 14, "5 skips · concat", size=10.5,
               fill="#78909C", anchor="middle", style="italic")
        # inter-level arrows
        if i < 4:
            S.arrow(enc_x(i) + BW_E / 2, y + BH,
                    enc_x(i + 1) + BW_E / 2, lvl_y(i + 1),
                    box_stroke, marker=arrmk)
            S.arrow(dec_x(i + 1) + BW_D / 2, lvl_y(i + 1),
                    dec_x(i) + BW_D / 2, y + BH,
                    box_stroke, marker=arrmk)

    # bottleneck row
    by = lvl_y(4) + BH + 26
    bx, bw, bh = 610, 630, 60
    S.rect(bx, by, bw, bh, pal["bfill"], box_stroke, sw=2.0)
    S.lines_in_box(bx, by, bw, [
        ("in0: UNetBlock 256→256 + single-head self-attn "
         "(72 tokens) ⚡ 1.58 M", "b"),
        "in1: UNetBlock 256→256 · 1.31 M",
    ], size=11, fill=txt)
    S.text(bx + bw / 2, by + bh + 16,
           "SongUNet bottleneck · 12×6 · the only attention "
           "in the network (attn_resolutions=[] disables the rest)",
           size=10.5, fill="#78909C", anchor="middle", style="italic")
    # enc L4 -> bottleneck -> dec L4
    S.arrow(enc_x(4) + BW_E / 2, lvl_y(4) + BH,
            enc_x(4) + BW_E / 2, by, box_stroke, marker=arrmk)
    S.arrow(dec_x(4) + BW_D / 2, by,
            dec_x(4) + BW_D / 2, lvl_y(4) + BH, box_stroke, marker=arrmk)
    return by + bh


def legend_strip(S, y, emb_note, pal):
    S.rect(30, y, 1500, 64, "#FAFAFA", "#B0BEC5", rx=6, sw=1.2)
    S.text(45, y + 25,
           "UNetBlock (DDPM++ residual block, adaptive_scale=False):   "
           "x → GroupNorm(32, ε 1e-6) + SiLU → Conv 3×3 "
           "→ ⊕ emb-shift → GroupNorm + SiLU → Dropout 0.1 "
           "→ Conv 3×3 (init ×1e-5) → ⊕ skip "
           "(identity / 1×1 conv) → × 1/√2",
           size=11.5, weight="bold")
    S.text(45, y + 48, emb_note + "   ·   ↓2×/↑2× fold a "
           "[1,1] box-filter resample into conv0 + skip   ·   in0 "
           "additionally: → self-attn → × 1/√2",
           size=11.5, fill="#455A64")


# ----------------------------------------------------------------------------
# Figure 1 — regression StormCastUNet
# ----------------------------------------------------------------------------

def figure_regression():
    S = SVG(1560, 1010)
    pal = {"fill": "#ECEFF1", "stroke": "#546E7A", "text": "#263238",
           "bfill": "#CFD8DC"}

    S.text(30, 42, "Stage 1 · StormCastUNet Fθ — deterministic "
           "regression mean (frozen ❄ during MeanFlow training)",
           size=21, weight="bold", fill="#37474F")
    S.text(30, 66, "SongUNet backbone · 30 ch in → 4 ch out · "
           "80,726,404 params · 55 UNetBlocks · grid 192×96 "
           "· μ_t+1 = Fθ(M_t, S_t, I)", size=13, fill="#546E7A")

    # inputs
    inp = [
        ("M_t · HighRes state (64, 4, 192, 96)",
         "u10 · v10 · t2m · qpepre", "#1565C0", "#E3F2FD"),
        ("S_t · ERA5 background (64, 24, 192, 96)",
         "mslp t2m u10 v10 + q/t/u/v/z @ 4 levels", "#6A1B9A", "#F3E5F5"),
        ("I · invariants (64, 2, 192, 96)",
         "lsm · orog (broadcast)", "#00695C", "#E0F2F1"),
    ]
    for k, (t1, t2, stc, fll) in enumerate(inp):
        yb = 200 + 66 * k
        S.rect(30, yb, 250, 54, fll, stc, rx=6)
        S.lines_in_box(30, yb, 250, [(t1, "b"), t2], size=10.2)
        S.arrow(280, yb + 27, 306, 292, stc, sw=1.4)
    S.circle(315, 292, 11, "#FFFFFF", "#37474F")
    S.text(315, 296.5, "⊕", size=15, anchor="middle", weight="bold")
    S.text(397, 252, "(64, 30, 192, 96)", size=9.5, anchor="end",
           fill="#546E7A", style="italic")
    S.text(327, 318, "concat", size=9.5, anchor="start", fill="#546E7A")
    S.arrow(326, 292, ENC_X0 - 3, lvl_y(0) + BH / 2, "#37474F", sw=1.8)

    enc_l0 = [
        "Conv 3×3 30→128 (34,688 p) · + spatial_emb",
        "(1×128×192×96, 2.36 M) · UNetBlock 128→128 ×4",
        ("Σ 1.48 M + 2.36 M pos-embed", "b"),
    ]
    bot = draw_unet(S, pal, enc_l0, None, None, "arrDark")

    # output
    S.rect(1210, 90, 290, 56, "#ECEFF1", "#37474F", sw=2.0)
    S.lines_in_box(1210, 90, 290, [
        ("μ_t+1  (64, 4, 192, 96)", "b"),
        "deterministic mean · residual target R = M_t+1 − μ",
    ], size=11)
    S.arrow(dec_x(0) + BW_D - 40, lvl_y(0), 1370, 146 + 3, "#37474F", sw=1.8)

    # notes
    ny = 470
    S.rect(30, ny, 340, 178, "#FFFFFF", "#546E7A", rx=6, sw=1.3, dash="5 4")
    S.lines_in_box(30, ny, 340, [
        ("StormCastUNet specifics", "b"),
        "embedding_type='zero' — no map_noise /",
        "map_layer MLP; forward = SongUNet(x, 0).",
        "Every UNetBlock affine Linear(512→C) sees a",
        "zero vector → bias-only shift (no time cond.).",
        "Frozen ❄ in Stage 2 — runs under no_grad,",
        "provides μ_t+1 to the condition bundle c.",
        "Single self-attn: dec.12x12_in0 · 1 head.",
    ], size=11, anchor="start", pad=14)

    legend_strip(S, bot + 44,
                 "emb ≡ 0 here (zero embedding) → the ⊕ adds "
                 "only the affine bias", pal)

    S.text(30, bot + 44 + 92,
           "Σ params — encoder 26.73 M · decoder 51.64 M · "
           "spatial_emb 2.36 M · total 80,726,404 · measured by "
           "instantiation at (192, 96) · ModuleDict names use the "
           "y-resolution only (enc.192x192_* actually runs at 192×96)",
           size=11.5, fill="#546E7A")
    return S.render()


# ----------------------------------------------------------------------------
# Figure 2 — MeanFlow residual head
# ----------------------------------------------------------------------------

def figure_meanflow():
    S = SVG(1560, 1060)
    pal = {"fill": "#E8F5E9", "stroke": "#2E7D32", "text": "#1B3A1E",
           "bfill": "#C8E6C9"}

    S.text(30, 42, "Stage 2 · MeanFlowPrecond uθ — "
           "average-velocity residual head (trainable)",
           size=21, weight="bold", fill="#1B5E20")
    S.text(30, 66, "SongUNet backbone · 14 ch in → 4 ch out · "
           "81,040,772 params · 55 UNetBlocks · grid 192×96 "
           "· uθ(z_r, r, t, c) ≈ mean velocity over [r, t] in "
           "standardized residual space", size=13, fill="#2E7D32")

    # inputs
    S.rect(30, 190, 250, 54, "#E8F5E9", "#2E7D32", rx=6)
    S.lines_in_box(30, 190, 250, [
        ("z_r · flow state (64, 4, 192, 96)", "b"),
        "train: (1−r)·x₀ + r·x₁ · infer: z₀ ~ N(0, I)",
    ], size=10.2)
    S.arrow(280, 217, 306, 288, "#2E7D32", sw=1.4)

    S.rect(30, 258, 250, 84, "#E3F2FD", "#1565C0", rx=6)
    S.lines_in_box(30, 258, 250, [
        ("condition c (64, 10, 192, 96)", "b"),
        "M_t (4) · HighRes state at t",
        "μ_t+1 (4) ❄ from frozen Stage 1",
        "I (2) lsm, orog · no background S_t",
    ], size=10.2)
    S.arrow(280, 300, 306, 296, "#1565C0", sw=1.4)

    S.circle(315, 292, 11, "#FFFFFF", "#1B5E20")
    S.text(315, 296.5, "⊕", size=15, anchor="middle", weight="bold")
    S.text(397, 252, "(64, 14, 192, 96)", size=9.5, anchor="end",
           fill="#2E7D32", style="italic")
    S.text(327, 318, "concat", size=9.5, anchor="start", fill="#2E7D32")
    S.arrow(326, 292, ENC_X0 - 3, lvl_y(0) + BH / 2, "#1B5E20", sw=1.8)

    enc_l0 = [
        "Conv 3×3 14→128 (16,256 p) · + spatial_emb",
        "(1×128×192×96, 2.36 M) · UNetBlock 128→128 ×4",
        ("Σ 1.46 M + 2.36 M pos-embed", "b"),
    ]
    bot = draw_unet(S, pal, enc_l0, None, None, "arrGreen")

    # output
    S.rect(1210, 90, 290, 56, "#E8F5E9", "#1B5E20", sw=2.0)
    S.lines_in_box(1210, 90, 290, [
        ("uθ(z_r, r, t, c)  (64, 4, 192, 96)", "b"),
        "average velocity · r = t ⇒ FlowCast field",
    ], size=11)
    S.arrow(dec_x(0) + BW_D - 40, lvl_y(0), 1370, 146 + 3, "#1B5E20", sw=1.8)

    # ---- time-conditioning inset ----------------------------------------
    ix, iy, iw = 30, 400, 345
    S.rect(ix, iy, iw, 300, "#FFFDE7", "#F9A825", rx=8, sw=1.6)
    S.text(ix + iw / 2, iy + 22, "time conditioning (MeanFlow-specific)",
           size=12.5, weight="bold", fill="#E65100", anchor="middle")

    def tbox(y, lines, h=36):
        S.rect(ix + 18, y, iw - 36, h, "#FFF8E1", "#F57F17", rx=5, sw=1.2)
        S.lines_in_box(ix + 18, y, iw - 36, lines, size=10.2, pad=4,
                       lh=13.5, fill="#4E342E")
        return y + h

    y = tbox(iy + 34, [("r ×1000 → PositionalEmbedding(128)", "b"),
                       "f_i = 10000^(−i/63) · [cos | sin] → (128,)"])
    yb2 = tbox(y + 12, [("t−r → sin/cos, 16 log-freqs 1→1000", "b"),
                        "→ (32,)"])
    yb3 = tbox(yb2 + 12, [("map_augment: Linear 32→128 (no bias)", "b")],
               h=24)
    S.arrow(ix + iw / 2, y, ix + iw / 2, y + 12, "#F57F17", sw=1.3,
            marker="arrAmber")
    S.arrow(ix + iw / 2, yb2, ix + iw / 2, yb2 + 12, "#F57F17", sw=1.3,
            marker="arrAmber")
    # merge node
    my = yb3 + 22
    S.circle(ix + iw / 2, my, 9, "#FFFFFF", "#E65100")
    S.text(ix + iw / 2, my + 4, "⊕", size=13, anchor="middle",
           weight="bold", fill="#E65100")
    S.arrow(ix + iw / 2, yb3, ix + iw / 2, my - 9, "#F57F17", sw=1.3,
            marker="arrAmber")
    # elbow from box1 right side to merge node
    S.path(f"M {ix + iw - 18} {(iy + 34 + y) / 2} L {ix + iw - 6} "
           f"{(iy + 34 + y) / 2} L {ix + iw - 6} {my} L {ix + iw / 2 + 9} {my}",
           "#F57F17", sw=1.3, marker="arrAmber")
    y4 = tbox(my + 13, [("map_layer0: Linear 128→512 + SiLU", "b")], h=24)
    S.arrow(ix + iw / 2, my + 9, ix + iw / 2, my + 13, "#F57F17", sw=1.3)
    y5 = tbox(y4 + 10, [("map_layer1: Linear 512→512 + SiLU", "b")], h=24)
    S.arrow(ix + iw / 2, y4, ix + iw / 2, y4 + 10, "#F57F17", sw=1.3,
            marker="arrAmber")
    S.text(ix + iw / 2, y5 + 20, "emb (512,) · Σ mapping 0.33 M",
           size=11, weight="bold", fill="#E65100", anchor="middle")
    S.text(ix + iw / 2, y5 + 37,
           "→ shift-only FiLM: affine Linear(512→C) of all 55 UNetBlocks",
           size=10, fill="#E65100", anchor="middle", style="italic")
    # dashed broadcast arrow into the U (symbolic — reaches every block)
    S.arrow(ix + iw, 667, enc_x(4) - 4, 667, "#F57F17", sw=1.5,
            dash="6 4", marker="arrAmber")

    # small notes
    S.lines_in_box(30, 742, 360, [
        "EMA shadow (decay 0.999) = inference weights (ema_state.pt).",
        "Same SongUNet backbone as the FlowCast / EDM heads —",
        "only the objective and time embedding differ.",
    ], size=10.5, anchor="start", pad=0, lh=15, fill="#455A64")

    legend_strip(S, bot + 44,
                 "emb = 512-d time vector (amber inset) → shift-only FiLM",
                 pal)

    S.text(30, bot + 44 + 88,
           "sampler:  z_{i+1} = z_i + Δt · uθ(z_i, t_i, t_{i+1}, c)"
           "  over 1–2 segments of [0,1]  ·  R̂ = z₁ × "
           "σ_data (0.5)  ·  X̂_t+1 = μ_t+1 + R̂  ·  "
           "NFE 1–2 (FlowCast 10, EDM 18–36)",
           size=12, weight="bold", fill="#1B5E20")
    S.text(30, bot + 44 + 110,
           "Σ params — mapping 0.33 M · spatial_emb 2.36 M · "
           "encoder 26.71 M · decoder 51.64 M · total 81,040,772 · "
           "u(z_r, r, t) = 1/(t−r) ∫_r^t v(z_τ, τ) dτ "
           "· measured by instantiation at (192, 96)",
           size=11.5, fill="#546E7A")
    return S.render()


if __name__ == "__main__":
    for name, fig in [("regression_model.svg", figure_regression()),
                      ("meanflow_model.svg", figure_meanflow())]:
        p = os.path.join(OUT_DIR, name)
        with open(p, "w") as f:
            f.write(fig)
        print("wrote", p)
