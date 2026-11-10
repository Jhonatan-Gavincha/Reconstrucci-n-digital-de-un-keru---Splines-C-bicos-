# -*- coding: utf-8 -*-
"""
Reconstrucción digital del borde de un keru Tiwanaku con Splines Cúbicos (paramétricos).
Genera el video completo (1280x720, sin audio) a partir del modelo matemático.

Uso:
    python keru_spline_video.py render  [foto.png]      -> keru_spline.mp4
    python keru_spline_video.py frames 5 30 60 ...      -> PNG de tiempos concretos (prueba)
"""
import sys, subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
from scipy.interpolate import CubicSpline
from PIL import Image

# ----------------------------------------------------------------- CONFIGURACIÓN
IMG_PATH = "keru.png"                 # foto elegida (se puede pasar como 2º argumento)
NOMBRE   = "Jhonatan Pongo Gavincha"
CURSO    = "Desafío final · Reconstrucción arqueológica con Splines"
FOTO_CREDITO = "Foto: registro visual de la pieza (fuente en la descripción)"
FPS, W, H, DUR = 20, 1280, 720, 275.0
OUT = "keru_spline.mp4"

# ----------------------------------------------------------------- DATOS (píxeles de la foto, y hacia abajo)
# 13 puntos de control sobre el borde frontal SANO (a ambos lados de la rotura)
PUNTOS = [(141, 252), (195, 300), (253, 328), (310, 341), (380, 351), (457, 354), (533, 358),
          (840, 333), (880, 326), (940, 316), (980, 289), (1000, 276), (1020, 258)]
# Borde de fractura (solo delimita el relleno; NO entra al spline)
FRACTURA = [(536, 362), (550, 374), (572, 402), (592, 417), (615, 437), (635, 456), (652, 467),
            (670, 477), (686, 477), (710, 465), (735, 460), (760, 440), (772, 425), (790, 407),
            (807, 387), (817, 362), (830, 340)]
ZX0, ZX1, ZY0, ZY1 = 120, 1050, 230, 490      # zona elegida (recorte)
UP = 1.5                                       # ampliación del recorte

# ----------------------------------------------------------------- MODELO: SPLINE CÚBICO PARAMÉTRICO
P = np.array(PUNTOS, float)
n = len(P) - 1                                           # número de tramos (12)
t = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]  # parámetro t = longitud de cuerda
cx = CubicSpline(t, P[:, 0], bc_type="natural")
cy = CubicSpline(t, P[:, 1], bc_type="natural")

def coef(cs, i):                      # a, b, c, d del tramo i  (S_i = a + b s + c s^2 + d s^3)
    d, c, b, a = cs.c[:, i]
    return a, b, c, d

SEG = []                              # puntos densos de cada tramo, calculados con los coeficientes
for i in range(n):
    N = max(30, int((t[i + 1] - t[i]) * 1.2))
    s = np.linspace(0, t[i + 1] - t[i], N)
    ax_, bx_, cx_, dx_ = coef(cx, i); ay_, by_, cy_, dy_ = coef(cy, i)
    SEG.append(np.c_[ax_ + bx_*s + cx_*s**2 + dx_*s**3, ay_ + by_*s + cy_*s**2 + dy_*s**3])
CURVA = np.vstack(SEG)

def saltos():                         # verificación numérica de C0, C1, C2 en los nodos interiores
    j0 = j1 = j2 = 0.0
    for k in range(1, n):
        h = t[k] - t[k - 1]
        for cs in (cx, cy):
            a0, b0, c0, d0 = coef(cs, k - 1); a1, b1, c1, d1 = coef(cs, k)
            j0 = max(j0, abs(a0 + b0*h + c0*h**2 + d0*h**3 - a1))
            j1 = max(j1, abs(b0 + 2*c0*h + 3*d0*h**2 - b1))
            j2 = max(j2, abs(2*c0 + 6*d0*h - 2*c1))
    return j0, j1, j2
J0, J1, J2 = saltos()

def validacion_loo():                 # quitar un punto sano y predecirlo con los demás
    errs = []
    for k in range(1, len(P) - 1):
        Q = np.delete(P, k, axis=0)
        tq = np.r_[0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]
        sx = CubicSpline(tq, Q[:, 0], bc_type="natural"); sy = CubicSpline(tq, Q[:, 1], bc_type="natural")
        tt = np.linspace(0, tq[-1], 6000)
        errs.append(np.hypot(sx(tt) - P[k, 0], sy(tt) - P[k, 1]).min())
    return float(np.mean(errs)), float(np.max(errs))
LOO_MEAN, LOO_MAX = validacion_loo()
GAP = P[7, 0] - P[6, 0]; ANCHO = P[-1, 0] - P[0, 0]

# ----------------------------------------------------------------- IMÁGENES
def cargar(path):
    global PHOTO, CROP, MASK, REC, XS
    PHOTO = np.array(Image.open(path).convert("RGB"))
    crop = Image.fromarray(PHOTO).crop((ZX0, ZY0, ZX1, ZY1))
    crop = crop.resize((int((ZX1 - ZX0) * UP), int((ZY1 - ZY0) * UP)), Image.LANCZOS)
    CROP = np.array(crop)
    # región faltante = entre el spline (arriba) y el borde de fractura (abajo)
    m = (CURVA[:, 0] >= FRACTURA[0][0]) & (CURVA[:, 0] <= FRACTURA[-1][0])
    poly = np.vstack([CURVA[m], np.array(FRACTURA, float)[::-1]])
    from PIL import ImageDraw
    mk = Image.new("L", crop.size, 0)
    ImageDraw.Draw(mk).polygon([((x - ZX0) * UP, (y - ZY0) * UP) for x, y in poly], fill=255)
    MASK = np.array(mk) > 0
    hh, ww = MASK.shape
    yy, xx = np.mgrid[0:hh, 0:ww]
    XS = xx / UP + ZX0
    cyan = np.array([0, 212, 255], float)
    REC = (CROP * 0.55 + cyan * 0.45).astype(np.uint8)
    hatch = ((xx + yy) % 16) < 2
    REC[hatch] = (REC[hatch] * 0.4 + cyan * 0.6).astype(np.uint8)

# ----------------------------------------------------------------- ESTILO / UTILIDADES
BG, FG, DIM = "#0e1117", "#e8eaed", "#8b949e"
CY, MG, YE, OR, GR, RD = "#00d4ff", "#ff5ad1", "#ffd23f", "#ff9f43", "#4ade80", "#ff5d5d"
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
STROKE = [pe.withStroke(linewidth=2.5, foreground="black")]

def T(x, y, s, size=15, color=FG, weight="normal", ha="left", va="top", fam=None, alpha=1.0, stroke=False):
    return fig.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
                    family=fam, alpha=alpha, path_effects=STROKE if stroke else None)

def box(x, y, w, h, fc="#161b22", ec="#30363d"):
    fig.add_artist(Rectangle((x, y), w, h, transform=fig.transFigure, fc=fc, ec=ec, lw=1.2))

def appear(t0, tl, dur=0.6):         # 0→1 desde el instante t0
    return float(np.clip((tl - t0) / dur, 0, 1))

STEPS = ["1 Pieza original", "2 Zona", "3 Puntos", "4 Primeros segmentos",
         "5 Spline completo", "6 Reconstrucción", "7 Resultado final"]
def step_at(tt):
    for k, lim in enumerate([10, 25, 50, 100, 120, 216]):
        if tt < lim: return k
    return 6

def topbar(tt):
    cur = step_at(tt)
    for k, s in enumerate(STEPS):
        col = CY if k == cur else (FG if k < cur else "#4b5563")
        T(0.04 + k * 0.1383, 0.975, s, 9.5, col, "bold" if k == cur else "normal", va="top")
    fig.add_artist(Rectangle((0.04, 0.945), 0.92, 0.003, transform=fig.transFigure, fc="#30363d", lw=0))
    fig.add_artist(Rectangle((0.04, 0.945), 0.92 * min(1, (cur + 1) / 7), 0.003, transform=fig.transFigure, fc=CY, lw=0))

def caption(s):
    T(0.5, 0.028, s, 11, DIM, ha="center", va="bottom")

def zone_axes(img=None):
    ax = fig.add_axes((0.04, 0.43, 0.92, 0.457))
    ax.imshow(CROP if img is None else img, extent=(ZX0, ZX1, ZY1, ZY0), aspect="auto", interpolation="bilinear")
    ax.set_xlim(ZX0, ZX1); ax.set_ylim(ZY1, ZY0); ax.axis("off")
    return ax

def plano(rect, curva, titulo, orig=False):
    """Gráfico en el plano (x, y) como en clase: ejes, cuadrícula, puntos rojos y spline azul."""
    ax = fig.add_axes(rect)
    ax.set_xlim(ZX0, ZX1); ax.set_ylim(ZY1, ZY0)
    if orig:
        ax.imshow(CROP, extent=(ZX0, ZX1, ZY1, ZY0), aspect="auto", interpolation="bilinear")
    else:
        ax.set_facecolor("white"); ax.grid(True, color="#c8c8c8", lw=0.8)
        if curva:
            ax.plot(CURVA[:, 0], CURVA[:, 1], "-", color="#1f77b4", lw=2, label="Spline cúbica")
        ax.plot(P[:, 0], P[:, 1], "o", color="red", ms=5, label="Puntos seleccionados")
        ax.legend(loc="lower center", fontsize=9, facecolor="white", edgecolor="#999999", labelcolor="black")
    ax.set_title(titulo, color=FG, fontsize=12)
    ax.tick_params(colors=FG, labelsize=8)
    ax.set_ylabel("y (px)", color=DIM, fontsize=8, labelpad=1)
    for sp in ax.spines.values(): sp.set_color(DIM)
    return ax

def pts(ax, k, labels=True, size=8, ms=8):
    k = int(k)
    if k <= 0: return
    ax.plot(P[:k, 0], P[:k, 1], "o", color=YE, ms=ms, mec="black", mew=1.2, zorder=5)
    if labels:
        for i in range(k):
            ax.text(P[i, 0], P[i, 1] - 14, f"P{i}", color="white", fontsize=size, ha="center", va="bottom",
                    path_effects=STROKE, zorder=6)

def segs(ax, upto, colors=None, lw=3.2, special=None):
    """dibuja los tramos 0..upto (upto puede ser decimal: el último se traza parcialmente)"""
    full = int(np.floor(upto))
    for i in range(min(full + 1, n)):
        frac = 1.0 if i < full else upto - full
        if frac <= 0: continue
        S = SEG[i][: max(2, int(len(SEG[i]) * frac))]
        col = (special if (special and i == 6) else (colors[i % len(colors)] if colors else CY))
        ax.plot(S[:, 0], S[:, 1], "-", color=col, lw=lw, solid_capstyle="round", zorder=4,
                path_effects=[pe.withStroke(linewidth=lw + 2, foreground="black")])

def fade_in(tl, dur=0.7):
    a = 1 - appear(0, tl, dur)
    if a > 0:
        fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, fc="black", alpha=a, lw=0, zorder=100))

def poly_txt(nom, cs, i):
    a, b, c, d = coef(cs, i)
    return f"{nom}(t) = {a:.1f} {b:+.4f}·s {c:+.2e}·s² {d:+.2e}·s³"

# ----------------------------------------------------------------- ESCENAS
def parte1(tt):                                               # 0:00 – 0:25
    tl = tt
    if tl < 4:
        a = min(1, tl / 1.0)
        T(0.5, 0.60, "Reconstrucción digital de un keru de Tiwanaku", 32, FG, "bold", ha="center", alpha=a)
        T(0.5, 0.49, "mediante Splines Cúbicos paramétricos", 24, CY, ha="center", alpha=a)
        T(0.5, 0.36, NOMBRE, 17, DIM, ha="center", alpha=a)
        return
    ax = fig.add_axes((0.05, 0.07, 0.331, 0.82))
    ax.imshow(PHOTO); ax.axis("off")
    r = appear(10, tl, 0.8)
    if r > 0:
        pulse = 2.5 + 1.2 * np.sin(tl * 5)
        ax.add_patch(Rectangle((ZX0, ZY0), ZX1 - ZX0, ZY1 - ZY0, fill=False, ec=CY, lw=pulse, alpha=r))
    items = [(4.5, "Pieza:", "Keru (kero), vaso ceremonial", FG),
             (7.0, "Cultura:", "Tiwanaku · Horizonte Medio (600–1000 d.C.)", FG),
             (9.0, "Material:", "cerámica pintada · Art Institute of Chicago", FG),
             (12.5, "Zona elegida:", "borde superior frontal, roto en V", CY),
             (16.0, "¿Por qué?", "perfil curvo y suave + sección incompleta", FG)]
    y = 0.80
    for (t0, lab, txt, col) in items:
        a = appear(t0, tl)
        T(0.46, y, lab, 15, YE if col == FG else CY, "bold", alpha=a)
        T(0.605, y, txt, 15, col, alpha=a)
        y -= 0.095
    T(0.605, y + 0.095 - 0.055, "= justo lo que un spline puede reconstruir", 15, FG, alpha=appear(17.5, tl))
    T(0.46, 0.22, "Objetivo: reconstruir la parte del borde que falta,", 15, DIM, alpha=appear(20, tl))
    T(0.46, 0.17, "calculada por el modelo matemático (no dibujada a mano).", 15, DIM, alpha=appear(20.8, tl))

def parte2(tt):                                               # 0:25 – 0:50
    tl = tt - 25
    k = np.clip((tl - 2) / 1.0, 0, 13)
    if tl >= 19.5:                                             # gráfico de los puntos en el plano (x, y)
        plano((0.10, 0.455, 0.80, 0.40), False, "Gráfico de los puntos seleccionados Pᵢ = (xᵢ, yᵢ)")
    else:
        ax = zone_axes()
        pts(ax, k, True, 9, 9)
        if tl > 15.5:                                          # tramo sin datos (rotura)
            a = appear(15.5, tl)
            ax.annotate("", xy=(P[7, 0], 380), xytext=(P[6, 0], 380),
                        arrowprops=dict(arrowstyle="<->", color=RD, lw=2.2, alpha=a))
            ax.text((P[6, 0] + P[7, 0]) / 2, 392, "rotura: sin puntos", color=RD, ha="center", va="top",
                    fontsize=11, alpha=a, path_effects=STROKE)
    T(0.04, 0.395, "Puntos de control  Pᵢ = (xᵢ, yᵢ)   —   13 puntos, en píxeles de la foto (origen arriba-izquierda, y hacia abajo)",
      13, FG, "bold")
    for i in range(13):
        if i < k:
            col, row = divmod(i, 4)
            T(0.05 + col * 0.23, 0.335 - row * 0.062, f"P{i:<2d} = ({int(P[i,0]):4d}, {int(P[i,1]):3d})",
              14, YE if i == int(k) - 1 else FG, fam="monospace")
    T(0.05, 0.098, "Fuente: fotografía de la pieza · coordenadas medidas sobre una cuadrícula superpuesta a la imagen",
      12, DIM, alpha=appear(16, tl))
    caption("Zona elegida: borde frontal del keru")

def parte3(tt):                                               # 0:50 – 1:15
    tl = tt - 50
    ax = zone_axes()
    pts(ax, 13, True, 8, 7)
    f0 = np.clip((tl - 3) / 5, 0, 1); f1 = np.clip((tl - 8) / 4, 0, 1)
    segs(ax, f0 + f1 if f0 >= 1 else f0, colors=[CY, MG])
    if tl > 12:
        a = appear(12, tl)
        ax.plot([P[1, 0]], [P[1, 1]], "o", ms=20, mfc="none", mec="white", mew=2, alpha=a, zorder=7)
        d = np.array([cx(t[1], 1), cy(t[1], 1)]); d = d / np.hypot(*d) * 70
        ax.annotate("", xy=(P[1, 0] + d[0], P[1, 1] + d[1]), xytext=(P[1, 0] - d[0], P[1, 1] - d[1]),
                    arrowprops=dict(arrowstyle="-|>", color=YE, lw=2.4, alpha=a), zorder=8)
        ax.text(P[1, 0] + 45, P[1, 1] - 38, "nodo P1: misma tangente a ambos lados", color=YE, fontsize=11, alpha=a, va="bottom", path_effects=STROKE)
    T(0.04, 0.395, "Spline cúbico paramétrico  ·  t = longitud de cuerda", 14, FG, "bold",
      alpha=appear(1, tl))
    T(0.04, 0.33, r"$X_i(t)=a_{x,i}+b_{x,i}(t-t_i)+c_{x,i}(t-t_i)^2+d_{x,i}(t-t_i)^3$", 17, CY, alpha=appear(4, tl))
    T(0.04, 0.25, r"$Y_i(t)=a_{y,i}+b_{y,i}(t-t_i)+c_{y,i}(t-t_i)^2+d_{y,i}(t-t_i)^3$", 17, CY, alpha=appear(7, tl))
    T(0.04, 0.16, r"13 puntos $\rightarrow$ 12 tramos cúbicos $\rightarrow$ 48 coeficientes por coordenada", 13, FG,
      alpha=appear(20, tl))
    T(0.04, 0.115, r"$48 = 24$ (pasa por los puntos) $+\ 11\ (C^1) + 11\ (C^2) + 2$ (natural: $S''=0$ en los extremos)", 12, DIM,
      alpha=appear(21.5, tl))
    T(0.62, 0.395, "Los tramos se unen con continuidad:", 14, FG, "bold", alpha=appear(10, tl))
    T(0.62, 0.33, r"$C^0$:  $S_i(t_{i+1})=S_{i+1}(t_{i+1})$  sin saltos", 14, GR, alpha=appear(12, tl))
    T(0.62, 0.25, r"$C^1$:  $S_i'(t_{i+1})=S_{i+1}'(t_{i+1})$  sin picos", 14, YE, alpha=appear(15, tl))
    T(0.62, 0.17, r"$C^2$:  $S_i''(t_{i+1})=S_{i+1}''(t_{i+1})$  curvatura suave", 14, MG, alpha=appear(18, tl))
    caption("Cada tramo es un polinomio cúbico en x(t) e y(t); juntos forman una curva continua y suave")

CODE = ("P = np.array(PUNTOS)                          # 13 puntos (x, y)\n"
        "t = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]\n"
        "cx = CubicSpline(t, P[:, 0], bc_type='natural')\n"
        "cy = CubicSpline(t, P[:, 1], bc_type='natural')\n"
        "d, c, b, a = cx.c[:, i]        # coeficientes del tramo i\n"
        "X, Y = cx(tt), cy(tt)          # curva reconstruida\n"
        "# --- animación (cada cuadro dibuja parte de la curva del modelo) ---\n"
        "for i in range(12):                       # un tramo tras otro\n"
        "    S = np.c_[X_i(s), Y_i(s)]             # polinomio cúbico del tramo i\n"
        "    ax.plot(S[:k, 0], S[:k, 1])           # k crece con el tiempo\n"
        "img[mask & (XS <= xp)] = relleno          # relleno progresivo")

def parte4(tt):                                               # 1:15 – 2:00
    tl = tt - 75
    ax = zone_axes()
    # cronograma: tramo 0 (0–10 s), tramos 1-3 (10–22 s), resto (22–34 s), completo (34–45 s)
    if tl < 10: u = np.clip((tl - 1) / 8, 0, 1)
    elif tl < 22: u = 1 + (tl - 10) / 12 * 3
    elif tl < 34: u = 4 + (tl - 22) / 12 * 8
    else: u = 12
    full = tl >= 34
    pts(ax, 13, not full, 8, 7)
    segs(ax, min(u, 11.999) if u < 12 else 12, colors=[CY] if full else [CY, MG], special=None if full else OR)
    if full:
        ax.plot(P[1:-1, 0], P[1:-1, 1], "o", ms=13, mfc="none", mec="white", mew=1.6, zorder=7)
    cur = int(min(u, 11.999))
    # panel de código
    box(0.04, 0.07, 0.50, 0.33)
    T(0.05, 0.385, "Código (Python · SciPy · Matplotlib)", 11, DIM, "bold")
    T(0.05, 0.345, CODE, 10.2, "#c9d1d9", fam="monospace")
    # panel de coeficientes
    box(0.56, 0.07, 0.40, 0.33)
    if not full:
        esp = "  ← cruza la rotura: aquí está la reconstrucción" if cur == 6 else ""
        T(0.57, 0.385, f"Tramo {cur}:  P{cur} → P{cur+1}{esp}", 12.5, OR if cur == 6 else CY, "bold")
        T(0.57, 0.335, "s = t − t_i   (a, b, c, d: coeficientes del tramo)", 10.5, DIM)
        T(0.57, 0.285, poly_txt("X", cx, cur), 10.8, FG, fam="monospace")
        T(0.57, 0.235, poly_txt("Y", cy, cur), 10.8, FG, fam="monospace")
        T(0.57, 0.160, f"Tramos dibujados: {min(int(u) + 1, 12)} / 12", 12, YE)
    else:
        a = appear(34, tl)
        T(0.57, 0.385, "Spline completo: 12 tramos unidos", 13, CY, "bold", alpha=a)
        T(0.57, 0.330, "Verificación numérica en los 11 nodos interiores:", 11, DIM, alpha=appear(36, tl))
        T(0.57, 0.280, f"salto máx. en C⁰ = {J0:.1e}", 12.5, GR, fam="monospace", alpha=appear(37, tl))
        T(0.57, 0.230, f"salto máx. en C¹ = {J1:.1e}", 12.5, YE, fam="monospace", alpha=appear(38.5, tl))
        T(0.57, 0.180, f"salto máx. en C² = {J2:.1e}", 12.5, MG, fam="monospace", alpha=appear(40, tl))
        T(0.57, 0.115, "→ curva continua, sin picos y de curvatura suave", 11.5, FG, alpha=appear(42, tl))
    caption("Los tramos se van agregando uno a uno hasta formar el spline completo")

def parte5(tt):                                               # 2:00 – 2:35
    tl = tt - 120
    if tl >= 28:                                               # plano: spline vs. imagen original (como en clase)
        plano((0.13, 0.52, 0.74, 0.37), True, "Imagen generada mediante Splines")
        plano((0.13, 0.08, 0.74, 0.37), False, "Imagen original", orig=True)
        return
    img = CROP
    ax_fill = 0.0
    if 6 <= tl < 18:                                           # relleno progresivo de izquierda a derecha
        ax_fill = (tl - 6) / 12
    elif tl >= 18:
        ax_fill = 1.0
    if ax_fill > 0:
        xp = FRACTURA[0][0] + (FRACTURA[-1][0] - FRACTURA[0][0]) * ax_fill
        img = CROP.copy(); sel = MASK & (XS <= xp); img[sel] = REC[sel]
    wipe = None
    if tl >= 18:                                               # comparación antes / después
        w = tl - 18
        if w < 2: wipe = 0.55
        elif w < 5: wipe = 0.55 + 0.45 * (w - 2) / 3
        elif w < 9: wipe = 1.0 - 1.0 * (w - 5) / 4
        else: wipe = 0.0
        xw = ZX0 + (ZX1 - ZX0) * wipe
        img = CROP.copy(); sel = MASK & (XS >= xw); img[sel] = REC[sel]
        # (a la derecha de la línea se ve la reconstrucción; a la izquierda, el original)
        if wipe >= 0.999: img = CROP
    ax = zone_axes(img)
    if tl < 6:
        a = appear(1, tl)
        F = np.array(FRACTURA)
        ax.plot(F[:, 0], F[:, 1], "-", color=RD, lw=2.4, alpha=a, zorder=5)
        ax.text(690, 492, "borde roto (medido: solo delimita el relleno)", color=RD, fontsize=11, ha="center",
                va="bottom", alpha=a, path_effects=STROKE)
    if tl >= 6:
        xmin = ZX0 + (ZX1 - ZX0) * wipe if (wipe is not None and wipe < 0.999) else -1e9
        mc = CURVA[:, 0] >= xmin
        ax.plot(CURVA[mc, 0], CURVA[mc, 1], "-", color=CY, lw=3, zorder=4, path_effects=[pe.withStroke(linewidth=5, foreground="black")])
        mp = P[:, 0] >= xmin
        ax.plot(P[mp, 0], P[mp, 1], "o", color=YE, ms=7, mec="black", mew=1.2, zorder=5)
    if tl >= 18 and wipe is not None:
        xw = ZX0 + (ZX1 - ZX0) * wipe
        ax.axvline(xw, color="white", lw=2.2, zorder=9)
        ax.text(ZX0 + 6, ZY0 + 8, "ORIGINAL (con rotura)", color="white", fontsize=12, va="top", path_effects=STROKE)
        ax.text(ZX1 - 6, ZY0 + 8, "RECONSTRUCCIÓN DIGITAL", color=CY, fontsize=12, va="top", ha="right", path_effects=STROKE)
    box(0.04, 0.07, 0.92, 0.33)
    if tl < 6:
        T(0.06, 0.37, "6 · Reconstrucción digital", 18, CY, "bold")
        T(0.06, 0.29, "La rotura deja un tramo del borde sin datos (rojo).", 15, FG, alpha=appear(2, tl))
        T(0.06, 0.23, "El spline lo cruza con una curva C² que continúa el borde sano.", 15, FG, alpha=appear(3.5, tl))
    elif tl < 18:
        T(0.06, 0.37, "6 · Reconstrucción digital", 18, CY, "bold")
        T(0.06, 0.29, "Relleno = región entre el spline y el borde roto.", 15, FG)
        T(0.06, 0.23, f"Tramo reconstruido: {GAP:.0f} px de {ANCHO:.0f} px de borde  (≈ {100*GAP/ANCHO:.0f} %)", 15, YE,
          alpha=appear(9, tl))
        T(0.06, 0.17, "El fragmento sale del modelo X(t), Y(t); no está dibujado a mano.", 13, DIM, alpha=appear(11, tl))
    else:
        T(0.06, 0.37, "Comparación antes / después del borde frontal", 18, CY, "bold")
        T(0.06, 0.29, "Izquierda de la línea: pieza original. Derecha: borde reconstruido con el spline.", 15, FG)
        T(0.06, 0.23, "13 puntos  →  12 tramos cúbicos  →  curva continua y suave (C⁰, C¹, C²)", 15, YE)
    caption("Animación generada directamente desde el modelo del spline")

def parte6(tt):                                               # 2:35 – 3:00
    tl = tt - 155
    ax = zone_axes(CROP.copy() if False else None)
    img = CROP.copy(); img[MASK] = REC[MASK]
    ax.images[0].set_data(img)
    ax.plot(CURVA[:, 0], CURVA[:, 1], "-", color=CY, lw=3, path_effects=[pe.withStroke(linewidth=5, foreground="black")])
    pts(ax, 13, False, 8, 7)
    T(0.04, 0.395, "Conclusión", 20, CY, "bold")
    T(0.04, 0.335, "Qué se reconstruyó:", 14.5, YE, "bold", alpha=appear(0.5, tl))
    T(0.25, 0.335, f"el borde frontal del keru; el tramo roto, ≈ {100*GAP/ANCHO:.0f} % del borde.", 14.5, FG, alpha=appear(0.5, tl))
    T(0.04, 0.275, "Utilidad:", 14.5, YE, "bold", alpha=appear(3.5, tl))
    T(0.25, 0.275, "con pocos puntos, una curva suave que pasa por los datos y completa lo faltante.", 14.5, FG, alpha=appear(3.5, tl))
    T(0.04, 0.215, "Validación:", 14.5, GR, "bold", alpha=appear(6.5, tl))
    T(0.25, 0.215, f"quitando un punto sano y prediciéndolo, el error medio fue {LOO_MEAN:.1f} px (máx. {LOO_MAX:.1f}).", 14.5, FG,
      alpha=appear(6.5, tl))
    T(0.04, 0.155, "Limitación:", 14.5, RD, "bold", alpha=appear(9.5, tl))
    T(0.25, 0.155, "foto en perspectiva y sin puntos dentro de la rotura: es una estimación, no el original.", 14.5, FG, alpha=appear(9.5, tl))
    T(0.04, 0.095, "Aplicación futura:", 14.5, MG, "bold", alpha=appear(12.5, tl))
    T(0.25, 0.095, "restauración virtual 3D (fotogrametría + perfil de revolución) e impresión 3D.", 14.5, FG, alpha=appear(12.5, tl))
    if tl > 16.5:
        T(0.5, 0.035, "¡Gracias!", 15, FG, "bold", ha="center", va="bottom", alpha=appear(16.5, tl))

# =================================================================== CONTORNO COMPLETO (sección nueva)
BACK  = [(141, 250), (150, 219), (180, 196), (210, 182), (250, 165), (290, 158), (330, 144), (380, 134), (430, 128),
         (480, 123), (540, 120), (600, 121), (660, 122), (720, 125), (760, 129), (835, 140), (870, 146), (910, 157),
         (950, 171), (985, 187), (1005, 201), (1021, 218), (1030, 238), (1031, 254)]
LEFT  = [(141, 252), (149, 332), (160, 402), (173, 472), (189, 542), (200, 612), (212, 682), (222, 752), (232, 822),
         (241, 892), (256, 962), (270, 1032), (275, 1150), (283, 1233), (292, 1289), (303, 1339), (317, 1372)]
RIGHT = [(1031, 254), (1020, 332), (1005, 402), (997, 472), (986, 542), (976, 612), (967, 682), (961, 752), (950, 822),
         (939, 892), (930, 962), (923, 1032), (917, 1150), (914, 1233), (906, 1289), (889, 1339), (867, 1389)]
BASE  = [(317, 1372), (350, 1405), (389, 1436), (433, 1464), (489, 1480), (561, 1489), (628, 1487), (694, 1478),
         (750, 1467), (806, 1450), (844, 1422), (867, 1389)]
CHIP  = [(260, 201), (254, 188), (260, 172), (272, 160), (290, 160), (310, 153), (330, 149), (341, 160), (344, 172),
         (332, 185), (310, 195), (285, 200)]                       # zona C: spline CERRADO (periódico)
NOTCH = [(775, 135), (780, 138), (785, 143), (790, 148), (795, 151), (800, 155), (805, 159), (810, 163), (815, 162),
         (820, 156), (825, 148), (830, 142)]                       # borde de la muesca (solo delimita el relleno)
SLIVER = [(919, 331), (923, 334), (929, 336), (935, 336), (940, 332), (945, 325), (949, 317), (952, 311)]  # zona D: borde de la pequeña muesca
TAB = {"back": "#1f77b4", "front": "#d62728", "left": "#2ca02c", "right": "#ff7f0e", "base": "#9467bd", "chip": "#e377c2"}
NOMBRES = {"back": "Borde trasero", "front": "Borde frontal", "left": "Pared izquierda", "right": "Pared derecha",
           "base": "Base", "chip": "Desprendimiento (cerrado)"}
ORDEN = ["back", "front", "left", "right", "base"]

def region(Pl, closed=False):
    Pa = np.array(Pl, float)
    Q = np.vstack([Pa, Pa[:1]]) if closed else Pa
    tq = np.r_[0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]
    bc = "periodic" if closed else "natural"
    rx = CubicSpline(tq, Q[:, 0], bc_type=bc); ry = CubicSpline(tq, Q[:, 1], bc_type=bc)
    sg = []
    for i in range(len(Q) - 1):
        ss = np.linspace(tq[i], tq[i + 1], max(30, int(tq[i + 1] - tq[i])))
        sg.append(np.c_[rx(ss), ry(ss)])
    return dict(P=Pa, t=tq, SEG=sg, C=np.vstack(sg), n=len(Q) - 1, closed=closed)

REG = {"back": region(BACK), "front": region(PUNTOS), "left": region(LEFT), "right": region(RIGHT),
       "base": region(BASE), "chip": region(CHIP, True)}
NPTS = sum(len(REG[k]["P"]) for k in ORDEN); NSEG = sum(REG[k]["n"] for k in ORDEN)

def loo_todo():
    errs = []
    for k in ORDEN:
        Pk = REG[k]["P"]
        for j in range(1, len(Pk) - 1):
            Q = np.delete(Pk, j, axis=0)
            tq = np.r_[0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]
            sx = CubicSpline(tq, Q[:, 0], bc_type="natural"); sy = CubicSpline(tq, Q[:, 1], bc_type="natural")
            tt = np.linspace(0, tq[-1], 4000)
            errs.append(np.hypot(sx(tt) - Pk[j, 0], sy(tt) - Pk[j, 1]).min())
    return float(np.mean(errs)), float(np.max(errs))
LOO2_MEAN, LOO2_MAX = loo_todo()

PX0, PX1, PY0, PY1 = 100, 1060, 90, 1520          # recuadro de la pieza completa
ZONAS = {}

def poly_mask(poly, size):
    from PIL import ImageDraw
    mk = Image.new("L", size, 0)
    ImageDraw.Draw(mk).polygon([(float(x), float(y)) for x, y in poly], fill=255)
    return np.array(mk) > 0

def cargar_full():
    global PIECE, RECP, POLYS, MALL
    h, w = PHOTO.shape[:2]
    Cf = REG["front"]["C"]; Cb = REG["back"]["C"]; Cc = REG["chip"]["C"]
    mA = (Cf[:, 0] >= FRACTURA[0][0]) & (Cf[:, 0] <= FRACTURA[-1][0])
    mB = (Cb[:, 0] >= NOTCH[0][0]) & (Cb[:, 0] <= NOTCH[-1][0])
    POLYS = {"A": np.vstack([Cf[mA], np.array(FRACTURA, float)[::-1]]),
             "B": np.vstack([Cb[mB], np.array(NOTCH, float)[::-1]]),
             "C": Cc,
             "D": np.vstack([Cf[(Cf[:, 0] >= SLIVER[0][0]) & (Cf[:, 0] <= SLIVER[-1][0])], np.array(SLIVER, float)[::-1]])}
    masks = {k: poly_mask(v, (w, h)) for k, v in POLYS.items()}
    MALL = masks["A"] | masks["B"] | masks["C"] | masks["D"]
    cyan = np.array([0, 212, 255], float)
    yy, xx = np.mgrid[0:h, 0:w]
    rec = (PHOTO * 0.55 + cyan * 0.45).astype(np.uint8)
    hatch = ((xx + yy) % 16) < 2
    rec[hatch] = (rec[hatch] * 0.4 + cyan * 0.6).astype(np.uint8)
    full = PHOTO.copy(); full[MALL] = rec[MALL]
    sc = 0.6
    def red(a): return np.array(Image.fromarray(a[PY0:PY1, PX0:PX1]).resize((int((PX1-PX0)*sc), int((PY1-PY0)*sc)), Image.LANCZOS))
    PIECE = red(PHOTO); RECP = red(full)
    cajas = {"A": (500, 320, 870, 500), "B": (715, 100, 885, 205), "C": (225, 128, 375, 218), "D": (895, 295, 985, 345)}
    for k, box in cajas.items():
        x0, y0, x1, y1 = box; ww, hh = x1 - x0, y1 - y0
        s_ = min(794 / ww, 576 / hh); W2, H2 = int(ww * s_), int(hh * s_)
        base = np.array(Image.fromarray(PHOTO).crop(box).resize((W2, H2), Image.LANCZOS))
        mk = np.array(Image.fromarray((masks[k] * 255).astype(np.uint8)).crop(box).resize((W2, H2), Image.BILINEAR)) > 127
        y2, x2 = np.mgrid[0:H2, 0:W2]
        r2 = (base * 0.55 + cyan * 0.45).astype(np.uint8)
        hh2 = ((x2 + y2) % 16) < 2
        r2[hh2] = (r2[hh2] * 0.4 + cyan * 0.6).astype(np.uint8)
        xsz = x2 / s_ + x0
        ZONAS[k] = dict(box=box, base=base, mask=mk, rec=r2, xs=xsz, xmin=xsz[mk].min(), xmax=xsz[mk].max(),
                        area=int(masks[k].sum()))

def piece_axes(rect, alpha=1.0, plane=False, img=None):
    ax = fig.add_axes(rect)
    if plane: ax.set_facecolor("white")
    ax.imshow(PIECE if img is None else img, extent=(PX0, PX1, PY1, PY0), alpha=alpha, interpolation="bilinear")
    ax.set_xlim(PX0, PX1); ax.set_ylim(PY1, PY0)
    ax.axis("off")
    return ax

def curva_reg(ax, k, f=1.0, lw=2.4, color=None, stroke=True, xmin=None):
    C = REG[k]["C"]
    n_ = max(2, int(len(C) * f))
    C = C[:n_]
    if xmin is not None:
        C = np.where((C[:, 0] >= xmin)[:, None], C, np.nan)
    ax.plot(C[:, 0], C[:, 1], "-", color=color or TAB[k], lw=lw, solid_capstyle="round", zorder=4,
            path_effects=[pe.withStroke(linewidth=lw + 1.8, foreground="black")] if stroke else None)

def puntos_reg(ax, k, f=1.0, ms=3.6, xmin=None):
    Pk = REG[k]["P"]; m = int(round(len(Pk) * f))
    Pk = Pk[:m]
    if xmin is not None: Pk = Pk[Pk[:, 0] >= xmin]
    ax.plot(Pk[:, 0], Pk[:, 1], "o", color="red", ms=ms, mec="black", mew=0.5, zorder=5)

def panel_regiones(tl, t0s, extra=None):
    """tabla de regiones (color, nombre, puntos, tramos) a la derecha"""
    T(0.52, 0.90, "Contorno completo = varios splines", 17, FG, "bold", alpha=appear(0.2, tl))
    y = 0.80
    for i, k in enumerate(ORDEN):
        a = appear(t0s[i], tl)
        fig.add_artist(Rectangle((0.52, y - 0.028), 0.018, 0.034, transform=fig.transFigure, fc=TAB[k], lw=0, alpha=a))
        T(0.55, y, f"{NOMBRES[k]}", 14.5, FG, alpha=a)
        T(0.775, y, f"{len(REG[k]['P'])} puntos · {REG[k]['n']} tramos", 11.5, DIM, alpha=a, fam="monospace")
        y -= 0.075
    return y

def n1(tt):                                                   # puntos del contorno completo
    tl = tt - 148
    ax = piece_axes((0.03, 0.09, 0.46, 0.80))
    t0s = [0.8 + 2.1 * i for i in range(5)]
    for i, k in enumerate(ORDEN):
        puntos_reg(ax, k, np.clip((tl - t0s[i]) / 1.8, 0, 1), ms=4.2)
    y = panel_regiones(tl, t0s)
    T(0.52, y - 0.01, f"Total: {NPTS} puntos de control  →  {NSEG} tramos cúbicos", 15, YE, "bold", alpha=appear(11.0, tl))
    T(0.52, y - 0.075, "Puntos seleccionados sobre el borde de la pieza (rojo).", 12.5, DIM, alpha=appear(11.0, tl))
    T(0.52, y - 0.115, "Los tramos rotos se dejan SIN puntos: los cruza el spline.", 12.5, DIM, alpha=appear(11.8, tl))
    caption("Contorno completo de la pieza: puntos de control por regiones")

def n2(tt):                                                   # splines por regiones sobre la foto
    tl = tt - 160
    ax = piece_axes((0.03, 0.09, 0.46, 0.80))
    t0s = [0.6 + 2.3 * i for i in range(5)]
    for i, k in enumerate(ORDEN):
        f = np.clip((tl - t0s[i]) / 2.1, 0, 1)
        if f > 0: curva_reg(ax, k, f, 2.6)
        puntos_reg(ax, k, 1.0, 3.0)
    if tl > 13:
        a = appear(13, tl)
        for k, sidx in (("back", 14), ("front", 6)):
            S = REG[k]["SEG"][sidx]
            ax.plot(S[:, 0], S[:, 1], "-", color=YE, lw=4.0, alpha=a, zorder=6,
                    path_effects=[pe.withStroke(linewidth=6, foreground="black")])
    y = panel_regiones(tl, t0s)
    T(0.52, y - 0.01, "Cada región: spline cúbico paramétrico natural", 14, CY, "bold", alpha=appear(12, tl))
    T(0.52, y - 0.065, "con continuidad C⁰, C¹ y C² dentro de cada región.", 13, FG, alpha=appear(12.6, tl))
    T(0.52, y - 0.125, "Amarillo: los dos tramos que cruzan una rotura", 13, YE, alpha=appear(13.5, tl))
    T(0.52, y - 0.17, "(borde trasero → zona B, borde frontal → zona A).", 12, DIM, alpha=appear(13.5, tl))
    caption("Cada región del contorno se construye con su propio spline cúbico")

def n3(tt):                                                   # transición: de la pieza real al modelo
    tl = tt - 176
    u = float(np.clip((tl - 1.0) / 5.0, 0, 1))
    ax = piece_axes((0.03, 0.09, 0.46, 0.80), alpha=1 - u, plane=True)
    if u > 0.15:
        ax.axis("on"); ax.grid(True, color="#c8c8c8", lw=0.8, alpha=u)
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values(): sp.set_color(DIM)
    for k in ORDEN:
        curva_reg(ax, k, 1.0, 2.2, stroke=(u < 0.6))
        puntos_reg(ax, k, 1.0, 3.0)
    T(0.52, 0.86, "Transición: del objeto real", 20, FG, "bold", alpha=appear(0.5, tl))
    T(0.52, 0.79, "al modelo matemático", 20, CY, "bold", alpha=appear(1.5, tl))
    T(0.52, 0.66, "Se desvanece la fotografía y queda solo", 15, FG, alpha=appear(3.5, tl))
    T(0.52, 0.61, "lo que calculan los splines.", 15, FG, alpha=appear(3.5, tl))
    T(0.52, 0.50, f"{NPTS} puntos  →  {NSEG} tramos cúbicos", 16, YE, "bold", alpha=appear(5.0, tl))
    T(0.52, 0.44, "Imagen generada mediante Splines", 14, DIM, alpha=appear(5.8, tl))
    caption("Plano (x, y) en píxeles: solo puntos de control y splines")

def zona_scene(tl, k, d, reg, seg, color, titulo, lineas):
    Z = ZONAS[k]; x0, y0, x1, y1 = Z["box"]
    fr_c = float(np.clip((tl - 0.12 * d) / (0.38 * d), 0, 1)); fr_f = float(np.clip((tl - 0.52 * d) / (0.42 * d), 0, 1))
    img = Z["base"]
    if fr_f > 0:
        xp = Z["xmin"] + (Z["xmax"] - Z["xmin"]) * fr_f
        img = Z["base"].copy(); sel = Z["mask"] & (Z["xs"] <= xp); img[sel] = Z["rec"][sel]
    ax = fig.add_axes((0.03, 0.08, 0.62, 0.80))
    ax.imshow(img, extent=(x0, x1, y1, y0), interpolation="bilinear")
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.axis("off")
    R = REG[reg]; Pp = R["P"]
    inb = (Pp[:, 0] >= x0) & (Pp[:, 0] <= x1) & (Pp[:, 1] >= y0) & (Pp[:, 1] <= y1)
    if tl > 0.08 * d: ax.plot(Pp[inb, 0], Pp[inb, 1], "o", color="red", ms=8, mec="black", mew=1.1, zorder=6)
    C = R["C"]
    if R["closed"]:
        cur = C[:max(2, int(len(C) * fr_c))]
    else:
        xcut = x0 + (x1 - x0) * fr_c
        cur = C[(C[:, 0] >= x0 - 5) & (C[:, 0] <= xcut)]
    if len(cur) > 1 and fr_c > 0:
        ax.plot(cur[:, 0], cur[:, 1], "-", color=color, lw=3.4, zorder=5, path_effects=[pe.withStroke(linewidth=5.5, foreground="black")])
    if seg is not None and fr_c > 0:
        S = R["SEG"][seg]; Sx = S[S[:, 0] <= x0 + (x1 - x0) * fr_c]
        if len(Sx) > 1: ax.plot(Sx[:, 0], Sx[:, 1], "-", color=YE, lw=4.0, zorder=6, path_effects=[pe.withStroke(linewidth=6.2, foreground="black")])
    T(0.68, 0.88, titulo, 17, color, "bold", alpha=appear(0.1, tl))
    yy = 0.78
    for (t0, txt, col, sz) in lineas:
        T(0.68, yy, txt, sz, col, alpha=appear(t0 * d, tl)); yy -= 0.062 if sz >= 13 else 0.05

def n4(tt):
    ra = REG["front"]; rb = REG["back"]; rc = REG["chip"]
    if tt < 189:
        tl = tt - 184
        zona_scene(tl, "A", 5.0, "front", 6, TAB["front"], "Zona A · Rotura en V", [
            (0.1, "Dónde: borde frontal", FG, 14),
            (0.3, f"P6 = ({int(ra['P'][6,0])}, {int(ra['P'][6,1])})", DIM, 11.5),
            (0.3, f"P7 = ({int(ra['P'][7,0])}, {int(ra['P'][7,1])})", DIM, 11.5),
            (0.5, f"Tramo reconstruido: {ra['t'][7]-ra['t'][6]:.0f} px", YE, 13.5),
            (0.7, f"Superficie repuesta: ≈ {ZONAS['A']['area']} px²", FG, 13.5)])
        caption("Zona A: ya reconstruida en la primera parte")
    elif tt < 199:
        tl = tt - 189
        zona_scene(tl, "B", 10.0, "back", 14, TAB["back"], "Zona B · Muesca trasera", [
            (0.08, "Dónde: borde trasero (arriba)", FG, 14),
            (0.20, "El borde sano a cada lado define el spline:", DIM, 11.5),
            (0.25, f"P14 = ({int(rb['P'][14,0])}, {int(rb['P'][14,1])})", DIM, 11.5),
            (0.25, f"P15 = ({int(rb['P'][15,0])}, {int(rb['P'][15,1])})", DIM, 11.5),
            (0.55, f"Tramo reconstruido: {rb['t'][15]-rb['t'][14]:.0f} px", YE, 13.5),
            (0.75, f"Superficie repuesta: ≈ {ZONAS['B']['area']} px²", FG, 13.5)])
        caption("Zona B: el spline cierra la muesca del borde trasero")
    elif tt < 210:
        tl = tt - 199
        zona_scene(tl, "C", 11.0, "chip", None, TAB["chip"], "Zona C · Desprendimiento", [
            (0.08, "Dónde: borde trasero izquierdo", FG, 14),
            (0.15, "Arcilla expuesta por una astilla perdida.", DIM, 11.5),
            (0.30, "Se delimita con un spline CERRADO:", CY, 13.5),
            (0.35, f"{len(rc['P'])} puntos · {rc['n']} tramos periódicos", DIM, 11.5),
            (0.60, f"Superficie repuesta: ≈ {ZONAS['C']['area']} px²", FG, 13.5),
            (0.75, "(el spline cerrado vuelve a su punto inicial)", DIM, 11.5)])
        caption("Zona C: spline cerrado que delimita la zona dañada")
    else:
        tl = tt - 210
        zona_scene(tl, "D", 6.0, "front", None, TAB["front"], "Zona D · Pequeña muesca", [
            (0.1, "Dónde: borde frontal, junto a la V", FG, 14),
            (0.3, "Escalón con arcilla expuesta (≈ 33 × 15 px).", DIM, 11.5),
            (0.5, "El spline del borde frontal la cruza.", CY, 13.5),
            (0.75, f"Superficie repuesta: ≈ {ZONAS['D']['area']} px²", FG, 13.5)])
        caption("Zona D: pequeña muesca junto a la rotura en V")

def n5(tt):                                                   # pieza completa: antes/después y plano final
    tl = tt - 216
    if tl < 12:
        w = tl
        if w < 1.5: wp = 0.5
        elif w < 5: wp = 0.5 + 0.5 * (w - 1.5) / 3.5
        elif w < 10: wp = 1.0 - 1.0 * (w - 5) / 5
        else: wp = 0.0
        h_, w_ = PIECE.shape[:2]; xi = int(w_ * wp)
        img = PIECE.copy(); img[:, xi:] = RECP[:, xi:]
        ax = piece_axes((0.30, 0.08, 0.40, 0.82), img=img)
        xw = PX0 + (PX1 - PX0) * wp
        ax.axvline(xw, color="white", lw=2.2, zorder=9)
        for k in ORDEN:
            curva_reg(ax, k, 1.0, 1.8, xmin=xw)
        curva_reg(ax, "chip", 1.0, 1.8, color=TAB["chip"], xmin=xw)
        for lab, (lx, ly) in {"A": (690, 505), "B": (800, 108), "C": (300, 128), "D": (936, 362)}.items():
            ax.text(lx, ly, lab, color="white", fontsize=13, fontweight="bold", ha="center", va="center", path_effects=STROKE, zorder=10)
        T(0.03, 0.84, "ANTES", 20, FG, "bold"); T(0.03, 0.77, "pieza original", 14, DIM)
        T(0.03, 0.70, "4 zonas con pérdidas:", 13.5, FG)
        for i, (lab, tx) in enumerate([("A", "rotura en V (borde frontal)"), ("B", "muesca (borde trasero)"), ("C", "desprendimiento (arcilla)"), ("D", "pequeña muesca junto a la V")]):
            T(0.03, 0.64 - i * 0.055, f"{lab} · {tx}", 12, DIM)
        T(0.72, 0.84, "DESPUÉS", 20, CY, "bold"); T(0.72, 0.77, "reconstrucción digital", 14, DIM)
        T(0.72, 0.70, f"{NPTS} puntos de control", 13.5, FG); T(0.72, 0.645, f"{NSEG} tramos + 1 spline cerrado", 13.5, FG)
        T(0.72, 0.59, "4 zonas repuestas", 13.5, YE)
        caption("Comparación antes / después de la pieza completa")
    else:
        tl2 = tl - 12
        a = piece_axes((0.07, 0.10, 0.40, 0.76), plane=True, alpha=0.0)
        ax = a; ax.axis("on"); ax.grid(True, color="#c8c8c8", lw=0.8)
        ax.tick_params(colors=FG, labelsize=8)
        for sp in ax.spines.values(): sp.set_color(DIM)
        for kk in ("A", "B", "C", "D"):
            pl = POLYS[kk]; ax.fill(pl[:, 0], pl[:, 1], color="#00bcd4", alpha=0.45, lw=0, zorder=2)
        for k in ORDEN: curva_reg(ax, k, 1.0, 2.0, stroke=False); puntos_reg(ax, k, 1.0, 3.0)
        curva_reg(ax, "chip", 1.0, 2.0, stroke=False); puntos_reg(ax, "chip", 1.0, 3.0)
        for lab, (lx, ly) in {"A": (690, 405), "B": (805, 108), "C": (300, 128), "D": (936, 355)}.items():
            ax.text(lx, ly, lab, color="black", fontsize=12, fontweight="bold", ha="center", va="center", zorder=8)
        from matplotlib.lines import Line2D
        hs = [Line2D([0], [0], color=TAB[k], lw=2.5, label=NOMBRES[k]) for k in ORDEN + ["chip"]]
        hs.append(Rectangle((0, 0), 1, 1, fc="#00bcd4", alpha=0.45, label="Zona reconstruida"))
        hs.append(Line2D([0], [0], marker="o", color="none", mfc="red", mec="black", ms=5, label="Puntos seleccionados"))
        ax.legend(handles=hs, loc="center", bbox_to_anchor=(0.5, 0.5), fontsize=7.5, facecolor="white", edgecolor="#999999", labelcolor="black")
        ax.set_title("Imagen generada mediante Splines", color=FG, fontsize=12)
        ax2 = piece_axes((0.55, 0.10, 0.40, 0.76), img=PIECE)
        ax2.axis("on"); ax2.tick_params(colors=FG, labelsize=8)
        for sp in ax2.spines.values(): sp.set_color(DIM)
        ax2.set_title("Imagen original", color=FG, fontsize=12)
        caption("Resultado final: contorno completo y zonas reconstruidas con splines cúbicos")

import textwrap
def concl(tt):                                                # conclusión
    tl = tt - 242
    fig.add_axes((0.03, 0.05, 0.30, 0.85)).imshow(RECP, extent=(PX0, PX1, PY1, PY0), interpolation="bilinear")
    fig.axes[-1].axis("off")
    T(0.37, 0.90, "Conclusión", 22, CY, "bold")
    items = [
        (0.5, "Qué se reconstruyó:", YE, f"el contorno completo del keru ({NPTS} puntos, 5 splines abiertos) y 4 zonas dañadas: A rotura en V, B muesca trasera, C desprendimiento (spline cerrado) y D pequeña muesca."),
        (4.0, "Utilidad:", YE, "con pocos puntos, una curva suave que pasa por los datos, cierra lo faltante y se anima desde el modelo."),
        (8.0, "Validación:", GR, f"quitando un punto sano y prediciéndolo con el resto: error medio {LOO2_MEAN:.1f} px (máx. {LOO2_MAX:.1f}) en todo el contorno."),
        (12.0, "Limitación:", RD, "foto en perspectiva y sin puntos dentro de las roturas: es una estimación, no el original."),
        (16.0, "Aplicación futura:", MG, "restauración virtual 3D (fotogrametría + perfil de revolución) e impresión 3D."),
    ]
    y = 0.82
    for (t0, lab, col, txt) in items:
        a = appear(t0, tl)
        T(0.37, y, lab, 15, col, "bold", alpha=a)
        wrapped = textwrap.fill(txt, width=78)
        T(0.37, y - 0.045, wrapped, 13.5, FG, alpha=a)
        y -= 0.045 + 0.043 * (wrapped.count("\n") + 1) + 0.022

def cierre(tt):                                               # la "cereza del pastel"
    tl = tt - 264
    ax = fig.add_axes((0.03, 0.07, 0.30, 0.84))
    ax.imshow(RECP, extent=(PX0, PX1, PY1, PY0), interpolation="bilinear"); ax.axis("off")
    T(0.37, 0.80, "Con métodos numéricos,", 27, CY, "bold", alpha=appear(0.1, tl))
    T(0.37, 0.69, "aplicando splines cúbicos,", 27, FG, "bold", alpha=appear(2.0, tl))
    T(0.37, 0.58, "podemos reconstruir el pasado.", 27, YE, "bold", alpha=appear(4.5, tl))
    T(0.37, 0.43, f"{NPTS} puntos de control bastaron para devolverle la forma", 16, FG, alpha=appear(6.3, tl))
    T(0.37, 0.385, "a un keru de Tiwanaku de hace más de mil años.", 16, FG, alpha=appear(6.3, tl))
    T(0.37, 0.30, "Las matemáticas también sirven para cuidar nuestra historia.", 14.5, DIM, alpha=appear(7.5, tl))
    T(0.37, 0.15, "¡Gracias!", 22, FG, "bold", alpha=appear(8.3, tl))
    T(0.37, 0.095, NOMBRE, 14, DIM, alpha=appear(8.6, tl))

def draw(tt):
    fig.clf(); fig.patch.set_facecolor(BG)
    topbar(tt)
    for (a, b, f) in [(0, 25, parte1), (25, 50, parte2), (50, 75, parte3), (75, 120, parte4), (120, 148, parte5),
                      (148, 160, n1), (160, 176, n2), (176, 184, n3), (184, 216, n4), (216, 242, n5), (242, 264, concl), (264, 275.01, cierre)]:
        if a <= tt < b:
            f(tt); fade_in(tt - a); break

# ----------------------------------------------------------------- SALIDA
# ---- velocidad por escena: (inicio, fin, factor) en "tiempo de guion"; factor > 1 = más rápido
VELOCIDAD = [(0, 25, 1.4), (25, 50, 1.4), (50, 75, 1.2), (75, 120, 1.8), (120, 148, 1.7), (148, 160, 1.6),
             (160, 176, 1.6), (176, 184, 1.5), (184, 216, 1.8), (216, 242, 1.6), (242, 264, 1.68), (264, 275, 1.0)]
VDUR = sum((b - a) / f for a, b, f in VELOCIDAD)               # duración real del video (s)

def vt2st(v):
    """tiempo del video -> tiempo de guion (aplica la aceleración por escena)"""
    acc = 0.0
    for a, b, f in VELOCIDAD:
        d = (b - a) / f
        if v < acc + d: return a + (v - acc) * f
        acc += d
    return VELOCIDAD[-1][1]

def render(out=OUT, a=0, b=None):
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-crf", "18", out], stdin=subprocess.PIPE)
    N = int(VDUR * FPS) if b is None else b
    for k in range(a, N):
        draw(vt2st(k / FPS)); fig.canvas.draw()
        ff.stdin.write(bytes(fig.canvas.buffer_rgba()))
        if k % 200 == 0: print(f"{k}/{N}", flush=True)
    ff.stdin.close(); ff.wait(); print("listo:", out)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) > 1 and args[0] == "render" and args[1].lower().endswith((".png", ".jpg", ".jpeg")):
        IMG_PATH = args[1]
    elif args and args[0] == "frames" and False: pass
    cargar(IMG_PATH); cargar_full()
    if args and args[0] == "part":          # part  A  B  salida.mp4   (frames A..B-1)
        render(args[3], int(args[1]), int(args[2]))
    elif args and args[0] == "frames":
        for s in args[1:]:
            draw(float(s)); fig.savefig(f"frame_{float(s):06.2f}.png", dpi=100, facecolor=BG)
    else:
        render()
