"""Generador del logo: trébol de 4 hojas.

Produce un tkinter.PhotoImage sin depender de PIL ni archivos externos.
Se usa como icono de la ventana y como emblema dentro de la UI.
"""
from __future__ import annotations

import tkinter as tk
from math import sqrt


# Paleta
_GREEN_DARK = "#1f6b3a"
_GREEN_MID = "#2e9b4e"
_GREEN_LIGHT = "#4ec26a"
_STEM = "#3a5a1f"
_BG = ""  # transparente


def _blend(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _leaf(x: float, y: float, cx: float, cy: float, r: float) -> bool:
    """Hoja en forma de corazón alrededor de (cx, cy)."""
    dx = x - cx
    dy = y - cy
    # Distancia normalizada para una forma de hoja de trébol (corazón invertido)
    # Ecuación: (x^2 + y^2 - r^2)^3 - x^2 * y^3 <= 0 (curva de corazón)
    a = dx * dx + dy * dy - r * r
    return a * a * a - dx * dx * dy * dy * dy <= 0


def build_clover_image(size: int = 64) -> tk.PhotoImage:
    """Construye la imagen del trébol de 4 hojas a resolución `size`."""
    img = tk.PhotoImage(width=size, height=size)
    # Fondo transparente: no pintamos los píxeles fuera del trébol.
    cx = size / 2.0
    cy = size / 2.0
    r = size * 0.22  # radio de cada hoja
    offset = size * 0.20  # separación de cada hoja respecto al centro

    # Centros de las 4 hojas (arriba, derecha, abajo, izquierda)
    # Rotadas 45° para que queden en diagonal (estilo trébol clásico)
    centers = [
        (cx - offset, cy - offset),  # sup-izq
        (cx + offset, cy - offset),  # sup-der
        (cx + offset, cy + offset),  # inf-der
        (cx - offset, cy + offset),  # inf-izq
    ]

    for py in range(size):
        for px in range(size):
            # Coordenadas centradas, invertir Y para que corazón apunte al centro
            inside = False
            shade_t = 0.0
            for (lcx, lcy) in centers:
                # Rotar la hoja para que la punta mire al centro
                ddx = px - lcx
                ddy = py - lcy
                # Vector hacia el centro
                vx = cx - lcx
                vy = cy - lcy
                vlen = sqrt(vx * vx + vy * vy) or 1.0
                ux, uy = vx / vlen, vy / vlen
                # Rotar punto para que "arriba" de la hoja apunte hacia el centro
                rx = ddx * uy - ddy * ux
                ry = ddx * ux + ddy * uy
                # Forma de corazón: (x^2+y^2-1)^3 - x^2*y^3 <= 0
                xs = rx / r
                ys = -ry / r  # invertir para que el corazón apunte hacia fuera del centro
                val = (xs * xs + ys * ys - 1) ** 3 - xs * xs * ys * ys * ys
                if val <= 0:
                    inside = True
                    # Sombreado radial dentro de la hoja
                    d = sqrt(ddx * ddx + ddy * ddy) / r
                    shade_t = max(shade_t, min(1.0, 1.0 - d * 0.6))
                    break
            if inside:
                # Mezcla oscuro→claro según shade_t
                color = _blend(_GREEN_DARK, _GREEN_LIGHT, shade_t)
                img.put(color, (px, py))

    # Tallo: línea diagonal desde centro hacia abajo-derecha
    stem_len = int(size * 0.22)
    for i in range(stem_len):
        x = int(cx + i * 0.4)
        y = int(cy + i * 0.9)
        if 0 <= x < size and 0 <= y < size:
            img.put(_STEM, (x, y))
            if x + 1 < size:
                img.put(_STEM, (x + 1, y))

    return img


def apply_window_icon(root: tk.Tk) -> tk.PhotoImage | None:
    """Aplica el logo como icono de la ventana. Devuelve la imagen para mantener la referencia."""
    try:
        icon = build_clover_image(64)
        root.iconphoto(True, icon)
        return icon
    except Exception:
        return None
