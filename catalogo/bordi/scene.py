"""Scene finte per misurare e addestrare la ricerca dei bordi della carta.

Una carta in prospettiva su sfondi veri (texture CC0 di Poly Haven in catalogo/cache/sfondi), con intorno
altre carte, pile, oggetti e penne, e a volte dita, bustina, ombre, riflessi, sfocatura e JPEG.
Il 30% delle texture e delle carte serve solo alla verifica: il modello non le vede mai in addestramento.
"""
import glob
import hashlib
import math
import os

import cv2
import numpy as np

CACHE = os.path.join(os.path.dirname(__file__), '..', 'cache')
CARD = 88 / 63               # altezza / larghezza di una carta


def split(files, test):
    return [f for f in files if (int(hashlib.md5(os.path.basename(f).encode()).hexdigest(), 16) % 10 < 3) == test]


def rounded(card):
    """Carta BGRA con gli angoli arrotondati come quelle vere (raggio ≈ 4,5% della larghezza)."""
    h, w = card.shape[:2]
    a = np.zeros((h, w), np.uint8)
    r = max(2, int(0.045 * w))
    cv2.rectangle(a, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(a, (0, r), (w, h - r), 255, -1)
    for x, y in ((r, r), (w - r - 1, r), (r, h - r - 1), (w - r - 1, h - r - 1)):
        cv2.circle(a, (x, y), r, 255, -1, cv2.LINE_AA)
    if card.shape[2] == 4:
        a = np.minimum(a, card[:, :, 3])
    return np.dstack([card[:, :, :3], a])


def paste(img, src, quad, alpha=1.0):
    """Incolla src (BGRA) in img (float32 BGR) deformandolo sui 4 angoli quad (TL, TR, BR, BL)."""
    H, W = img.shape[:2]
    x0, y0 = np.clip(np.floor(quad.min(0)).astype(int) - 1, 0, [W, H])
    x1, y1 = np.clip(np.ceil(quad.max(0)).astype(int) + 2, 0, [W, H])
    if x1 <= x0 or y1 <= y0:
        return
    h, w = src.shape[:2]
    m = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), np.float32(quad - [x0, y0]))
    warped = cv2.warpPerspective(src, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    a = warped[:, :, 3:4].astype(np.float32) * (alpha / 255)
    roi = img[y0:y1, x0:x1]
    roi[:] = roi * (1 - a) + warped[:, :, :3].astype(np.float32) * a


def rect(cx, cy, w, h, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.float32([[cx + x * c - y * s, cy + x * s + y * c] for x, y in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))])


class Scenes:
    """Scene W×H (verticali 3:4 come i fotogrammi della fotocamera nativa)."""

    def __init__(self, test, W=480, H=640):
        self.W, self.H, self.k = W, H, W / 480
        self.tex = split(sorted(glob.glob(os.path.join(CACHE, 'sfondi', '*.jpg'))), test)
        self.cards = split(sorted(glob.glob(os.path.join(CACHE, 'img', '*', '*.webp'))), test)
        assert self.tex and self.cards, 'mancano catalogo/cache/sfondi o catalogo/cache/img'

    def card(self, r):
        c = cv2.imread(str(r.choice(self.cards)), cv2.IMREAD_UNCHANGED)
        if c.shape[2] == 3:
            c = cv2.cvtColor(c, cv2.COLOR_BGR2BGRA)
        return rounded(c)

    def shadow(self, img, quad, r, strength):
        m = np.zeros(img.shape[:2], np.float32)
        off = np.float32([r.uniform(-1, 1), r.uniform(0.3, 1.5)]) * 0.02 * self.W
        cv2.fillConvexPoly(m, (quad + off).astype(np.int32), 1.0, cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), r.uniform(2, 9) * self.k)
        img *= (1 - strength * m)[:, :, None]

    def background(self, r, kind):
        W, H = self.W, self.H
        if kind == 'tinta':  # tavolo bianco, grigio, scuro o colorato
            col = [r.choice([235, 245, 200, 150, 40, 20])] * 3 if r.random() < 0.7 else [int(v) for v in r.integers(0, 256, 3)]
            img = np.full((H, W, 3), col, np.float32) + np.float32(r.normal(0, 3, (H, W, 1)))
        else:
            t = cv2.imread(str(r.choice(self.tex)), cv2.IMREAD_REDUCED_COLOR_2 if W < 300 else cv2.IMREAD_COLOR)
            if r.random() < 0.5:
                t = cv2.rotate(t, cv2.ROTATE_90_CLOCKWISE)
            s = r.uniform(0.6, 2.0) * max(W / t.shape[1], H / t.shape[0])
            t = cv2.resize(t, (max(W, int(t.shape[1] * s)), max(H, int(t.shape[0] * s))), interpolation=cv2.INTER_AREA)
            y, x = r.integers(0, t.shape[0] - H + 1), r.integers(0, t.shape[1] - W + 1)
            img = t[y:y + H, x:x + W].astype(np.float32)
        return img * np.float32(r.uniform(0.8, 1.2, 3)) * r.uniform(0.7, 1.3)

    def __call__(self, r, has_card=True):
        """r = numpy Generator. Restituisce (immagine BGR uint8, angoli 4×2 o None, etichette per il resoconto)."""
        W, H, k = self.W, self.H, self.k
        while True:
            ch = r.uniform(0.42, 0.92) * H
            cw = ch / CARD
            cx, cy = W / 2 + r.uniform(-0.15, 0.15) * W, H / 2 + r.uniform(-0.12, 0.12) * H
            deg = float(np.clip(r.normal(0, 9), -30, 30))
            quad = rect(cx, cy, cw, ch, deg) + np.float32(r.uniform(-0.05, 0.05, (4, 2)) * ch)  # prospettiva
            lo, hi = quad.min(0), quad.max(0)
            if lo[0] >= 4 * k and lo[1] >= 4 * k and hi[0] <= W - 4 * k and hi[1] <= H - 4 * k:
                break
        kind = 'tinta' if r.random() < 0.15 else 'texture'
        img = self.background(r, kind)
        tags = {'sfondo': kind}
        # Oggetti intorno: altre carte, scatole o libri, penne.
        n = int(r.integers(0, 8))
        tags['oggetti'] = n
        for _ in range(n):
            kk = r.random()
            if kk < 0.45:
                s = r.uniform(0.6, 1.2)
                q = rect(r.uniform(-0.1, 1.1) * W, r.uniform(-0.1, 1.1) * H, cw * s, ch * s, r.uniform(-180, 180))
                self.shadow(img, q, r, r.uniform(0, 0.4))
                paste(img, self.card(r), q)
            elif kk < 0.75:
                q = rect(r.uniform(0, W), r.uniform(0, H), r.uniform(0.2, 0.8) * W, r.uniform(0.1, 0.6) * H, r.uniform(-90, 90))
                self.shadow(img, q, r, r.uniform(0, 0.5))
                obj = np.zeros((8, 8, 4), np.uint8)
                obj[:] = [*r.integers(0, 256, 3), 255]
                paste(img, obj, q)
            else:
                p = r.uniform(0, 1, 4) * [W, H, W, H]
                col = tuple(float(v) for v in r.integers(0, 256, 3))
                cv2.line(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), col, max(1, int(r.integers(3, 14) * k)), cv2.LINE_AA)
        if not has_card:
            return self.finish(img, r), None, tags
        # Pila: carte sotto quella da leggere, un po' spostate.
        tags['pila'] = r.random() < 0.2
        if tags['pila']:
            for _ in range(int(r.integers(1, 4))):
                q = rect(cx + r.uniform(-0.06, 0.06) * cw, cy + r.uniform(-0.06, 0.06) * ch, cw, ch, deg + r.uniform(-6, 6))
                paste(img, self.card(r), q)
        self.shadow(img, quad, r, r.uniform(0.1, 0.6))
        paste(img, self.card(r), quad)
        # Bustina trasparente: un velo appena più grande della carta.
        tags['bustina'] = r.random() < 0.25
        if tags['bustina']:
            c = quad.mean(0)
            veil = np.full((8, 8, 4), 255, np.uint8)
            paste(img, veil, c + (quad - c) * 1.035, alpha=r.uniform(0.06, 0.18))
        # Dita che tengono la carta: ellissi color pelle sul bordo.
        tags['dita'] = r.random() < 0.25
        if tags['dita']:
            skin = [r.uniform(60, 180), r.uniform(100, 200), r.uniform(150, 240)]
            for _ in range(int(r.integers(1, 3))):
                i, t = int(r.integers(0, 4)), r.uniform(0, 1)
                p = quad[i] * (1 - t) + quad[(i + 1) % 4] * t
                out = p - quad.mean(0)
                ang = math.degrees(math.atan2(out[1], out[0]))
                cv2.ellipse(img, (int(p[0]), int(p[1])), (int(r.uniform(0.25, 0.45) * cw), int(r.uniform(0.07, 0.11) * cw)), ang, 0, 360, skin, -1, cv2.LINE_AA)
        # Riflesso sulla carta.
        if r.random() < 0.4:
            g = np.zeros((H, W), np.float32)
            p = quad[int(r.integers(0, 4))] * 0.5 + quad.mean(0) * 0.5
            cv2.circle(g, (int(p[0]), int(p[1])), int(r.uniform(0.1, 0.35) * cw), 1.0, -1)
            g = cv2.GaussianBlur(g, (0, 0), r.uniform(5, 20) * k) * r.uniform(0.3, 0.9)
            img += g[:, :, None] * (255 - img)
        return self.finish(img, r), quad, tags

    def finish(self, img, r):
        # Luce non uniforme, ombra del telefono, sfocatura, rumore, compressione.
        W, H, k = self.W, self.H, self.k
        a = r.uniform(0, 2 * math.pi)
        g = ((np.arange(W, dtype=np.float32)[None, :] - W / 2) * math.cos(a) + (np.arange(H, dtype=np.float32)[:, None] - H / 2) * math.sin(a)) / max(W, H)
        img *= (1 + g * r.uniform(0, 0.8))[:, :, None]
        if r.random() < 0.2:
            m = np.zeros((H, W), np.float32)
            cv2.ellipse(m, (int(r.uniform(0, W)), int(r.uniform(0, H))), (int(r.uniform(0.2, 0.5) * W), int(r.uniform(0.2, 0.5) * H)), r.uniform(0, 180), 0, 360, 1.0, -1)
            img *= (1 - cv2.GaussianBlur(m, (0, 0), 25 * k) * r.uniform(0.2, 0.5))[:, :, None]
        s = r.uniform(0, 1.6) * k
        if s > 0.3:
            img = cv2.GaussianBlur(img, (0, 0), s)
        img += np.float32(r.normal(0, r.uniform(0, 6), img.shape))
        img = np.clip(img, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, int(r.integers(45, 95))])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
