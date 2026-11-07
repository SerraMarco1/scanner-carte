"""Modello che trova la carta nell'inquadratura: dice per ogni punto se è carta (maschera), poi il contorno
della maschera dà i 4 angoli con le stesse regole di web/bordi.js.

Ingresso: immagine RGB 192×256 (verticale), valori 0-255. Uscita: probabilità «carta» 96×128.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

IN_W, IN_H = 192, 256


def block(cin, cout):
    """Convoluzione separabile 3×3: poco calcolo anche a metà risoluzione."""
    return nn.Sequential(nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
                         nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Net(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        import torchvision
        f = torchvision.models.mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None).features
        self.s2, self.s4, self.s8, self.s16, self.s32 = f[0], f[1], f[2:4], f[4:9], f[9:]
        c = 32
        self.l32, self.l16, self.l8, self.l4, self.l2 = (nn.Conv2d(n, c, 1) for n in (576, 48, 24, 16, 16))
        self.m16, self.m8, self.m4, self.m2 = block(c, c), block(c, c), block(c, c), block(c, c)
        self.out = nn.Conv2d(c, 1, 1)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(self, x, logits=False):
        x = (x - self.mean) / self.std
        s2 = self.s2(x)
        s4 = self.s4(s2)
        s8 = self.s8(s4)
        s16 = self.s16(s8)
        s32 = self.s32(s16)
        up = lambda t: F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False)
        p = self.m16(self.l16(s16) + up(self.l32(s32)))
        p = self.m8(self.l8(s8) + up(p))
        p = self.m4(self.l4(s4) + up(p))
        p = self.m2(self.l2(s2) + up(p))
        y = self.out(p)
        return y if logits else torch.sigmoid(y)


def order(p):
    s, d = p.sum(1), p[:, 1] - p[:, 0]
    return np.float32([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]])


def card_shaped(q):
    n = lambda a, b: np.linalg.norm(q[a] - q[b])
    r = (n(0, 1) + n(3, 2)) / (n(0, 3) + n(1, 2))
    return 0.55 < r < 0.9


def straighten(q, pts):
    """Angoli più precisi: per ogni lato una retta sui punti del contorno lontani dagli angoli (arrotondati) e dalle dita,
    poi gli incroci delle rette. Misurato: angoli entro il 3% da 70% a 80% delle scene. Come straighten in web/bordi.js."""
    lines = []
    for k in range(4):
        a, d = q[k], q[(k + 1) % 4] - q[k]
        n = np.linalg.norm(d)
        u = d / n
        t, off = (pts - a) @ u, np.abs((pts - a) @ np.float32([-u[1], u[0]]))
        sel = pts[(t > 0.15 * n) & (t < 0.85 * n) & (off < 0.04 * n + 2)]
        if len(sel) < 5:
            return q
        lines.append(cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())  # vx, vy, x0, y0
    out = []
    for k in range(4):
        (ax, ay, px, py), (bx, by, rx, ry) = lines[k - 1], lines[k]
        det = bx * ay - ax * by
        if abs(det) < 1e-6:
            return q
        s = (bx * (ry - py) - by * (rx - px)) / det  # incrocio: p + s·a = r + t·b
        out.append((px + s * ax, py + s * ay))
    out = np.float32(out)
    return out if np.abs(out - q).max() < 0.1 * IN_W else q


def quad_from_mask(prob, w, h):
    """Maschera di probabilità → 4 angoli in pixel di un'immagine w×h, o None. Come quadFromMask in web/bordi.js."""
    m = (cv2.resize(prob, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)
    cs = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.06 * IN_W * IN_H:
        return None
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    q = None
    for eps in (0.02, 0.03, 0.04, 0.05, 0.06, 0.08):  # il poligono a 4 lati più fedele all'involucro
        poly = cv2.approxPolyDP(hull, eps * peri, True)
        if len(poly) == 4:
            q = order(poly.reshape(4, 2).astype(np.float32))
            break
        if len(poly) < 4:
            break
    if q is None:
        q = order(cv2.boxPoints(cv2.minAreaRect(hull)))
    q = straighten(q, c.reshape(-1, 2).astype(np.float32))
    if not card_shaped(q):
        return None
    return q * [w / IN_W, h / IN_H]


class Finder:
    """Per misura.py: carica il modello ONNX e trova la carta in un'immagine BGR."""

    def __init__(self, path):
        import onnxruntime
        self.s = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])

    def __call__(self, img):
        h, w = img.shape[:2]
        x = cv2.cvtColor(cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        prob = self.s.run(None, {'image': x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
        return quad_from_mask(prob, w, h)
