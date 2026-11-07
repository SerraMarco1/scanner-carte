"""Misura la ricerca dei bordi su scene finte mai viste (texture e carte di verifica).

Uso: catalogo/cache/venv/Scripts/python catalogo/bordi/misura.py [--n=400] [--modello=catalogo/cache/bordi.onnx] [--salva=10]
Giusta = ogni angolo entro il 3% della diagonale; usabile = entro il 6% (basta per riconoscerla).
Sulle scene senza carta, giusto = non trovare niente.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from scene import Scenes  # noqa: E402

arg = lambda k, d: next((a.split('=', 1)[1] for a in sys.argv if a.startswith(f'--{k}=')), d)


def order(p):
    s, d = p.sum(1), p[:, 1] - p[:, 0]
    return np.float32([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]])


def card_shaped(q):
    n = lambda a, b: np.linalg.norm(q[a] - q[b])
    r = (n(0, 1) + n(3, 2)) / (n(0, 3) + n(1, 2))
    return 0.55 < r < 0.9


def quad_of(c, area):
    """Come web/bordi.js: poligono a 4 lati convesso, oppure il rettangolo ruotato che il contorno riempie."""
    poly = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
    if len(poly) == 4 and cv2.isContourConvex(poly):
        return order(poly.reshape(4, 2).astype(np.float32))
    rr = cv2.minAreaRect(c)
    if area > 0.85 * rr[1][0] * rr[1][1]:
        return order(cv2.boxPoints(rr))
    return None


def classic(img):
    """Algoritmo attuale (web/bordi.js, ScannerPlugin.findCard): Canny a 3 soglie e contorni."""
    k = 480 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    best, best_area = None, small.shape[0] * small.shape[1] * 0.06
    for lo, hi in ((20, 60), (40, 120), (80, 200)):
        edges = cv2.dilate(cv2.Canny(gray, lo, hi), np.ones((3, 3), np.uint8))
        for c in cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(c)
            if area > best_area:
                q = quad_of(c, area)
                if q is not None and card_shaped(q):
                    best, best_area = q, area
    return None if best is None else best / k


def evaluate(name, find, n, save):
    sc = Scenes(test=True)
    r = np.random.default_rng(12345)  # stesse scene per ogni metodo
    groups, times, bad = {}, [], 0
    for i in range(n):
        img, quad, tags = sc(r, has_card=i % 10 != 9)
        t = cv2.getTickCount()
        q = find(img)
        times.append((cv2.getTickCount() - t) / cv2.getTickFrequency() * 1000)
        if quad is None:
            ok = usable = q is None
            key = 'senza carta'
        else:
            err = np.linalg.norm(q - quad, axis=1).max() / np.linalg.norm(quad[2] - quad[0]) if q is not None else 9
            ok, usable = err <= 0.03, err <= 0.06
            key = 'tinta unita' if tags['sfondo'] == 'tinta' else 'pila' if tags['pila'] else 'dita' if tags['dita'] else \
                'bustina' if tags['bustina'] else 'texture, 0-2 oggetti' if tags['oggetti'] <= 2 else 'texture, 3+ oggetti'
        g = groups.setdefault(key, [0, 0, 0])
        g[0] += ok
        g[1] += 1
        g[2] += usable
        if not ok and bad < save:
            bad += 1
            out = img.copy()
            if quad is not None:
                cv2.polylines(out, [quad.astype(np.int32)], True, (0, 255, 0), 2)
            if q is not None:
                cv2.polylines(out, [q.astype(np.int32)], True, (0, 0, 255), 2)
            os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'cache', 'bordi-errori'), exist_ok=True)
            cv2.imwrite(os.path.join(os.path.dirname(__file__), '..', 'cache', 'bordi-errori', f'{name}-{i}.jpg'), out)
    a, b, u = (sum(g[i] for g in groups.values()) for i in range(3))
    print(f'\n{name}: giuste {a}/{b} = {100 * a / b:.0f}%, usabili {100 * u / b:.0f}% - tempo mediano {np.median(times):.0f} ms')
    for k, (a, b, u) in sorted(groups.items()):
        print(f'  {k:22s} {a:4d}/{b:<4d} {100 * a / b:4.0f}%   usabili {100 * u / b:4.0f}%')


if __name__ == '__main__':
    n, save = int(arg('n', 400)), int(arg('salva', 0))
    evaluate('attuale', classic, n, save)
    model = arg('modello', None)
    if model:
        from modello import Finder
        evaluate('modello', Finder(model), n, save)
