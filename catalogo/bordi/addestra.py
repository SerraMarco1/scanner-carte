"""Addestra il modello dei bordi su scene finte e lo esporta in ONNX (catalogo/cache/bordi.onnx).

Uso: catalogo/cache/venv/Scripts/python catalogo/bordi/addestra.py [--scene=50000] [--epoche=12] [--riprendi=3]
--riprendi=N: N giri già fatti, riparte da catalogo/cache/bordi.pt (se l'addestramento si è interrotto).
Prima genera le scene una volta sola (catalogo/cache/bordi-scene.pkl, su tutti i core), poi addestra sul processore.
"""
import math
import multiprocessing as mp
import os
import pickle
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from modello import IN_H, IN_W, Net  # noqa: E402
from scene import CACHE, Scenes  # noqa: E402

arg = lambda k, d: next((a.split('=', 1)[1] for a in sys.argv if a.startswith(f'--{k}=')), d)
DATA = os.path.join(CACHE, 'bordi-scene.pkl')
OUT = os.path.join(CACHE, 'bordi.onnx')


def make(chunk):
    """Un pezzo di scene: fotogramma 480×640 come dal telefono, ridotto all'ingresso del modello, in JPEG."""
    start, n = chunk
    sc, r, out = Scenes(test=False), np.random.default_rng(1000 + start), []
    for i in range(n):
        img, quad, _ = sc(r, has_card=r.random() > 0.08)
        small = cv2.resize(img, (IN_W, IN_H), interpolation=cv2.INTER_AREA)
        out.append((cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(),
                    None if quad is None else quad * [IN_W / sc.W, IN_H / sc.H]))
    return out


class Data(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        buf, quad = self.items[i]
        img = cv2.cvtColor(cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = np.zeros((IN_H // 2, IN_W // 2), np.float32)
        if quad is not None:
            cv2.fillConvexPoly(mask, np.round(quad * 0.5 * 16).astype(np.int32), 1.0, cv2.LINE_AA, shift=4)
        if np.random.random() < 0.5:  # specchio: la maschera non cambia significato
            img, mask = img[:, ::-1], mask[:, ::-1]
        x = torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32))
        x = x * float(np.random.uniform(0.85, 1.15)) + float(np.random.uniform(-15, 15))
        return x.clamp(0, 255), torch.from_numpy(mask.copy())[None]


def main():
    n, epochs = int(arg('scene', 50000)), int(arg('epoche', 12))
    if not os.path.exists(DATA):
        t = time.time()
        with mp.Pool(max(1, os.cpu_count() - 2)) as pool:
            items = [x for part in pool.imap_unordered(make, [(s, min(500, n - s)) for s in range(0, n, 500)]) for x in part]
        with open(DATA, 'wb') as f:
            pickle.dump(items, f)
        print(f'scene generate: {len(items)} in {time.time() - t:.0f} s', flush=True)
    with open(DATA, 'rb') as f:
        items = pickle.load(f)
    torch.set_num_threads(max(1, os.cpu_count() - 4))
    dl = torch.utils.data.DataLoader(Data(items), batch_size=32, shuffle=True, num_workers=3, persistent_workers=True, drop_last=True)
    net, start = Net(), int(arg('riprendi', 0))
    if start:
        net.load_state_dict(torch.load(os.path.join(CACHE, 'bordi.pt')))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    total, step, t = epochs * len(dl), start * len(dl), time.time()
    for ep in range(start, epochs):
        net.train()
        run = 0.0
        for x, y in dl:
            for g in opt.param_groups:
                g['lr'] = 3e-3 * 0.5 * (1 + math.cos(math.pi * step / total))
            logit = net(x, logits=True)
            p = torch.sigmoid(logit)
            dice = 1 - (2 * (p * y).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
            loss = F.binary_cross_entropy_with_logits(logit, y) + dice.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
            step += 1
            if step % 200 == 0:
                print(f'epoca {ep + 1} passo {step}/{total} perdita {run / 200:.4f}', flush=True)
                run = 0.0
        torch.save(net.state_dict(), os.path.join(CACHE, 'bordi.pt'))
        export(net)


def export(net):
    net.eval()
    torch.onnx.export(net, (torch.zeros(1, 3, IN_H, IN_W),), OUT, input_names=['image'], output_names=['card'], opset_version=17, dynamo=False)
    print(f'esportato {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)', flush=True)


if __name__ == '__main__':
    main()
