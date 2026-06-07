# -*- coding: utf-8 -*-
import math
import numpy as np
import torch


def evaluate(model, train_ui, test_ui, K=20, batch_items=4096):
    model.eval()
    with torch.no_grad():
        _, _, Zu, Zi = model.z_embeddings()

        device = Zu.device
        I = model.I
        users = list(test_ui.keys())

        R, N = [], []
        batch_u = max(1, batch_items)
        for start in range(0, len(users), batch_u):
            u_batch = users[start:start + batch_u]
            if not u_batch:
                continue
            u_idx = torch.tensor(u_batch, device=device, dtype=torch.long)
            scores = Zu[u_idx] @ Zi.t()
            for row, u in enumerate(u_batch):
                items = train_ui.get(u, set())
                if items:
                    idx = torch.tensor(list(items), device=device, dtype=torch.long)
                    scores[row, idx] = float("-inf")

            topk = torch.topk(scores, k=min(K, I), dim=1).indices
            for row, u in enumerate(u_batch):
                gold = test_ui.get(u, set())
                if len(gold) == 0:
                    continue
                gold_set = set(gold)
                hits = [1 if int(i.item()) in gold_set else 0 for i in topk[row][:K]]
                nhits = sum(hits)
                R.append(nhits / len(gold_set))
                dcg = 0.0
                for rank, h in enumerate(hits, start=1):
                    if h:
                        dcg += 1.0 / math.log2(rank + 1.0)
                ideal = sum(
                    1.0 / math.log2(r + 1.0)
                    for r in range(1, min(K, len(gold_set)) + 1)
                )
                N.append(dcg / (ideal + 1e-12))

        return float(np.mean(R)), float(np.mean(N))
