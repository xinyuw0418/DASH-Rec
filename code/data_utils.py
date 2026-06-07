# -*- coding: utf-8 -*-
from collections import defaultdict
import torch


def load_pairs_adjlist(txt):
    pairs = []
    with open(txt, "r", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            if not toks:
                continue
            u = int(toks[0])
            for x in toks[1:]:
                pairs.append((u, int(x)))
    return list(set(pairs))


def build_index(train_pairs, test_pairs):
    users = sorted(set([u for u, _ in train_pairs] + [u for u, _ in test_pairs]))
    items = sorted(set([i for _, i in train_pairs] + [i for _, i in test_pairs]))
    umap = {u: idx for idx, u in enumerate(users)}
    imap = {i: idx for idx, i in enumerate(items)}
    train = [(umap[u], imap[i]) for u, i in train_pairs]
    test = [(umap[u], imap[i]) for u, i in test_pairs if u in umap and i in imap]
    U, I = len(umap), len(imap)
    train_ui = defaultdict(set)
    test_ui = defaultdict(set)
    for u, i in train:
        train_ui[u].add(i)
    for u, i in test:
        test_ui[u].add(i)
    return U, I, train, test, train_ui, test_ui


def csr_from_pairs(U, I, pairs, device, return_sparse=False):
    if len(pairs) == 0:
        if return_sparse:
            return torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long),
                torch.empty(0),
                (U, I),
                device=device,
            )
        return torch.zeros((U, I), device=device)

    u = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
    i = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)
    idx = torch.stack([u, i], dim=0)
    vals = torch.ones(len(u), device=device, dtype=torch.float32)
    sparse_t = torch.sparse_coo_tensor(idx, vals, (U, I), device=device).coalesce()

    if return_sparse:
        return sparse_t
    return sparse_t.to_dense()


def sample_subgraph(train_edges_tensor, train_ui, U, I, user_sample=800,
                    item_sample=None, item_pad=0, device="cpu"):
    all_users = torch.arange(U, device=device)
    num_u = min(user_sample, U)
    perm = torch.randperm(U, device=device)[:num_u]
    seeds = all_users[perm]

    mask = torch.isin(train_edges_tensor[:, 0], seeds)
    edges = train_edges_tensor[mask]
    items = torch.unique(edges[:, 1])

    if item_sample is not None and len(items) > item_sample:
        perm_items = torch.randperm(len(items), device=device)[:item_sample]
        items = items[perm_items]
        mask_edges = torch.isin(edges[:, 1], items)
        edges = edges[mask_edges]

    if item_pad > 0:
        pad_items = torch.randint(0, I, (item_pad,), device=device)
        items = torch.unique(torch.cat([items, pad_items]))

    return seeds, items, edges
