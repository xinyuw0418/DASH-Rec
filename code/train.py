# -*- coding: utf-8 -*-
import os
import warnings
import torch

from data_utils import load_pairs_adjlist, build_index, csr_from_pairs, sample_subgraph
from evaluate import evaluate
from model import DASH
from utils import HAS_PROFILER, ProfilerActivity, profile, set_seed


def train(args):
    warnings.filterwarnings(
        "ignore",
        message="Sparse CSR tensor support is in beta state.*",
        category=UserWarning,
    )
    set_seed(args.seed)
    device = "cuda"

    train_pairs = load_pairs_adjlist(os.path.join(args.data_dir, "train.txt"))
    test_pairs = load_pairs_adjlist(os.path.join(args.data_dir, "test.txt"))
    U, I, train_pairs_idx, test_pairs_idx, train_ui, test_ui = build_index(train_pairs, test_pairs)

    model = DASH(
        U, I, d=args.dim, beta=args.beta, mu=args.mu, device=device,
        entmax_alpha=args.entmax_alpha,
        cl_weight=args.cl_weight, cl_tau=args.cl_tau,
        init_mode=args.init, init_std=args.init_std,
        param_reg=args.param_reg, use_neumann=args.use_neumann,
        neumann_K=args.neumann_K, sparse_rtilde=args.sparse_rtilde,
        lightgcn_layers=args.lightgcn_layers,
    ).to(device)

    R_full = csr_from_pairs(U, I, train_pairs_idx, device=device, return_sparse=True)
    full_users = torch.arange(U, device=device)
    full_items = torch.arange(I, device=device)

    train_tensor = torch.tensor(train_pairs_idx, dtype=torch.long, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Start Training: U={U}, I={I}, Batch={args.batch_size}")

    did_profile_flops = False
    best_ndcg = float("-inf")
    best_epoch = 0
    stale_evals = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = args.steps_per_epoch

        for _ in range(steps):
            seeds, sub_items, sub_edges = sample_subgraph(
                train_tensor, train_ui, U, I,
                user_sample=args.subgraph_users,
                item_sample=args.subgraph_items,
                item_pad=args.subgraph_item_pad,
                device=device,
            )

            if len(sub_edges) == 0:
                continue

            u_map = torch.full((U,), -1, device=device)
            u_map[seeds] = torch.arange(len(seeds), device=device)
            i_map = torch.full((I,), -1, device=device)
            i_map[sub_items] = torch.arange(len(sub_items), device=device)

            local_u = u_map[sub_edges[:, 0]]
            local_i = i_map[sub_edges[:, 1]]

            R_sub = torch.zeros((len(seeds), len(sub_items)), device=device)
            R_sub[local_u, local_i] = 1.0

            model.prepare_graph(
                R_sub, user_idx=seeds, item_idx=sub_items,
                use_neumann_override=args.use_neumann,
            )

            idx = torch.randint(0, len(sub_edges), (args.batch_size,), device=device)
            batch_pos_edges = sub_edges[idx]
            batch_u = batch_pos_edges[:, 0]
            batch_ip = batch_pos_edges[:, 1]

            if args.dns_k > 1:
                batch_candidates = sub_items[
                    torch.randint(0, len(sub_items), (args.batch_size, args.dns_k), device=device)
                ]
                batch_ineg = model.select_hard_negatives(batch_u, batch_candidates)
            else:
                batch_ineg = sub_items[
                    torch.randint(0, len(sub_items), (args.batch_size,), device=device)
                ]

            profile_now = args.report_flops and (not did_profile_flops) and HAS_PROFILER
            if profile_now:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    with_flops=True,
                    record_shapes=False,
                ) as prof:
                    opt.zero_grad()
                    loss = model(batch_u, batch_ip, batch_ineg)
                    loss.backward()
                    opt.step()
                total_flops = sum(e.flops for e in prof.key_averages() if e.flops is not None)
                print(f"[FLOPs] 1 batch forward+backward: {total_flops/1e9:.2f} GFLOPs")
                did_profile_flops = True
            else:
                opt.zero_grad()
                loss = model(batch_u, batch_ip, batch_ineg)
                loss.backward()
                opt.step()
            total_loss += loss.item()

        metrics_str = ""
        if epoch % args.eval_every == 0:
            model.eval()
            model.prepare_graph(
                R_full, user_idx=full_users, item_idx=full_items,
                use_neumann_override=args.eval_use_neumann,
            )
            r20, n20 = evaluate(model, train_ui, test_ui, K=20)
            metrics_str = f" | \033[0;31mR@20: {r20:.4f} N@20: {n20:.4f}\033[0m"
            if args.early_stop_patience > 0:
                if n20 > best_ndcg + args.early_stop_delta:
                    best_ndcg = n20
                    best_epoch = epoch
                    stale_evals = 0
                else:
                    stale_evals += 1

        print(f"Epoch {epoch:3d} | Loss: {total_loss/steps:.4f}{metrics_str}")
        if args.early_stop_patience > 0 and stale_evals >= args.early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best N@20: {best_ndcg:.4f} at epoch {best_epoch}."
            )
            break
