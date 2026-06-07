# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax_bisect


def entmax_alpha_transform(x: torch.Tensor, alpha: float = 1.5, dim: int = -1,
                           n_iter: int = 25, tol: float = 1e-5):
    if alpha == 1.0:
        return torch.softmax(x, dim=dim)
    return entmax_bisect(x, alpha=alpha, dim=dim, n_iter=n_iter)


class ASHA:
    def __init__(self, mu=0.85, entmax_alpha: float = None, neumann_K: int = 4):
        self.mu = float(mu)
        self.entmax_alpha = entmax_alpha
        self.neumann_K = neumann_K
        self.R_tilde = None
        self.R_tilde_csr = None
        self.R_tilde_t_csr = None
        self.Cu, self.Ci = None, None
        self.Hu, self.Hi = None, None
        self.Su, self.Si = None, None
        self._active_use_neumann = False

    def _spmm(self, A, X):
        return A.matmul(X)

    def prepare(self, R_tilde, R_tilde_csr, R_tilde_t_csr, active_use_neumann: bool):
        self.R_tilde = R_tilde
        self.R_tilde_csr = R_tilde_csr
        self.R_tilde_t_csr = R_tilde_t_csr
        self._active_use_neumann = bool(active_use_neumann)
        if self._active_use_neumann:
            self.Cu, self.Ci = None, None
            self.Hu, self.Hi = None, None
            self.Su, self.Si = None, None
            return

        self.Cu = self.R_tilde @ self.R_tilde.t()
        self.Ci = self.R_tilde.t() @ self.R_tilde
        mu2 = self.mu * self.mu
        num_active_u, num_active_i = self.R_tilde.shape
        dev = self.R_tilde.device
        Iu = torch.eye(num_active_u, device=dev)
        Ii = torch.eye(num_active_i, device=dev)
        self.Hu = torch.linalg.solve(Iu - mu2 * self.Cu, mu2 * self.Cu)
        self.Hi = torch.linalg.solve(Ii - mu2 * self.Ci, mu2 * self.Ci)
        if self.entmax_alpha is not None:
            self.Su = entmax_alpha_transform(self.Hu, alpha=self.entmax_alpha, dim=1)
            self.Si = entmax_alpha_transform(self.Hi, alpha=self.entmax_alpha, dim=1)
        else:
            self.Su, self.Si = None, None

    def _op_Cu(self, T: torch.Tensor) -> torch.Tensor:
        if self.R_tilde_csr is not None and self.R_tilde_t_csr is not None:
            return self._spmm(self.R_tilde_csr, self._spmm(self.R_tilde_t_csr, T))
        return self.R_tilde @ (self.R_tilde.t() @ T)

    def _op_Ci(self, T: torch.Tensor) -> torch.Tensor:
        if self.R_tilde_csr is not None and self.R_tilde_t_csr is not None:
            return self._spmm(self.R_tilde_t_csr, self._spmm(self.R_tilde_csr, T))
        return self.R_tilde.t() @ (self.R_tilde @ T)

    def apply_Hu(self, B: torch.Tensor) -> torch.Tensor:
        if not self._active_use_neumann and self.Hu is not None:
            return self.Hu @ B
        mu2 = self.mu * self.mu
        Z = torch.zeros_like(B)
        T = B
        for _ in range(max(1, self.neumann_K)):
            T = mu2 * self._op_Cu(T)
            Z = Z + T
        return Z

    def apply_Hi(self, B: torch.Tensor) -> torch.Tensor:
        if not self._active_use_neumann and self.Hi is not None:
            return self.Hi @ B
        mu2 = self.mu * self.mu
        Z = torch.zeros_like(B)
        T = B
        for _ in range(max(1, self.neumann_K)):
            T = mu2 * self._op_Ci(T)
            Z = Z + T
        return Z


class DFF(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = float(beta)

    def forward(self, Eu_act: torch.Tensor, Ei_act: torch.Tensor, asha: ASHA,
                eta_u: torch.Tensor, eta_i: torch.Tensor,
                gamma_u: torch.Tensor, gamma_i: torch.Tensor):
        beta2 = self.beta * self.beta

        if asha.R_tilde_csr is not None:
            RtEi = asha._spmm(asha.R_tilde_csr, Ei_act)
            RtT_Eu = asha._spmm(asha.R_tilde_t_csr, Eu_act)
        elif asha.R_tilde.is_sparse:
            RtEi = torch.sparse.mm(asha.R_tilde, Ei_act)
            RtT_Eu = torch.sparse.mm(asha.R_tilde.t(), Eu_act)
        else:
            RtEi = asha.R_tilde @ Ei_act
            RtT_Eu = asha.R_tilde.t() @ Eu_act

        Fu = (1.0 - eta_u) * Eu_act + (eta_u * beta2) * RtEi
        Fi = (1.0 - eta_i) * Ei_act + (eta_i * beta2) * RtT_Eu

        if asha.Su is not None and asha.Si is not None:
            Zu_h = asha.Su @ Fu
            Zi_h = asha.Si @ Fi
        else:
            Zu_h = asha.apply_Hu(Fu)
            Zi_h = asha.apply_Hi(Fi)

        Zu = (1.0 - gamma_u) * Eu_act + gamma_u * Zu_h
        Zi = (1.0 - gamma_i) * Ei_act + gamma_i * Zi_h

        return F.normalize(Fu, dim=1), F.normalize(Fi, dim=1), \
               F.normalize(Zu, dim=1), F.normalize(Zi, dim=1)


class DASH(nn.Module):
    def __init__(self, U, I, d=64, beta=1.0, mu=0.85, device="cuda", entmax_alpha: float = None,
                 cl_weight: float = 0.0, cl_tau: float = 0.2,
                 init_mode: str = "xavier", init_std: float = 0.1,
                 param_reg: float = 1e-6,
                 use_neumann: bool = False, neumann_K: int = 4, sparse_rtilde: bool = True,
                 lightgcn_layers: int = 2):
        super().__init__()
        self.U, self.I, self.d = U, I, d
        self.beta, self.mu = float(beta), float(mu)
        self.device = torch.device(device)

        self.Eu = nn.Parameter(torch.empty(U, d))
        self.Ei = nn.Parameter(torch.empty(I, d))
        if str(init_mode).lower() == "normal":
            nn.init.normal_(self.Eu, std=init_std)
            nn.init.normal_(self.Ei, std=init_std)
        else:
            nn.init.xavier_uniform_(self.Eu)
            nn.init.xavier_uniform_(self.Ei)

        self.eta_u = nn.Parameter(torch.tensor(0.5))
        self.eta_i = nn.Parameter(torch.tensor(0.5))
        self.gamma_u = nn.Parameter(torch.tensor(0.5))
        self.gamma_i = nn.Parameter(torch.tensor(0.5))

        self.entmax_alpha = entmax_alpha
        self.cl_weight = cl_weight
        self.cl_tau = cl_tau
        self.param_reg = param_reg
        self.lightgcn_layers = lightgcn_layers
        self.asha = ASHA(mu=mu, entmax_alpha=entmax_alpha, neumann_K=neumann_K)
        self.dff = DFF(beta=beta)

        self.MLPc = nn.Sequential(
            nn.Linear(d, d), nn.ELU(), nn.Linear(d, d)
        )
        self.MLPs = nn.Sequential(
            nn.Linear(d, d), nn.ELU(), nn.Linear(d, d)
        )

        self.use_neumann = use_neumann
        self._active_use_neumann = use_neumann
        self.neumann_K = neumann_K
        self.sparse_rtilde = sparse_rtilde

        self.R = None
        self.R_tilde = None
        self.R_tilde_csr = None
        self.R_tilde_t_csr = None
        self.active_user_idx, self.active_item_idx = None, None
        self._u_lookup, self._i_lookup = None, None

    def _spmm(self, A, X):
        return A.matmul(X)

    @torch.no_grad()
    def prepare_graph(self, R: torch.Tensor, user_idx=None, item_idx=None, use_neumann_override=None):
        if not torch.is_tensor(R):
            R = torch.tensor(R, device=self.device)
        dev = R.device
        if user_idx is None:
            user_idx = torch.arange(self.U, device=dev)
        if item_idx is None:
            item_idx = torch.arange(self.I, device=dev)
        self.active_user_idx, self.active_item_idx = user_idx, item_idx
        num_active_u, num_active_i = user_idx.numel(), item_idx.numel()
        self._u_lookup = torch.full((self.U,), -1, device=dev, dtype=torch.long)
        self._i_lookup = torch.full((self.I,), -1, device=dev, dtype=torch.long)
        self._u_lookup[user_idx] = torch.arange(num_active_u, device=dev)
        self._i_lookup[item_idx] = torch.arange(num_active_i, device=dev)
        self._active_use_neumann = self.use_neumann if use_neumann_override is None else bool(use_neumann_override)
        if R.shape != (num_active_u, num_active_i):
            R = R[user_idx][:, item_idx]
        self.R = R
        if R.is_sparse:
            du = torch.sparse.sum(R, dim=1).to_dense()
            di = torch.sparse.sum(R, dim=0).to_dense()
        else:
            du = R.sum(dim=1)
            di = R.sum(dim=0)
        su = torch.where(du > 0, du.rsqrt(), torch.zeros_like(du))
        si = torch.where(di > 0, di.rsqrt(), torch.zeros_like(di))
        if self._active_use_neumann:
            if R.is_sparse:
                R_coo = R.coalesce()
            else:
                R_coo = R.to_sparse().coalesce()
            idx = R_coo.indices()
            vals = R_coo.values()
            scaled_vals = vals * su[idx[0]] * si[idx[1]]
            self.R_tilde = torch.sparse_coo_tensor(
                idx, scaled_vals, (num_active_u, num_active_i), device=dev
            ).coalesce()
            try:
                self.R_tilde_csr = self.R_tilde.to_sparse_csr()
                self.R_tilde_t_csr = self.R_tilde.t().to_sparse_csr()
            except Exception:
                self.R_tilde_csr = None
                self.R_tilde_t_csr = None
        else:
            self.R_tilde = (su[:, None] * R * si[None, :])
            if self.sparse_rtilde:
                self.R_tilde_csr = self.R_tilde.to_sparse_csr()
                self.R_tilde_t_csr = self.R_tilde.t().to_sparse_csr()
            else:
                self.R_tilde_csr = None
                self.R_tilde_t_csr = None
        self.asha.prepare(self.R_tilde, self.R_tilde_csr, self.R_tilde_t_csr, self._active_use_neumann)

    def z_embeddings(self):
        gamma_u, gamma_i = torch.sigmoid(self.gamma_u), torch.sigmoid(self.gamma_i)
        eta_u, eta_i = torch.sigmoid(self.eta_u), torch.sigmoid(self.eta_i)
        Eu_act = self.Eu[self.active_user_idx]
        Ei_act = self.Ei[self.active_item_idx]
        return self.dff(Eu_act, Ei_act, self.asha, eta_u, eta_i, gamma_u, gamma_i)

    def get_lightgcn_embedding(self):
        Eu_act = self.Eu[self.active_user_idx]
        Ei_act = self.Ei[self.active_item_idx]

        all_emb_u = [Eu_act]
        all_emb_i = [Ei_act]

        curr_u, curr_i = Eu_act, Ei_act
        for _ in range(self.lightgcn_layers):
            if self.R_tilde_csr is not None:
                next_u = self._spmm(self.R_tilde_csr, curr_i)
                next_i = self._spmm(self.R_tilde_t_csr, curr_u)
            elif self.R_tilde.is_sparse:
                next_u = torch.sparse.mm(self.R_tilde, curr_i)
                next_i = torch.sparse.mm(self.R_tilde.t(), curr_u)
            else:
                next_u = self.R_tilde @ curr_i
                next_i = self.R_tilde.t() @ curr_u

            all_emb_u.append(next_u)
            all_emb_i.append(next_i)
            curr_u, curr_i = next_u, next_i

        Ztilde_u = torch.stack(all_emb_u, dim=1).mean(dim=1)
        Ztilde_i = torch.stack(all_emb_i, dim=1).mean(dim=1)
        return F.normalize(Ztilde_u, dim=1), F.normalize(Ztilde_i, dim=1)

    def calc_cl_loss(self, Ztilde, Z, side='user'):
        if side == 'user':
            Ztilde_u_c = F.normalize(self.MLPc(Ztilde), dim=1)
            Zu_s = F.normalize(self.MLPs(Z), dim=1)
            logits = torch.mm(Ztilde_u_c, Zu_s.t()) / self.cl_tau
        else:
            Ztilde_i_c = F.normalize(self.MLPc(Ztilde), dim=1)
            Zi_s = F.normalize(self.MLPs(Z), dim=1)
            logits = torch.mm(Ztilde_i_c, Zi_s.t()) / self.cl_tau

        target = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, target)

    @torch.no_grad()
    def select_hard_negatives(self, u_global, candidates_global):
        u_local = self._u_lookup[u_global]
        cand_local = self._i_lookup[candidates_global]
        _, _, Zu, Zi = self.z_embeddings()
        scores = (Zu[u_local].unsqueeze(1) * Zi[cand_local]).sum(dim=-1)
        best_indices = torch.argmax(scores, dim=1)
        return candidates_global.gather(1, best_indices.unsqueeze(1)).squeeze(1)

    def forward(self, u, ip, ineg):
        u_local = self._u_lookup[u]
        ip_local = self._i_lookup[ip]
        ineg_local = self._i_lookup[ineg]
        mask = (u_local >= 0) & (ip_local >= 0) & (ineg_local >= 0)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        u_local, ip_local, ineg_local = u_local[mask], ip_local[mask], ineg_local[mask]

        Fu, Fi, Zu, Zi = self.z_embeddings()

        s_pos = (Zu[u_local] * Zi[ip_local]).sum(dim=1)
        s_neg = (Zu[u_local] * Zi[ineg_local]).sum(dim=1)
        bpr = -F.logsigmoid(s_pos - s_neg).mean()

        cl_loss = torch.tensor(0.0, device=self.device)
        if self.cl_weight > 0.0:
            Ztilde_u, Ztilde_i = self.get_lightgcn_embedding()

            uniq_u = torch.unique(u_local)
            uniq_i = torch.unique(torch.cat([ip_local, ineg_local]))

            if len(uniq_u) > 1:
                cl_loss = cl_loss + self.calc_cl_loss(Ztilde_u[uniq_u], Zu[uniq_u], side='user')
            if len(uniq_i) > 1:
                cl_loss = cl_loss + self.calc_cl_loss(Ztilde_i[uniq_i], Zi[uniq_i], side='item')

        reg = self.Eu.norm(p=2).pow(2) + self.Ei.norm(p=2).pow(2)
        return bpr + self.cl_weight * cl_loss + self.param_reg * reg
