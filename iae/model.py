"""Deictic evidence networks: frame-level hand-candidate scoring trained with video-level (MIL) and
program-level supervision.

EvidenceNet   : per-frame MLP, max over hands, temporal smoothing of logits.
EvidenceNetT  : temporal convolutions over each (hand, candidate) feature sequence before scoring, so the
                score can depend on approach, dwell and retraction; optional within-frame competition.
"""
import torch, torch.nn as nn, torch.nn.functional as F

NEG = -1e4


class EvidenceNet(nn.Module):
    def __init__(self, nf=28, hidden=48, kernel=7, tau=2.0, task_dim=2):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nf + task_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.temporal = nn.Conv1d(1, 1, kernel, padding=kernel // 2)
        with torch.no_grad():
            self.temporal.weight.fill_(1.0 / kernel); self.temporal.bias.zero_()
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def frame_logits(self, X, M, task):
        T, Hh, K, _ = X.shape
        te = task.view(1, 1, 1, -1).expand(T, Hh, K, -1)
        g = self.mlp(torch.cat([X, te], -1)).squeeze(-1)
        g = g.masked_fill(~M[:, :, None], NEG)
        z = g.max(1).values
        z0 = torch.where(z > NEG / 2, z, torch.full_like(z, -6.0))
        return self.temporal(z0.t().unsqueeze(1)).squeeze(1).t()

    def video_scores(self, z, topk=5):
        k = min(topk, z.shape[0])
        return z.topk(k, dim=0).values.mean(0) + self.bias + 2.0

    def forward(self, X, M, task):
        z = self.frame_logits(X, M, task)
        return z, self.video_scores(z, getattr(self, 'topk', 5))


class EvidenceNetT(EvidenceNet):
    def __init__(self, nf=28, hidden=48, kernel=5, task_dim=2, compete=False, dropout=0.1):
        super().__init__(nf, hidden, 7, 2.0, task_dim)
        self.conv = nn.Sequential(nn.Conv1d(nf + task_dim + 1, hidden, kernel, padding=kernel // 2), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2), nn.ReLU(), nn.Conv1d(hidden, 1, 1))
        self.compete = compete

    def frame_logits(self, X, M, task):
        T, Hh, K, Fdim = X.shape
        te = task.view(1, 1, 1, -1).expand(T, Hh, K, -1)
        m = M[:, :, None, None].expand(T, Hh, K, 1).float()
        x = torch.cat([X, te, m], -1).permute(1, 2, 3, 0).reshape(Hh * K, Fdim + te.shape[-1] + 1, T)
        g = self.conv(x).reshape(Hh, K, T).permute(2, 0, 1)
        g = g.masked_fill(~M[:, :, None], NEG)
        z = g.max(1).values
        z = torch.where(z > NEG / 2, z, torch.full_like(z, -6.0))
        if self.compete:
            z = z - torch.logsumexp(z, 1, keepdim=True) + torch.logsumexp(z, 1, keepdim=True).detach()
        return z
