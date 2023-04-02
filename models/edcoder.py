from typing import Optional
from itertools import chain
import torch
import torch.nn as nn
from utils import create_norm
from .gat import GAT
import torch.nn.functional as F


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead,
                 nhead_out, attn_drop, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    else:
        raise NotImplementedError

    return mod


def cos_loss(x, y, t=2):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    cos_m = (1 + (x * y).sum(dim=-1)) * 0.5
    loss = -torch.log(cos_m.pow_(t))
    return loss.mean(), loss


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            replace_rate: float = 0.1,
            momentum_rate: float = 0.1,
            edcoder_rate: float = 0.75,
            t: float = 1.,
            loss_r: float = 1.,
            loss_a: float = 1.
    ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._output_hidden_size = num_hidden
        self._edcoder_rate = edcoder_rate
        self._momentum_rate = momentum_rate
        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self.t = t
        self.loss_r = loss_r
        self.loss_a = loss_a
        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden

        self.online = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )
        self.target = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )
        self.recon_encoder = setup_module(
            m_type=decoder_type,
            enc_dec="encoding",
            in_dim=num_hidden,
            num_hidden=(round(num_hidden * self._edcoder_rate)),
            out_dim=(round(num_hidden * self._edcoder_rate)),
            num_layers=1,
            nhead=1,
            nhead_out=1,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
        )

        self.recon_decoder = nn.Linear(round(dec_in_dim * self._edcoder_rate), dec_in_dim, bias=False)
        self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)
        self.decoder = nn.Linear(dec_in_dim, in_dim, bias=False)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.rep_mask = nn.Parameter(torch.zeros(1, num_hidden))

        self._init_target()

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]

        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def momentum_update(self, base_momentum=0.1):
        for param_encoder, param_teacher in zip(self.online.parameters(),
                                                self.target.parameters()):
            param_teacher.data = param_teacher.data * base_momentum + \
                                 param_encoder.data * (1. - base_momentum)

    def _init_target(self):
        for param_encoder, param_teacher in zip(self.online.parameters(), self.target.parameters()):
            param_teacher.detach()
            param_teacher.data.copy_(param_encoder.data)
            param_teacher.requires_grad = False

    def forward(self, g, x, mse_mean, mse_none):
        loss, loss_align, rec_loss = self.mask_attr_prediction(g, x, mse_mean, mse_none)
        loss_item = {"loss": loss.item()}
        return loss, loss_item, loss_align, rec_loss

    def mask_attr_prediction(self, g, x, mse_mean, mse_none):
        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)
        use_g = pre_use_g
        enc_rep, all_hidden = self.online(use_g, use_x, return_hidden=True)

        with torch.no_grad():
            x_t = x.clone()
            x_t[keep_nodes] = 0.0
            x_t[keep_nodes] += self.enc_mask_token
            enc_rep_t, all_hidden_t = self.target(use_g, x_t, return_hidden=True)
            rep_t = enc_rep_t
            self.momentum_update(self._momentum_rate)

        rep = enc_rep
        rep = self.encoder_to_decoder(rep)

        rep[mask_nodes] = 0.
        rep[mask_nodes] += self.rep_mask
        rep = self.recon_encoder(pre_use_g, rep)
        rep = self.recon_decoder(rep)

        online = rep[mask_nodes]
        target = rep_t[mask_nodes]

        rep[keep_nodes] = 0.0
        recon = self.decoder(rep)
        x_init = x[mask_nodes]
        x_rec = recon[mask_nodes]

        loss_align = mse_mean(online, target)
        _ = mse_none(online, target)
        rec_loss, _ = cos_loss(x_rec, x_init, t=self.t)
        loss = self.loss_r * rec_loss + self.loss_a * loss_align

        return loss, loss_align, rec_loss

    def embed(self, g, x):
        rep = self.online(g, x)
        return rep