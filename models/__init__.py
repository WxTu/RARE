from .edcoder import PreModel


def build_model(args):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    residual = args.residual
    attn_drop = args.attn_drop
    in_drop = args.in_drop
    norm = args.norm
    negative_slope = args.negative_slope
    encoder_type = args.encoder
    decoder_type = args.decoder
    mask_rate = args.mask_rate
    replace_rate = args.replace_rate
    activation = args.activation
    num_features = args.num_features
    momentum_rate = args.momentum_rate
    edcoder_rate = args.edcoder_rate
    t = args.t
    loss_r = args.loss_r
    loss_a = args.loss_a

    model = PreModel(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        residual=residual,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_rate=mask_rate,
        norm=norm,
        replace_rate=replace_rate,
        momentum_rate=momentum_rate,
        edcoder_rate=edcoder_rate,
        t=t,
        loss_r=loss_r,
        loss_a=loss_a
    )
    return model
