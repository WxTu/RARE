import torch.nn
from utils import *
from data_util import load_transductive_dataset
from models import build_model


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
mean_loss = torch.nn.MSELoss(reduction='mean')
none_loss = torch.nn.MSELoss(reduction='none')


def pretrain(model, linear_model, graph, feat, optimizer, max_epoch, device, scheduler, num_classes, lr_f,
             weight_decay_f, max_epoch_f, linear_prob, logger=None, mse_mean=mean_loss, mse_none=none_loss):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)
    epoch_iter = tqdm(range(max_epoch))
    acc_list = []
    estp_acc_list = []
    for epoch in epoch_iter:
        model.train()
        loss, loss_dict, loss_align, rec_loss = model(graph, x, mse_mean, mse_none)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
        if logger is not None:
            loss_dict["lr"] = get_current_lr(optimizer)
            logger.note(loss_dict, step=epoch)

        final_acc, estp_acc = node_classification_evaluation(model, linear_model, graph, x, num_classes, lr_f,
                                                             weight_decay_f, max_epoch_f, device, linear_prob, mute=True)
        acc_list.append(final_acc * 100)
        estp_acc_list.append(estp_acc * 100)

    return model


def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    max_epoch_f = args.max_epoch_f
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    replace_rate = args.replace_rate
    optim_type = args.optimizer
    lr = args.lr
    weight_decay = args.weight_decay
    lr_f = args.lr_f
    weight_decay_f = args.weight_decay_f
    linear_prob = args.linear_prob
    load_model = args.load_model
    logs = args.logging
    use_scheduler = args.scheduler

    graph, (num_features, num_classes) = load_transductive_dataset(dataset_name)
    args.num_features = num_features

    for i, seed in enumerate(args.seeds):
        print(f"####### Run {i} for seed {seed}")
        set_random_seed(seed)

        if logs:
            logger = TBLogger(
                name=f"{dataset_name}__rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{decoder_type}")
        else:
            logger = None

        model = build_model(args)
        model.to(device)
        optimizer = create_optimizer(optim_type, model, lr, weight_decay)

        linear_model = LogisticRegression(args.num_hidden, num_classes)
        if use_scheduler:
            logging.info("Use schedular")

            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)
        else:
            scheduler = None

        x = graph.ndata["feat"]
        if not load_model:
            model = pretrain(
                model, linear_model, graph, x,
                optimizer, max_epoch,
                device, scheduler, num_classes,
                lr_f, weight_decay_f, max_epoch_f, linear_prob,
                logger, mean_loss, none_loss)

            model = model.cpu()

        final_acc, estp_acc = node_classification_evaluation(model, linear_model, graph, x, num_classes, lr_f,
                                                             weight_decay_f, max_epoch_f, device, linear_prob)

        output_file = open("./{}_final_result.csv".format(args.dataset), "a+")

        output_file.write('Settings: {}'.format(args))
        output_file.write('\n')
        output_file.write(f'final_acc: {final_acc:.4f}, early-stopping_acc: {estp_acc:.4f}, seed:{seed}')
        output_file.write('\n')
        if logger is not None:
            logger.finish()


if __name__ == "__main__":
    args = build_args()
    dataset = ['wiki']
    for name in dataset:
        args.dataset = name
        if args.use_cfg:
            args = load_best_configs(args, "configs.yml")
        args.seeds = [0]
        main(args)
