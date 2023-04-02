import torch
import numpy as np
from dgl.data import WikiCSDataset

GRAPH_DICT = {
    'wiki': WikiCSDataset,
}


def select_DatasSet(labels):
    label_set = {}
    for i, label in enumerate(labels, 0):
        if label not in label_set.keys():
            label_set[label] = [i]
        else:
            label_set[label].extend([i])
    train_id = []
    valid_id = []
    test_id = []
    for key, value in label_set.items():
        test_id.extend(value[:int(len(value) * 0.86)])
        valid_id.extend(value[int(len(value) * 0.86): int(len(value) * 0.93)])
        train_id.extend(value[int(len(value) * 0.93):])

    return np.array(train_id), np.array(valid_id), np.array(test_id)


def load_transductive_dataset(dataset_name):
    dataset = GRAPH_DICT[dataset_name]()
    if dataset_name in ['wiki']:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
        labels = graph.ndata['label'].numpy()
        num_nodes = graph.num_nodes()
        train_idx, val_idx, test_idx = select_DatasSet(labels)
        train_idx = torch.as_tensor(train_idx)
        val_idx = torch.as_tensor(val_idx)
        test_idx = torch.as_tensor(test_idx)
        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx.long(), True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx.long(), True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx.long(), True)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    else:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()

    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)