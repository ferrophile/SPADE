import torch
import numpy as np


def instances_to_boxes(instances_tensor, semantic_tensor, filter_classes):
    instances_fine_tensors = []
    instances_box_tensors = []
    semantic_class_lists = []

    instances_tensors = torch.split(instances_tensor, 1, dim=0)
    semantic_tensors = torch.split(semantic_tensor, 1, dim=0)
    instances_counts = torch.amax(instances_tensor, dim=(1, 2)).type(torch.int64).cpu().numpy() + 1

    for t, n, s in zip(instances_tensors, instances_counts, semantic_tensors):
        fine_tensor = torch.zeros(n, *t.shape[1:]).cuda()
        fine_tensor = fine_tensor.scatter_(0, t.type(torch.int64), torch.ones_like(t))
        fine_tensor = fine_tensor[:-1]  # remove background

        box_np = fine_tensor.cpu().numpy()
        rows = np.any(box_np, axis=2)
        cols = np.any(box_np, axis=1)
        semantic_classes = np.zeros(box_np.shape[0], dtype=int)

        for i in range(box_np.shape[0]):
            semantic_values = torch.masked_select(s, fine_tensor[i].type(torch.BoolTensor))
            semantic_class = torch.mode(semantic_values).values
            semantic_classes[i] = int(semantic_class.cpu().numpy())

            rmin, rmax = np.nonzero(rows[i])[0][[0, -1]]
            cmin, cmax = np.nonzero(cols[i])[0][[0, -1]]
            box_np[i, rmin:rmax+1, cmin:cmax+1] = 1

        if filter_classes:
            null_mask = (np.isin(semantic_classes, [3, 8, 10, 14]))
        else:
            null_mask = (semantic_classes != 0)

        semantic_classes = semantic_classes[null_mask]
        box_np = box_np[null_mask]
        fine_tensor = fine_tensor[null_mask]

        box_tensor = torch.Tensor(box_np).cuda()
        instances_fine_tensors.append(fine_tensor)
        instances_box_tensors.append(box_tensor)
        semantic_class_lists.append(semantic_classes)

    return instances_fine_tensors, instances_box_tensors, semantic_class_lists


'''
def boxes_to_labels(box_tensor, semantic_classes, nc=30):
    labels_tensor = torch.zeros(nc, *box_tensor.shape[1:]).cuda()

    semantic_tensor = torch.Tensor(semantic_classes).cuda().view(-1, 1, 1)
    semantic_indices = box_tensor * semantic_tensor
    semantic_indices = semantic_indices.type(torch.int64)

    labels_tensor = labels_tensor.scatter_add_(0, semantic_indices, torch.ones_like(box_tensor))
    labels_tensor = labels_tensor.type(torch.cuda.BoolTensor)
    return labels_tensor
'''


def boxes_to_labels(box_tensor, semantic_classes, nc=30):
    box_np = box_tensor.cpu().detach().numpy()

    labels_np = np.zeros((nc, *box_np.shape[1:]))
    for i, s in enumerate(semantic_classes):
        labels_np[s] += box_np[i]

    labels_tensor = torch.tensor(labels_np, dtype=torch.float).cuda()
    return labels_tensor
