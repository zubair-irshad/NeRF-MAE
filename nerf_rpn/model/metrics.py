import torch.nn.functional as F
import torch
from torchmetrics import JaccardIndex
import matplotlib.pyplot as plt

# def mse(pred, target, valid_mask=None, reduction="mean"):
#     loss = F.mse_loss(pred, target, reduction="none")
#     if valid_mask is not None:
#         loss = loss * valid_mask
#     if reduction == "mean":
#         return loss.mean()
#     elif reduction == "sum":
#         return loss.sum()
#     else:
#         return loss


def intersection_over_union(pred, target, mask):
    # Example usage with JaccardIndex for multi-label segmentation
    jaccard = JaccardIndex(
        task="multiclass",
        num_classes=20,
        average="macro",
        ignore_index=None,
    ).to(pred.device)

    # Flatten both the predicted and target tensors
    pred = pred.view(-1)
    target = target.view(-1)

    # Apply the mask to the target and predicted tensors
    pred = pred[mask.view(-1) == 1]
    target = target[mask.view(-1) == 1]

    # Calculate the Jaccard Index
    jaccard_score = jaccard(pred, target)

    # print("Jaccard Index (IoU) for Multi-label Segmentation:", jaccard_score.item())

    # # Calculate the intersection and union of predicted and target regions
    # intersection = (pred == target).sum()
    # union = (pred + target).sum() - intersection

    # # Calculate IoU and handle the case of division by zero
    # iou = intersection.float() / (union.float() + 1e-8)

    return jaccard_score.item()


def pixel_accuracy(pred, target, mask):
    # Flatten both the predicted and target tensors
    pred = pred.view(-1)
    target = target.view(-1)

    # Apply the mask to the target and predicted tensors
    pred = pred[mask.view(-1) == 1]
    target = target[mask.view(-1) == 1]

    # Calculate the number of correctly classified pixels
    correct_pixels = (pred == target).sum()
    total_pixels = mask.sum()

    # Calculate pixel accuracy and handle the case of division by zero
    accuracy = correct_pixels.float() / (total_pixels.float() + 1e-8)

    return accuracy.item()


def mse(pred_rgb, target_rgb, valid_mask):
    mse = (pred_rgb - target_rgb) ** 2
    valid_mask = valid_mask.expand_as(mse)
    mse = mse[valid_mask]
    mse = torch.mean(mse)

    return mse


def psnr(image_pred, image_gt, valid_mask=None):
    return -10 * torch.log10(mse(image_pred, image_gt, valid_mask))


EVAL_CLASS_IDS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
]


import numpy as np
from sklearn.metrics import confusion_matrix as CM


class Evaluator(object):
    def __init__(self, labels=None):
        self.num_classes = len(labels)
        self.labels = (
            np.arange(self.num_classes) if labels is None else np.array(labels)
        )
        assert self.labels.shape[0] == self.num_classes
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, pred_label, gt_label):
        """Update per instance

        Args:
            pred_label (np.ndarray): (num_points)
            gt_label (np.ndarray): (num_points,)

        """
        # convert ignore_label to num_classes
        # refer to sklearn.metrics.confusion_matrix
        if np.all(gt_label < 0):
            print("Invalid label.")
            return
        gt_label[gt_label == -100] = self.num_classes
        confusion_matrix = CM(
            gt_label.flatten(), pred_label.flatten(), labels=self.labels
        )
        self.confusion_matrix += confusion_matrix

    def batch_update(self, pred_labels, gt_labels):
        assert len(pred_labels) == len(gt_labels)
        for pred_label, gt_label in zip(pred_labels, gt_labels):
            self.update(pred_label, gt_label)

    @property
    def overall_acc(self):
        return np.sum(np.diag(self.confusion_matrix)) / np.sum(self.confusion_matrix)

    @property
    def overall_iou(self):
        return np.nanmean(self.class_iou)

    @property
    def class_seg_acc(self):
        return [
            self.confusion_matrix[i, i] / np.sum(self.confusion_matrix[i])
            for i in range(self.num_classes)
        ]

    @property
    def class_iou(self):
        iou_list = []
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i]
            p = self.confusion_matrix[:, i].sum()
            g = self.confusion_matrix[i, :].sum()
            union = p + g - tp
            if union == 0:
                iou = float("nan")
            else:
                iou = tp / union
            iou_list.append(iou)
        return iou_list


class mIoULoss_new(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss_new, self).__init__()
        self.classes = n_classes

    def to_one_hot(self, tensor):
        n, h, w, d, _ = tensor.size()
        one_hot = (
            torch.zeros(n, h, w, d, self.classes)
            .to(tensor.device)
            .scatter_(-1, tensor.view(n, h, w, d, 1).long(), 1)
        )
        return one_hot

    def forward(self, inputs, target, mask):
        # inputs => N x H x W x D x L x Classes
        # target_oneHot => N x H x W x D x L x Classes
        # mask => N x H x W x D x L (boolean array with True for valid pixels and False for invalid/ignored pixels)

        N = inputs.size()[0]

        mask = mask.float()  # Convert mask to float
        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=-1)

        # Apply the mask to the target and inputs
        # mask = mask.unsqueeze(-1)  # Expand mask to match the channel dimension
        target_valid = target * mask
        inputs_valid = inputs * mask

        # Convert the target to one-hot format
        target_oneHot = self.to_one_hot(target_valid)

        # Numerator Product
        inter = (
            inputs_valid[:, :, :, :, 1:] * target_oneHot[:, :, :, :, 1:]
        )  # Exclude class 0 from intersection
        inter = inter.view(N, -1, self.classes - 1).sum(1)  # Sum over all valid classes

        # Denominator
        union = (
            inputs_valid[:, :, :, :, 1:]
            + target_oneHot[:, :, :, :, 1:]
            - (inputs_valid[:, :, :, :, 1:] * target_oneHot[:, :, :, :, 1:])
        )  # Exclude class 0 from union
        union = union.view(N, -1, self.classes - 1).sum(1)  # Sum over all valid classes

        epsilon = 1e-8
        iou = inter / (union + epsilon)  # Add epsilon to avoid division by zero

        # loss = 1 - iou.mean()
        ## Return average loss over classes and batch
        return iou, iou.mean()


class mIoULoss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def to_one_hot(self, tensor):
        n, h, w, d, l, _ = tensor.size()
        one_hot = (
            torch.zeros(n, h, w, d, l, self.classes)
            .to(tensor.device)
            .scatter_(-1, tensor.view(n, h, w, d, l, 1).long(), 1)
        )
        return one_hot

    def forward(self, inputs, target, mask):
        # inputs => N x H x W x D x L x Classes
        # target_oneHot => N x H x W x D x L x Classes
        # mask => N x H x W x D x L (boolean array with True for valid pixels and False for invalid/ignored pixels)

        N = inputs.size()[0]

        mask = mask.float()  # Convert mask to float
        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=-1)

        # Apply the mask to the target and inputs
        # mask = mask.unsqueeze(-1)  # Expand mask to match the channel dimension
        target_valid = target * mask
        inputs_valid = inputs * mask

        # Convert the target to one-hot format
        target_oneHot = self.to_one_hot(target_valid)

        # Numerator Product
        inter = (
            inputs_valid[:, :, :, :, :, 1:] * target_oneHot[:, :, :, :, :, 1:]
        )  # Exclude class 0 from intersection
        inter = inter.view(N, -1, self.classes - 1).sum(1)  # Sum over all valid classes

        # Denominator
        union = (
            inputs_valid[:, :, :, :, :, 1:]
            + target_oneHot[:, :, :, :, :, 1:]
            - (inputs_valid[:, :, :, :, :, 1:] * target_oneHot[:, :, :, :, :, 1:])
        )  # Exclude class 0 from union
        union = union.view(N, -1, self.classes - 1).sum(1)  # Sum over all valid classes

        epsilon = 1e-8
        iou = inter / (union + epsilon)  # Add epsilon to avoid division by zero

        loss = 1 - iou.mean()
        ## Return average loss over classes and batch
        return loss, iou.mean()


# def calculate_class_weights(class_frequencies, smooth=1e-5):
#     # Exclude class 0 (void) from the calculation
#     if len(class_frequencies) > 1:
#         class_frequencies = class_frequencies[1:]

#     total_samples = np.sum(class_frequencies)
#     class_weights = np.zeros(len(class_frequencies), dtype=np.float32)

#     for class_idx, frequency in enumerate(class_frequencies):
#         class_weights[class_idx] = total_samples / (frequency + smooth)

#     # Normalize the class weights
#     class_weights /= np.sum(class_weights)
#     return class_weights


# import numpy as np

import torch


# def calculate_class_weights(train_loader, num_classes=19, c=1.02):
#     # Calculate class frequencies in the training dataset, excluding class 0
#     class_count = np.zeros(num_classes)
#     total = 0

#     for i, batch in enumerate(train_loader):
#         _, out_sem, _ = batch
#         # Iterate through each tensor in the list
#         for tensor in out_sem:
#             # print("out_sem tensor", tensor.shape)
#             # Convert the tensor to a NumPy array and flatten
#             flat_label = tensor.view(-1).numpy()
#             # flat_label = flat_label[flat_label != 0]
#             bincount = np.bincount(flat_label, minlength=num_classes)

#             mask = bincount > 0
#             # Multiply the mask by the pixel count. The resulting array has
#             # one element for each class. The value is either 0 (if the class
#             # does not exist in the label) or equal to the pixel count (if
#             # the class exists in the label)
#             total += mask * flat_label.size
#             class_count += bincount
#             # total += flat_label.size

#     # for data in train_dataset:
#     #     labels = data["labels"]  # Extract the labels from your training dataset
#     #     flat_label = labels.view(-1).numpy()
#     #     class_count += np.bincount(flat_label, minlength=num_classes)
#     #     total += flat_label.size

#     # Set the class frequency for class 0 (void) to zero
#     # class_count[0] = 0

#     freq = class_count / total
#     med = np.median(freq)

#     class_weights = med / freq

#     print("class_weights", class_weights)

#     return class_weights

#     # Compute propensity score and then the weights for each class with logarithm
#     # class_count[0] = 0

#     # # print("class_count", class_count)
#     # # print("total", total)

#     # propensity_score = class_count / total

#     # # print("propensity_score", propensity_score)
#     # class_weights = 1 / (np.log(c + propensity_score))

#     # class_weights[0] = 0

#     # return class_weights


def calculate_class_weights(train_loader, num_classes=19, c=1.02):
    # Calculate class frequencies in the training dataset, excluding class 0
    class_count = np.zeros(num_classes)
    total = 0

    for i, batch in enumerate(train_loader):
        _, out_sem, _ = batch
        # Iterate through each tensor in the list
        for tensor in out_sem:
            # print("out_sem tensor", tensor.shape)
            # Convert the tensor to a NumPy array and flatten
            flat_label = tensor.view(-1).numpy()

            unique_classes, class_counts = np.unique(flat_label, return_counts=True)

            print("unique_classes", unique_classes, "\n")
            print("class_counts", class_counts, "\n")
            flat_label = flat_label[flat_label != 0]
            class_count += np.bincount(flat_label, minlength=num_classes)
            total += flat_label.size

    # for data in train_dataset:
    #     labels = data["labels"]  # Extract the labels from your training dataset
    #     flat_label = labels.view(-1).numpy()
    #     class_count += np.bincount(flat_label, minlength=num_classes)
    #     total += flat_label.size

    # Set the class frequency for class 0 (void) to zero
    # class_count[0] = 0

    # Compute propensity score and then the weights for each class with logarithm
    class_count[0] = 0

    # print("class_count", class_count)
    # print("total", total)

    propensity_score = class_count / total

    # print("propensity_score", propensity_score)
    class_weights = 1 / (np.log(c + propensity_score))

    class_weights[0] = 0

    return class_weights


def plot_class_weights(class_weights, save_path="class_weights_plot.png"):
    num_classes = len(class_weights)
    class_indices = list(range(num_classes))

    plt.figure(figsize=(10, 6))
    plt.bar(class_indices, class_weights)
    plt.xlabel("Class")
    plt.ylabel("Class Weight")
    plt.title("Class Weights")
    plt.xticks(class_indices)
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(save_path)
    plt.close()  # Close the figure to free up memory


def save_class_distribution_plot(
    train_loader, num_classes=19, save_path="class_distribution_train.png"
):
    class_frequencies = np.zeros(num_classes)

    for i, batch in enumerate(train_loader):
        _, out_sem, _ = batch
        # Iterate through each tensor in the list
        for tensor in out_sem:
            print("out_sem tensor", tensor.shape)
            # Convert the tensor to a NumPy array and flatten
            out_sem_flat = tensor.numpy().flatten()

            # Count class occurrences in this tensor
            unique_classes, class_counts = np.unique(out_sem_flat, return_counts=True)

            # Ignore the zero class (adjust the index if your zero class has a different value)
            non_zero_classes = unique_classes[unique_classes != 0]
            non_zero_counts = class_counts[unique_classes != 0]

            print("non_zero_classes", non_zero_classes, "\n")

            # Update the overall class_frequencies only for non-zero classes
            class_frequencies[non_zero_classes] += non_zero_counts

    # Print the frequencies of individual classes
    for class_idx, frequency in enumerate(class_frequencies):
        print(f"Class {class_idx}: {frequency}")

    # Plot the class distribution
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(class_frequencies)), class_frequencies)
    plt.xlabel("Class")
    plt.ylabel("Frequency")
    plt.title("Class Distribution in Training Set")
    plt.xticks(range(len(class_frequencies)))
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(save_path)
    plt.close()  # Close the figure to free up memory

    return class_frequencies


def intersectionAndUnionGPU(output, target, K):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    # assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)

    # Ignore class 0 in both output and target
    ignore_mask = target != 0
    output = output[ignore_mask]
    target = target[ignore_mask]

    # Calculate the intersection and union areas for each class
    intersection = output[output == target]
    area_intersection = torch.histc(
        intersection, bins=K - 1, min=1, max=K - 1
    )  # Intersection (ignore class 0)
    area_output = torch.histc(
        output, bins=K - 1, min=1, max=K - 1
    )  # Area of predicted regions (ignore class 0)
    area_target = torch.histc(
        target, bins=K - 1, min=1, max=K - 1
    )  # Area of target regions (ignore class 0)

    # Calculate the union area by adding the areas of output and target and subtracting the intersection
    area_union = area_output + area_target - area_intersection

    return area_intersection, area_union, area_target


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def masked_cross_entropy(criterion, targets, logits_pred, mask, num_classes=19):
    # Apply the mask to the targets and logits
    targets_masked = targets * mask
    targets_masked = targets_masked.squeeze(-1)
    logits_pred_masked = logits_pred * mask

    logits_pred_masked = logits_pred_masked.reshape(-1, num_classes)
    targets_masked = targets_masked.reshape(-1)

    # Compute cross-entropy loss with the masked targets and logits
    loss = criterion(logits_pred_masked, targets_masked.long())

    return loss
