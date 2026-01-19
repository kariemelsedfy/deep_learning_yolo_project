import torch


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Computes IoU between predicted boxes and label boxes.

    Args:
        boxes_preds: Tensor of boxes (..., 4)
        boxes_labels: Tensor of boxes (..., 4)
        box_format: "midpoint" for (x, y, w, h) or "corners" for (x1, y1, x2, y2)

    Returns:
        Tensor of IoU values with shape broadcasted from inputs.
    """
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    else:
        raise ValueError('box_format must be "midpoint" or "corners"')

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = (box1_x2 - box1_x1).abs() * (box1_y2 - box1_y1).abs()
    box2_area = (box2_x2 - box2_x1).abs() * (box2_y2 - box2_y1).abs()

    return intersection / (box1_area + box2_area - intersection + 1e-6)
