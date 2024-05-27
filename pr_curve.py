import numpy as np
import matplotlib.pyplot as plt
import torch
from torchmetrics.classification import PrecisionRecallCurve, MulticlassAveragePrecision

loader_name = 'conv_resnet50'
pred_scores = np.load(f'result_numpy/{loader_name}/class_score.npy')
true_labels = np.load(f'result_numpy/{loader_name}/class_label.npy')
pred_scores_domain = np.load(f'result_numpy/{loader_name}/domain_score.npy')
true_labels_domain = np.load(f'result_numpy/{loader_name}/domain_label.npy')

pred_scores = torch.from_numpy(pred_scores)
true_labels = torch.from_numpy(true_labels)
pred_scores_domain = torch.from_numpy(pred_scores_domain)
true_labels_domain = torch.from_numpy(true_labels_domain)

pr_curve = PrecisionRecallCurve(task="multiclass", num_classes=65)
precision, recall, _ = pr_curve(pred_scores, true_labels)


pr_curve = PrecisionRecallCurve(task="multiclass", num_classes=4)
precision_domain, recall_domain, _ = pr_curve(pred_scores_domain, true_labels_domain)
# mAP_domain = get_map_domain(pred_scores_domain, true_labels_domain)

with torch.no_grad():
    precision = [t.tolist() for t in precision]
    recall = [t.tolist() for t in recall]

test_acc_class = (true_labels == pred_scores.argmax(1)).float().mean().item()
test_acc_domain = (true_labels_domain == pred_scores_domain.argmax(1)).float().mean().item()

get_map_class = MulticlassAveragePrecision(num_classes=65, average="macro", thresholds=None)
get_map_domain = MulticlassAveragePrecision(num_classes=4, average="macro", thresholds=None)
mAP = get_map_class(pred_scores, true_labels)
mAP_domain = get_map_domain(pred_scores_domain, true_labels_domain)

print(mAP, mAP_domain)
print(test_acc_class, test_acc_domain)

plt.figure()
for idx, _ in enumerate(recall):
    plt.plot(recall[idx], precision[idx])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR Curve')
plt.show()
