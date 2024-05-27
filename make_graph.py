from torchmetrics.classification import AveragePrecision, MulticlassAveragePrecision
import numpy as np
import matplotlib.pyplot as plt

loader_name = 'conv_resnet50'
train_class_acc = np.load(f'result_numpy/{loader_name}/train_class_acc.npy')
train_domain_acc = np.load(f'result_numpy/{loader_name}/train_domain_acc.npy')
train_loss = np.load(f'result_numpy/{loader_name}/train_loss.npy')
val_class_acc = np.load(f'result_numpy/{loader_name}/val_class_acc.npy')
val_domain_acc = np.load(f'result_numpy/{loader_name}/val_domain_acc.npy')
val_loss = np.load(f'result_numpy/{loader_name}/val_loss.npy')
val_class_map = np.load(f'result_numpy/{loader_name}/val_class_mAP.npy')
val_domain_map = np.load(f'result_numpy/{loader_name}/val_domain_mAP.npy')

plt.figure()
plt.plot(list(range(1, 11)), train_domain_acc)
plt.plot(list(range(1, 11)), val_domain_acc)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Domain Accuracy')
plt.legend(('train_acc', 'val_acc'))
plt.show()
