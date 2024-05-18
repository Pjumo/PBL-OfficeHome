# for OfficeHome
import data_loader
import numpy as np
import os

import torch
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data.sampler import WeightedRandomSampler
from torchmetrics.classification import AveragePrecision, PrecisionRecallCurve, MulticlassAveragePrecision


def fun():
    torch.multiprocessing.freeze_support()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    num_epochs = 20
    batch_size = 32

    train_loader, test_loader = data_loader.dataloader('smote', batch_size)

    cnt_progress = len(train_loader) // 30

    model_name = 'smote_resnet18'

    class ResNetFeatureExtractor(nn.Module):
        def __init__(self):
            super(ResNetFeatureExtractor, self).__init__()
            original_model = models.resnet18(pretrained=True)
            num_ftrs = original_model.fc.in_features
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.fc = nn.Linear(num_ftrs, 65)
            self.fc_domain = nn.Linear(num_ftrs, 4)

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            output = self.fc(x)
            output_domain = self.fc_domain(x)
            return output, output_domain

    model = ResNetFeatureExtractor().to(device)

    # 성능 평가
    def evaluate_model(model, test_loader):
        model.eval()
        true_labels = []
        pred_scores = [[]]
        true_labels_domain = []
        pred_scores_domain = [[]]
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                class_labels = labels // 4
                domain_labels = labels % 4

                outputs_class, outputs_domain = model(images)
                probabilities = outputs_class
                pred_scores.extend(probabilities.tolist())
                true_labels.extend(class_labels.tolist())

                probabilities_domain = outputs_domain
                pred_scores_domain.extend(probabilities_domain.tolist())
                true_labels_domain.extend(domain_labels.tolist())

        return true_labels, pred_scores[1:], true_labels_domain, pred_scores_domain[1:]

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0002, momentum=0.9)
    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()

    train_loss = []
    train_class_acc = []
    train_domain_acc = []

    val_loss = []
    val_class_acc = []
    val_domain_acc = []
    val_mAP_class = []
    val_mAP_domain = []

    # 훈련 루프 내 도메인 적응 코드 활성화
    for epoch in range(num_epochs):
        total_class_acc = 0
        total_domain_acc = 0
        total_train_loss = 0
        for cnt, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            class_labels = labels // 4
            domain_labels = labels % 4

            optimizer.zero_grad()
            outputs_class, outputs_domain = model(images)
            class_loss = criterion_class(outputs_class, class_labels)
            domain_loss = criterion_domain(outputs_domain, domain_labels)
            loss = domain_loss + class_loss
            loss.backward()
            optimizer.step()

            total_class_acc += (class_labels == outputs_class.argmax(1)).float().mean().item()
            total_domain_acc += (domain_labels == outputs_domain.argmax(1)).float().mean().item()

            total_train_loss += class_loss.item()
            if cnt % cnt_progress == 0:
                print(f'\rEpoch {epoch + 1} [', end='')
                for prog in range(cnt // cnt_progress):
                    print('■', end='')
                for prog in range((len(train_loader) - cnt) // cnt_progress):
                    print('□', end='')
                print(']', end='')

        total_train_loss /= len(train_loader)
        total_class_acc /= len(train_loader)
        total_domain_acc /= len(train_loader)
        train_loss.append(total_train_loss)
        train_class_acc.append(total_class_acc)
        train_domain_acc.append(total_domain_acc)
        print(f' - Train Loss: {total_train_loss:.4f}')
        print(f'Train Category Acc: {total_class_acc:.4f}, Train Domain Acc: {total_domain_acc:.4f}')

        # Validation
        true_labels, pred_scores, true_labels_domain, pred_scores_domain = evaluate_model(model, test_loader)

        true_labels = torch.tensor(true_labels)
        pred_scores = torch.tensor(pred_scores)
        true_labels_domain = torch.tensor(true_labels_domain)
        pred_scores_domain = torch.tensor(pred_scores_domain)

        total_val_acc_class = (true_labels == pred_scores.argmax(1)).float().mean().item()
        total_val_acc_domain = (true_labels_domain == pred_scores_domain.argmax(1)).float().mean().item()
        class_loss = criterion_class(pred_scores, true_labels)
        domain_loss = criterion_domain(pred_scores_domain, true_labels_domain)
        total_val_loss = domain_loss + class_loss

        val_loss.append(total_val_loss)
        val_class_acc.append(total_val_acc_class)
        val_domain_acc.append(total_val_acc_domain)

        # mAP
        get_map_class = MulticlassAveragePrecision(num_classes=65, average="macro", thresholds=None)
        get_map_domain = MulticlassAveragePrecision(num_classes=4, average="macro", thresholds=None)
        mAP = get_map_class(pred_scores, true_labels)
        mAP_domain = get_map_domain(pred_scores_domain, true_labels_domain)

        val_mAP_class.append(mAP)
        val_mAP_domain.append(mAP_domain)

        print(f'Val Category Acc: {total_val_acc_class:.4f}, Val Domain Acc: {total_val_acc_domain:.4f}')

    # Evaluate
    true_labels, pred_scores, true_labels_domain, pred_scores_domain = evaluate_model(model, test_loader)

    class_label = np.array(true_labels)
    class_score = np.array(pred_scores)
    domain_label = np.array(true_labels_domain)
    domain_score = np.array(pred_scores_domain)

    train_loss = np.array(train_loss)
    train_class_acc = np.array(train_class_acc)
    train_domain_acc = np.array(train_domain_acc)
    val_loss = np.array(val_loss)
    val_class_acc = np.array(val_class_acc)
    val_domain_acc = np.array(val_domain_acc)
    val_mAP_class = np.array(val_mAP_class)
    val_mAP_domain = np.array(val_mAP_domain)

    os.makedirs(f'result_numpy/{model_name}', exist_ok=True)
    np.save(os.path.join(f'result_numpy/{model_name}/train_loss'), train_loss)
    np.save(os.path.join(f'result_numpy/{model_name}/train_class_acc'), train_class_acc)
    np.save(os.path.join(f'result_numpy/{model_name}/train_domain_acc'), train_domain_acc)
    np.save(os.path.join(f'result_numpy/{model_name}/val_loss'), val_loss)
    np.save(os.path.join(f'result_numpy/{model_name}/val_class_acc'), val_class_acc)
    np.save(os.path.join(f'result_numpy/{model_name}/val_domain_acc'), val_domain_acc)
    np.save(os.path.join(f'result_numpy/{model_name}/val_class_mAP'), val_mAP_class)
    np.save(os.path.join(f'result_numpy/{model_name}/val_domain_mAP'), val_mAP_domain)
    np.save(os.path.join(f'result_numpy/{model_name}/class_label'), class_label)
    np.save(os.path.join(f'result_numpy/{model_name}/class_score'), class_score)
    np.save(os.path.join(f'result_numpy/{model_name}/domain_label'), domain_label)
    np.save(os.path.join(f'result_numpy/{model_name}/domain_score'), domain_score)

    model.to("cpu")
    torch.cuda.empty_cache()


if __name__ == '__main__':
    fun()
