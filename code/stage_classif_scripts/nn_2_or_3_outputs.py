import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 6)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 6)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 36 * 36, 128)
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 36 * 36)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
    
class Net_parametrizable(nn.Module):
    def __init__(self, num_classes, nb_kernels_conv1=6, nb_kernels_conv2=16, kernel_size_conv1=6, kernel_size_conv2=6, pool_size=2, fully_connected_size=128, image_size=171, dropout_value=0.1):

        super(Net_parametrizable, self).__init__()
        self.conv1 = nn.Conv2d(1, nb_kernels_conv1, kernel_size_conv1)
        self.bn1 = nn.BatchNorm2d(nb_kernels_conv1)
        self.conv2 = nn.Conv2d(nb_kernels_conv1, nb_kernels_conv2, kernel_size_conv2)
        self.bn2 = nn.BatchNorm2d(nb_kernels_conv2)
        self.pool = nn.MaxPool2d(pool_size, pool_size)

        def conv_output_size(size, kernel_size, stride=1, padding=0):
            return (size - kernel_size + 2 * padding) // stride + 1
        
        def pool_output_size(size, pool_size, stride=None):
            if stride is None:
                stride = pool_size
            return size // stride

        conv1_out = conv_output_size(image_size, kernel_size_conv1)
        pool1_out = pool_output_size(conv1_out, pool_size)
        conv2_out = conv_output_size(pool1_out, kernel_size_conv2)
        pool2_out = pool_output_size(conv2_out, pool_size)

        flattened_size = nb_kernels_conv2 * pool2_out * pool2_out

        self.fc1 = nn.Linear(flattened_size, fully_connected_size)
        self.dropout1 = nn.Dropout(dropout_value)
        self.fc2 = nn.Linear(fully_connected_size, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

def create_model(num_classes):
    return Net(num_classes)

def create_parametrizable_model(num_classes, nb_kernels_conv1, nb_kernels_conv2, kernel_size_conv1, kernel_size_conv2, pool_size, fully_connected_size, image_size, dropout_value):
    return Net_parametrizable(num_classes, nb_kernels_conv1, nb_kernels_conv2, kernel_size_conv1, kernel_size_conv2, pool_size, fully_connected_size, image_size, dropout_value)

def remap_labels(target):
    target_cpu = target.cpu()
    unique_labels = torch.unique(target_cpu)
    # print(f"Unique labels before remapping: {unique_labels}")  # Debugging
    
    label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}
    remapped_target = torch.tensor([label_map[x.item()] for x in target_cpu], dtype=torch.long)
    
    # print(f"Unique labels after remapping: {torch.unique(remapped_target)}")  # Debugging
    return remapped_target.to(target.device), label_map

def train_and_validate(model, model_name, train_loader, val_loader, criterion, optimizer, device, n_epochs=100):
    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    file = open(f"{model_name}_stats.csv", mode='w', newline='')
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy"])

    
    for epoch in range(1, n_epochs + 1):
        # Phase d'entraînement
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            target, label_map = remap_labels(target)  # Remap once
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == target).sum().item()
            total_train += target.size(0)
            
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)
        print(f"Epoch {epoch}/{n_epochs}, Training Loss: {train_losses[-1]:.4f}, Training Accuracy: {train_accuracies[-1]:.4f}")
        
        # Phase de validation
        model.eval()
        valid_loss, correct_valid, total_valid = 0.0, 0, 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                target, _ = remap_labels(target)  # Added remapping here
                
                outputs = model(data)
                loss = criterion(outputs, target)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_valid += (predicted == target).sum().item()
                total_valid += target.size(0)
                
        valid_accuracy = correct_valid / total_valid
        valid_losses.append(valid_loss / len(val_loader))
        valid_accuracies.append(valid_accuracy)

        writer.writerow([epoch, train_loss, train_accuracy, valid_loss, valid_accuracy])
        print(f"Epoch {epoch}/{n_epochs}, Validation Loss: {valid_losses[-1]:.4f}, Validation Accuracy: {valid_accuracies[-1]:.4f}")
    
    file.close()
    return train_losses, valid_losses, train_accuracies, valid_accuracies

def convert_label_to_text(label, stage, mappings, label_mapping):
    """Convertit un label numérique en texte selon le stage"""
    # Inverser les mappings pour avoir {idx: original_label}
    inverse_mappings = {
        'stage1': {v: k for k, v in mappings['stage1'].items()},
        'stage2_EL': {v: k for k, v in mappings['stage2_EL'].items()},
        'stage2_SBI': {v: k for k, v in mappings['stage2_SBI'].items()},
        'stage3': {v: k for k, v in mappings['stage3'].items()}
    }
    
    # # Table de conversion des labels numériques en texte
    # label_text = {
    #     2: 'E', 7: 'L',  # Stage 2 EL
    #     0: 'BS', 10: 'S/I',  # Stage 2 SBI
    #     5: 'S', 8: 'I'  # Stage 3
    # }

    # print("inverse_mappings:", inverse_mappings)
    # print("label_mapping:", label_mapping)
    # print("stage:", stage)
    # print("label:", label)

    # original_label = inverse_mappings[stage][label]
    
    return label_mapping.get(label, str(label))

def cascade_classification_with_labels(test_loader_1, test_loader_2, test_loader_3, test_loader_4, 
                                     model_1, model_2, model_3, model_4, device, mappings, label_mapping):
    predictions = []
    actual_labels = []
    predicted_labels_text = []
    actual_labels_text = []
    
    with torch.no_grad():
        for batch_idx, (data_1, labels_1) in enumerate(test_loader_1):
            data_1 = data_1.to(device)
            
            # Obtenir les batches correspondants des autres dataloaders
            try:
                data_2, labels_2 = next(iter(test_loader_2))
                data_3, labels_3 = next(iter(test_loader_3))
                data_4, labels_4 = next(iter(test_loader_4))
            except StopIteration:
                break
            
            # Premier niveau : E/L vs S
            output_1 = model_1(data_1)
            _, pred_1 = torch.max(output_1, 1)
            
            for i in range(len(pred_1)):
                current_data = data_1[i].unsqueeze(0)
                print(f"\nImage {batch_idx * len(pred_1) + i}:")
                print(f"Stage 1 prediction: {'Elliptical/Lenticular' if pred_1[i] == 0 else 'Spirals/Irregular'}")
                
                if pred_1[i] == 0:  # Classifié comme E/L
                    output_2 = model_2(current_data)
                    _, pred_2 = torch.max(output_2, 1)
                    pred_text = 'Elliptical' if pred_2.item() == 0 else 'Lenticular'
                    print(f"Stage 2 (E vs L) prediction: {pred_text}")
                    predictions.append(pred_text)
                    true_label = labels_2[i].item()
                    actual_labels.append(label_mapping.get(true_label, str(true_label)))
                    print(f"True label: {label_mapping.get(true_label, str(true_label))}")
                
                else:  # Classifié comme S
                    output_3 = model_3(current_data)
                    _, pred_3 = torch.max(output_3, 1)
                    pred_text = 'Barred Spirals' if pred_3.item() == 0 else 'Spirals/Irregular'
                    print(f"Stage 2 (BS vs S/I) prediction: {pred_text}")
                    
                    if pred_3.item() == 0:  # BS
                        predictions.append(pred_text)
                        true_label = labels_3[i].item()
                        actual_labels.append(label_mapping.get(true_label, str(true_label)))
                        print(f"True label: {label_mapping.get(true_label, str(true_label))}")
                    
                    else:  # S/I
                        output_4 = model_4(current_data)
                        _, pred_4 = torch.max(output_4, 1)
                        pred_text = 'Spirals' if pred_4.item() == 0 else 'Irregular'
                        print(f"Stage 3 (S vs I) prediction: {pred_text}")
                        predictions.append(pred_text)
                        true_label = labels_4[i].item()
                        actual_labels.append(label_mapping.get(true_label, str(true_label)))
                        print(f"True label: {label_mapping.get(true_label, str(true_label))}")

    return predictions, actual_labels

def test(test_loader_1, test_loader_2, test_loader_3, test_loader_4, 
                                     model_1, model_2, model_3, model_4, device, mappings, label_mapping):
    print(test_loader_1[0:10])
