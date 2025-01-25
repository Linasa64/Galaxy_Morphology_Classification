import os
import random
from astropy.io import fits
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import torch
from nn_2_or_3_outputs import remap_labels

def load_reduced_dataset(seed, path="../processed_dataset/"):
    datasets = {"train": {}, "val": {}, "test": {}}
    filenames = os.listdir(path)
    random.seed(seed)
    random.shuffle(filenames)

    train_size = int(0.8 * len(filenames))
    val_size = int(0.1 * len(filenames))
    test_size = len(filenames) - train_size - val_size

    train_files = filenames[:train_size]
    val_files = filenames[train_size:train_size + val_size]
    test_files = filenames[train_size + val_size:]

    for i, filename in tqdm(enumerate(filenames)):
        if filename.endswith(".fits"):
            try:
                key = filename.split("_")[0]
                with fits.open(os.path.join(path, filename)) as hdul:
                    # galaxy_type_1 = hdul[0].header["CAT1"]
                    # galaxy_type_2 = hdul[0].header["CAT2"]
                    # galaxy_type_3 = hdul[0].header["CAT3"]
                    galaxy_type_1 = hdul[0].header["CAT1bis"]
                    galaxy_type_2 = hdul[0].header["CAT2bis"]
                    galaxy_type_3 = hdul[0].header["CAT3bis"]
                    image_data = hdul[0].data  # Assuming the image is in the primary HDU

                    if i < train_size:
                        datasets["train"][key] = {"image": image_data, "galaxy_type_1": galaxy_type_1, "galaxy_type_2": galaxy_type_2, "galaxy_type_3": galaxy_type_3}
                    elif i < train_size + val_size:
                        datasets["val"][key] = {"image": image_data, "galaxy_type_1": galaxy_type_1, "galaxy_type_2": galaxy_type_2, "galaxy_type_3": galaxy_type_3}
                    else:
                        datasets["test"][key] = {"image": image_data, "galaxy_type_1": galaxy_type_1, "galaxy_type_2": galaxy_type_2, "galaxy_type_3": galaxy_type_3}
            except Exception as e:
              print(f"Error processing {filename}: {e}")
              continue

    return datasets

def preprocess_data(dataset):
    images = []
    labels_1 = []
    labels_2 = []
    labels_3 = []
    for key in dataset:
        image_data = dataset[key]['image']
        galaxy_type_1 = dataset[key]['galaxy_type_1']
        galaxy_type_2 = dataset[key]['galaxy_type_2']
        galaxy_type_3 = dataset[key]['galaxy_type_3']
        # Assuming image_data has 3 dimensions (i, r, g)
        if image_data.shape[0] == 3:  # Check for 3 dimensions
            images.append(image_data)
            labels_1.append(galaxy_type_1)
            labels_2.append(galaxy_type_2)
            labels_3.append(galaxy_type_3)
        else:
            print(f"Skipping image {key} due to incorrect dimensions: {image_data.shape}")

    images = np.array(images)

    # Calculate mean and standard deviation
    mean = np.mean(images, axis=(0, 1, 2), keepdims=True)
    std = np.std(images, axis=(0, 1, 2), keepdims=True)

    # Normalize
    images = (images - mean) / std
        
    return images, labels_1, labels_2, labels_3

def prepare_dataset(dataset, band=0):
    """
    Prépare les données et labels des ensembles d'entraînement, validation et test.

    Args:
        dataset (dict): Un dictionnaire contenant les ensembles 'train', 'val', 'test'.
                        Chaque ensemble doit inclure les images et les labels associés.
        band (int): Le canal à sélectionner dans les images (par défaut 0).

    Returns:
        dict: Un dictionnaire contenant les tenseurs des images et labels encodés
              pour les ensembles 'train', 'val' et 'test', ainsi que le mapping des labels encodés.
    """
    # Prétraitement des ensembles
    train_images, train_labels_1, train_labels_2, train_labels_3 = preprocess_data(dataset['train'])
    val_images, val_labels_1, val_labels_2, val_labels_3 = preprocess_data(dataset['val'])
    test_images, test_labels_1, test_labels_2, test_labels_3 = preprocess_data(dataset['test'])

    # Sélection d'un canal spécifique
    train_images = train_images[:, band, :, :]
    val_images = val_images[:, band, :, :]
    test_images = test_images[:, band, :, :]

    # Conversion des labels en numpy arrays
    train_labels_1 = np.array(train_labels_1)
    train_labels_2 = np.array(train_labels_2)
    train_labels_3 = np.array(train_labels_3)

    val_labels_1 = np.array(val_labels_1)
    val_labels_2 = np.array(val_labels_2)
    val_labels_3 = np.array(val_labels_3)

    test_labels_1 = np.array(test_labels_1)
    test_labels_2 = np.array(test_labels_2)
    test_labels_3 = np.array(test_labels_3)

    # Encodage des labels
    labels_encoder = LabelEncoder()
    all_labels = np.concatenate([train_labels_1, val_labels_1, test_labels_1,
                                 train_labels_2, val_labels_2, test_labels_2,
                                 train_labels_3, val_labels_3, test_labels_3])
    labels_encoder.fit(all_labels)

    # Encodage et conversion en tenseurs
    def encode_and_convert(labels):
        encoded = labels_encoder.transform(labels)
        return torch.tensor(encoded, dtype=torch.long)

    # Train
    train_data = {
        'images': torch.tensor(train_images, dtype=torch.float32),
        'labels_1': encode_and_convert(train_labels_1),
        'labels_2': encode_and_convert(train_labels_2),
        'labels_3': encode_and_convert(train_labels_3)
    }

    # Validation
    val_data = {
        'images': torch.tensor(val_images, dtype=torch.float32),
        'labels_1': encode_and_convert(val_labels_1),
        'labels_2': encode_and_convert(val_labels_2),
        'labels_3': encode_and_convert(val_labels_3)
    }

    # Test
    test_data = {
        'images': torch.tensor(test_images, dtype=torch.float32),
        'labels_1': encode_and_convert(test_labels_1),
        'labels_2': encode_and_convert(test_labels_2),
        'labels_3': encode_and_convert(test_labels_3)
    }

    # Mapping des labels encodés
    label_mapping = {index: label for index, label in enumerate(labels_encoder.classes_)}

    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'label_mapping': label_mapping
    }

data_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),        # Flip horizontal avec probabilité 50%
    transforms.RandomVerticalFlip(p=0.5),          # Flip vertical avec probabilité 50%
    transforms.RandomRotation(degrees=90),         # Rotation aléatoire jusqu'à 90°
    transforms.ToTensor()                          # Conversion en tenseur
])

default_transform = transforms.Compose([
    transforms.ToTensor()  # Conversion en tenseur sans augmentation
])

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Si l'image est un tenseur, convertir en numpy d'abord
        if torch.is_tensor(self.images[idx]):
            image = self.images[idx].numpy()
        else:
            image = self.images[idx]

        # Assurons-nous que l'image est au bon format pour PIL
        image = (image * 255).astype(np.uint8)  # Convertir l'image pour qu'elle soit en uint8

        # Si l'image a un seul canal (2D), elle est déjà dans le bon format
        if len(image.shape) == 2:  # Image en niveaux de gris (H, W)
            pass  # L'image est déjà en 2D (H, W), nous la gardons comme elle est
        elif image.shape[0] == 1:  # Si l'image est en forme (1, H, W)
            image = np.squeeze(image, axis=0)  # Retirer la dimension inutile

        # Si l'image a plus de 2 dimensions, nous devons la transformer (ex. (1, H, W) -> (H, W))
        image = Image.fromarray(image)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def create_dataloaders(data, label_mapping, batch_size=32):
    """
    Crée des DataLoaders pour l'entraînement, la validation et le test, avec des filtres spécifiques.
    Args:
        data (dict): Contient les images et labels pour train, val, test.
        label_mapping (dict): Mapping des indices de labels vers leurs descriptions textuelles.
        batch_size (int): Taille des batchs pour les DataLoaders.

    Returns:
        dict: Contient les DataLoaders pour chaque configuration de modèle.
    """

    train_images = data['train']['images']
    val_images = data['val']['images']
    test_images = data['test']['images']

    train_labels_1 = data['train']['labels_1']
    val_labels_1 = data['val']['labels_1']
    test_labels_1 = data['test']['labels_1']

    train_labels_2 = data['train']['labels_2']
    val_labels_2 = data['val']['labels_2']
    test_labels_2 = data['test']['labels_2']

    train_labels_3 = data['train']['labels_3']
    val_labels_3 = data['val']['labels_3']
    test_labels_3 = data['test']['labels_3']

    # Filtrage des données pour chaque modèle
    def filter_data(images, labels, condition):
        # Convertir les tenseurs en entiers pour l'indexation
        indices = [i for i, label in enumerate(labels) if condition(label_mapping[label.item()])]
        filtered_images = [images[i] for i in indices]
        filtered_labels = [labels[i] for i in indices]
        print(f"Nombre d'images après filtrage: {len(filtered_images)}")
        print(f"Labels uniques trouvés: {set(label_mapping[label.item()] for label in filtered_labels)}")
        return filtered_images, filtered_labels

    # Dataloader 1 : Labels 1
    train_dataset_1 = CustomDataset(train_images, train_labels_1, transform=data_augmentation)
    val_dataset_1 = CustomDataset(val_images, val_labels_1, transform=default_transform)
    test_dataset_1 = CustomDataset(test_images, test_labels_1, transform=default_transform)

    # # Dataloader 2 : Labels 2 (elliptiques ou lenticulaires)
    # train_images_2_el, train_labels_2_el = filter_data(
    #     train_images, train_labels_2, lambda label: label in ['Elliptical', 'Lenticular']
    # )
    # val_images_2_el, val_labels_2_el = filter_data(
    #     val_images, val_labels_2, lambda label: label in ['Elliptical', 'Lenticular']
    # )
    # test_images_2_el, test_labels_2_el = filter_data(
    #     test_images, test_labels_2, lambda label: label in ['Elliptical', 'Lenticular']
    # )
    # train_dataset_2_el = CustomDataset(train_images_2_el, train_labels_2_el, transform=data_augmentation)
    # val_dataset_2_el = CustomDataset(val_images_2_el, val_labels_2_el, transform=default_transform)
    # test_dataset_2_el = CustomDataset(test_images_2_el, test_labels_2_el, transform=default_transform)

    # # Dataloader 3 : Labels 2 (spirals, barred spirals, irregulars)
    # train_images_2_sbi, train_labels_2_sbi = filter_data(
    #     train_images, train_labels_2, lambda label: label in ['Barred Spirals', 'Spirals/Irregulars']
    # )
    # val_images_2_sbi, val_labels_2_sbi = filter_data(
    #     val_images, val_labels_2, lambda label: label in ['Barred Spirals', 'Spirals/Irregulars']
    # )
    # test_images_2_sbi, test_labels_2_sbi = filter_data(
    #     test_images, test_labels_2, lambda label: label in ['Barred Spirals', 'Spirals/Irregulars']
    # )
    # train_dataset_2_sbi = CustomDataset(train_images_2_sbi, train_labels_2_sbi, transform=data_augmentation)
    # val_dataset_2_sbi = CustomDataset(val_images_2_sbi, val_labels_2_sbi, transform=default_transform)
    # test_dataset_2_sbi = CustomDataset(test_images_2_sbi, test_labels_2_sbi, transform=default_transform)

    # # Dataloader 4 : Labels 3 (spirals, barred spirals)
    # train_images_3_sb, train_labels_3_sb = filter_data(
    #     train_images, train_labels_3, lambda label: label in ['Spirals', 'Irregulars']
    # )
    # val_images_3_sb, val_labels_3_sb = filter_data(
    #     val_images, val_labels_3, lambda label: label in ['Spirals', 'Irregulars']
    # )
    # test_images_3_sb, test_labels_3_sb = filter_data(
    #     test_images, test_labels_3, lambda label: label in ['Spirals', 'Irregulars']
    # )
    # train_dataset_3_sb = CustomDataset(train_images_3_sb, train_labels_3_sb, transform=data_augmentation)
    # val_dataset_3_sb = CustomDataset(val_images_3_sb, val_labels_3_sb, transform=default_transform)
    # test_dataset_3_sb = CustomDataset(test_images_3_sb, test_labels_3_sb, transform=default_transform)


    # Dataloader 2 : Labels 2 (elliptiques ou lenticulaires)
    train_images_2_el, train_labels_2_el = filter_data(
        train_images, train_labels_2, lambda label: label in ['Elliptical', 'Lenticular']
    )
    val_images_2_el, val_labels_2_el = filter_data(
        val_images, val_labels_2, lambda label: label in ['Elliptical', 'Lenticular']
    )
    test_images_2_el, test_labels_2_el = filter_data(
        test_images, test_labels_2, lambda label: label in ['Elliptical', 'Lenticular']
    )
    train_dataset_2_el = CustomDataset(train_images_2_el, train_labels_2_el, transform=data_augmentation)
    val_dataset_2_el = CustomDataset(val_images_2_el, val_labels_2_el, transform=default_transform)
    test_dataset_2_el = CustomDataset(test_images_2_el, test_labels_2_el, transform=default_transform)

    # Dataloader 3 : Labels 2 (spirals, barred spirals, irregulars)
    train_images_2_sbi, train_labels_2_sbi = filter_data(
        train_images, train_labels_2, lambda label: label in ['Irregulars', 'Spirals/Barred Spirals']
    )
    val_images_2_sbi, val_labels_2_sbi = filter_data(
        val_images, val_labels_2, lambda label: label in ['Irregulars', 'Spirals/Barred Spirals']
    )
    test_images_2_sbi, test_labels_2_sbi = filter_data(
        test_images, test_labels_2, lambda label: label in ['Irregulars', 'Spirals/Barred Spirals']
    )
    train_dataset_2_sbi = CustomDataset(train_images_2_sbi, train_labels_2_sbi, transform=data_augmentation)
    val_dataset_2_sbi = CustomDataset(val_images_2_sbi, val_labels_2_sbi, transform=default_transform)
    test_dataset_2_sbi = CustomDataset(test_images_2_sbi, test_labels_2_sbi, transform=default_transform)

    # Dataloader 4 : Labels 3 (spirals, barred spirals)

    ### J'ai rajouté irregular ici, à enlever pour redevenir comme avant !!
    train_images_3_sb, train_labels_3_sb = filter_data(
        train_images, train_labels_3, lambda label: label in ['Spirals', 'Barred Spirals', 'Irregulars']
    )
    val_images_3_sb, val_labels_3_sb = filter_data(
        val_images, val_labels_3, lambda label: label in ['Spirals', 'Barred Spirals', 'Irregulars']
    )
    test_images_3_sb, test_labels_3_sb = filter_data(
        test_images, test_labels_3, lambda label: label in ['Spirals', 'Barred Spirals', 'Irregulars']
    )
    train_dataset_3_sb = CustomDataset(train_images_3_sb, train_labels_3_sb, transform=data_augmentation)
    val_dataset_3_sb = CustomDataset(val_images_3_sb, val_labels_3_sb, transform=default_transform)
    test_dataset_3_sb = CustomDataset(test_images_3_sb, test_labels_3_sb, transform=default_transform)

    # Création des DataLoaders
    dataloaders = {
        "dataloader1": {
            "train": DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True),
            "val": DataLoader(val_dataset_1, batch_size=batch_size, shuffle=False),
            "test": DataLoader(test_dataset_1, batch_size=batch_size, shuffle=False),
        },
        "dataloader2": {
            "train": DataLoader(train_dataset_2_el, batch_size=batch_size, shuffle=True),
            "val": DataLoader(val_dataset_2_el, batch_size=batch_size, shuffle=False),
            "test": DataLoader(test_dataset_2_el, batch_size=batch_size, shuffle=False),
        },
        "dataloader3": {
            "train": DataLoader(train_dataset_2_sbi, batch_size=batch_size, shuffle=True),
            "val": DataLoader(val_dataset_2_sbi, batch_size=batch_size, shuffle=False),
            "test": DataLoader(test_dataset_2_sbi, batch_size=batch_size, shuffle=False),
        },
        "dataloader4": {
            "train": DataLoader(train_dataset_3_sb, batch_size=batch_size, shuffle=True),
            "val": DataLoader(val_dataset_3_sb, batch_size=batch_size, shuffle=False),
            "test": DataLoader(test_dataset_3_sb, batch_size=batch_size, shuffle=False),
        },
    }

    print("DataLoaders prêts.")
    return dataloaders

def calculate_class_weights(train_loader):
    """
    Calcule les poids des classes en fonction de leur fréquence dans le dataset d'entraînement.
    Args:
        train_loader (DataLoader): DataLoader contenant les données d'entraînement.
    Returns:
        torch.Tensor: Les poids des classes sous forme de tenseur.
    """
    # Collecter tous les labels
    class_counts = Counter()
    for _, labels in train_loader:
        class_counts.update(labels.cpu().numpy().flatten().tolist())
    
    print("Répartition des classes dans le train dataset :", class_counts)
    
    # Obtenir les classes uniques et les réindexer de 0 à n-1
    unique_classes = sorted(class_counts.keys())
    class_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_classes)}
    
    # Créer un nouveau compteur avec les indices réindexés
    remapped_counts = Counter({class_mapping[old_idx]: count 
                             for old_idx, count in class_counts.items()})
    
    # Calculer les poids
    num_classes = len(unique_classes)
    counts = torch.tensor([remapped_counts[i] for i in range(num_classes)], 
                         dtype=torch.float32)
    
    total_samples = counts.sum()
    weights = total_samples / (len(unique_classes) * counts)
    
    print(f"Classes originales: {unique_classes}")
    print(f"Nouveau mapping: {class_mapping}")
    print("Poids pour les classes :", weights)
    
    return weights

def get_stage_labels(dataloader):
    """Extrait les labels uniques d'un dataloader et leur mapping."""
    # Prendre le premier batch pour obtenir les labels
    for _, labels in dataloader:
        remapped_labels, label_map = remap_labels(labels)
        return remapped_labels.unique().tolist(), label_map
        break

def get_all_stage_labels(dataloaders):
    """Récupère les labels remappés et leur mapping pour tous les stages."""
    # Initialiser des dictionnaires pour stocker les résultats
    labels = {}
    mappings = {}
    
    # Stage 1: S vs EL
    labels_1, mappings_1 = get_stage_labels(dataloaders["dataloader1"]["train"])
    labels['stage1'] = labels_1
    mappings['stage1'] = mappings_1
    
    # Stage 2: E vs L et S vs BS/I
    labels_2_EL, mappings_2_EL = get_stage_labels(dataloaders["dataloader2"]["train"])
    labels['stage2_EL'] = labels_2_EL
    mappings['stage2_EL'] = mappings_2_EL
    
    labels_2_SBI, mappings_2_SBI = get_stage_labels(dataloaders["dataloader3"]["train"])
    labels['stage2_SBI'] = labels_2_SBI
    mappings['stage2_SBI'] = mappings_2_SBI
    
    # Stage 3: S vs IS
    labels_3, mappings_3 = get_stage_labels(dataloaders["dataloader4"]["train"])
    labels['stage3'] = labels_3
    mappings['stage3'] = mappings_3
    
    return labels, mappings