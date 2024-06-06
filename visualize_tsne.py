import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from facenet_pytorch import fixed_image_standardization, InceptionResnetV1
from torchvision import datasets, transforms
import torch.nn as nn
import argparse

class InceptionResnetV1_with_MLP(nn.Module):
    def __init__(self, num_classes, device):
        super(InceptionResnetV1_with_MLP, self).__init__()

        self.device = device
        
        self.frozen_model = InceptionResnetV1(
            classify=False,
            pretrained='vggface2',
        ).to(self.device)
        for param in self.frozen_model.parameters():
            param.requires_grad = False
        
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            x = self.frozen_model(x)

        x = self.relu(self.linear1(x))
        x = self.linear2(x)

        return x

def get_frozen_embeddings(dataloader, model, device):
    embeddings = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            emb = model.frozen_model(imgs).cpu().detach().numpy()
            embeddings.extend(emb)
            labels.extend(lbls.numpy())
    return np.array(embeddings), np.array(labels)

def filter_embeddings_by_classes(embeddings, labels, main_class, other_classes):
    filtered_embeddings = []
    filtered_labels = []
    for emb, label in zip(embeddings, labels):
        if label == main_class or label in other_classes:
            filtered_embeddings.append(emb)
            filtered_labels.append(label)
    return np.array(filtered_embeddings), np.array(filtered_labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='T-SNE visualization for specific classes')
    parser.add_argument('--main_class', type=int, required=True, help='Main class to visualize')
    parser.add_argument('--other_classes', type=int, nargs='+', required=True, help='Other classes to visualize')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the trained model checkpoint')
    args = parser.parse_args()

    main_class = args.main_class
    other_classes = args.other_classes

    # Transformations for the images
    trans = transforms.Compose([
        transforms.Lambda(lambda img: np.array(img, dtype=np.float32)),  # PIL 이미지를 float32로 변환
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        fixed_image_standardization  # 고정 이미지 표준화 적용
    ])

    # Load datasets
    train_dir = './Face_Dataset/Train_cropped'
    val_dir = './Face_Dataset/Validation_cropped'

    train_dataset = datasets.ImageFolder(train_dir, transform=trans)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = datasets.ImageFolder(val_dir, transform=trans)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # Load model and checkpoint
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = InceptionResnetV1_with_MLP(num_classes=60, device=device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # Extract embeddings for train and test sets
    train_embeddings, train_labels = get_frozen_embeddings(train_loader, model, device)
    test_embeddings, test_labels = get_frozen_embeddings(val_loader, model, device)

    # Print the number of embeddings and labels extracted
    print(f'Train embeddings: {train_embeddings.shape}, Train labels: {train_labels.shape}')
    print(f'Test embeddings: {test_embeddings.shape}, Test labels: {test_labels.shape}')

    # Filter embeddings by the specified classes for train data
    filtered_train_embeddings, filtered_train_labels = filter_embeddings_by_classes(train_embeddings, train_labels, main_class, other_classes)
    print(f'Filtered train embeddings: {filtered_train_embeddings.shape}, Filtered train labels: {filtered_train_labels.shape}')
    
    # Filter embeddings by the specified classes for test data (only for main class)
    filtered_test_embeddings, filtered_test_labels = filter_embeddings_by_classes(test_embeddings, test_labels, main_class, [])
    print(f'Filtered test embeddings: {filtered_test_embeddings.shape}, Filtered test labels: {filtered_test_labels.shape}')

    # Perform T-SNE with adjusted perplexity for train data
    if len(filtered_train_embeddings) > 1:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(filtered_train_embeddings) - 1))
        train_embeddings_2d = tsne.fit_transform(filtered_train_embeddings)
    else:
        train_embeddings_2d = np.zeros((len(filtered_train_embeddings), 2))

    # Perform T-SNE with adjusted perplexity for test data (only for main class)
    if len(filtered_test_embeddings) > 1:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(filtered_test_embeddings) - 1))
        test_embeddings_2d = tsne.fit_transform(filtered_test_embeddings)
    else:
        test_embeddings_2d = np.zeros((len(filtered_test_embeddings), 2))

    # Print the shape of T-SNE results
    print(f'T-SNE train embeddings: {train_embeddings_2d.shape}')
    print(f'T-SNE test embeddings: {test_embeddings_2d.shape}')

    # Visualization
    plt.figure(figsize=(12, 6))

    ax = plt.gca()

    # Train data visualization for main class
    for i, (x, y) in enumerate(train_embeddings_2d):
        label = filtered_train_labels[i]
        if label == main_class:
            ax.scatter(x, y, color='green', label='Train Main Class' if 'Train Main Class' not in ax.get_legend_handles_labels()[1] else "", marker='o')
            ax.annotate(f'Train {label}', (x, y), textcoords='offset points', xytext=(0, 0), ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='green'))
        else:
            ax.scatter(x, y, color='blue', label=f'Train Other Class {label}' if f'Train Other Class {label}' not in ax.get_legend_handles_labels()[1] else "", marker='o')
            ax.annotate(f'Train {label}', (x, y), textcoords='offset points', xytext=(0, 0), ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='blue'))

    # Test data visualization for main class
    for i, (x, y) in enumerate(test_embeddings_2d):
        label = filtered_test_labels[i]
        ax.scatter(x, y, color='red', label='Test Main Class' if 'Test Main Class' not in ax.get_legend_handles_labels()[1] else "", marker='x')
        ax.annotate(f'Test {label}', (x, y), textcoords='offset points', xytext=(0, 0), ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', edgecolor='none', facecolor='red'))

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.title('T-SNE visualization of specified classes')
    plt.xlabel('T-SNE Component 1')
    plt.ylabel('T-SNE Component 2')
    plt.show()
