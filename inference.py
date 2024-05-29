from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
from glob import glob
import ipdb
import random
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import argparse


random.seed(5)
np.random.seed(5)
torch.manual_seed(5)

# Transform 

trans = transforms.Compose([
    transforms.Lambda(lambda img: np.array(img, dtype=np.float32)),  # PIL 이미지를 float32로 변환
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    fixed_image_standardization  # 고정 이미지 표준화 적용
])

test_trans = transforms.Compose([
    transforms.Lambda(lambda img: np.array(img, dtype=np.float32)),  # PIL 이미지를 float32로 변환
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    fixed_image_standardization  # 고정 이미지 표준화 적용
])

# Using Cosine_similarity
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

cache_img = {}
def read_image(root, index):
    path = os.path.join(root, index[0], index[1])
    if path not in cache_img:
        image = trans(Image.open(path))
        cache_img[path] = image
    else:
        image = cache_img[path]
    return image

# Function to compute embedding
def get_embedding(model, img_tensor):
    with torch.no_grad():
        #ipdb.set_trace()
        embedding = model(img_tensor.unsqueeze(0))
    return embedding

# Function to perform face identification
def identify_face_top1(input_embedding, known_embeddings):
    min_distance = float('inf')
    identity = "Unknown"
    for name, embedding in known_embeddings.items():
        distance = torch.dist(input_embedding, embedding)
        if distance < min_distance:
            min_distance = distance
            identity = name
    return identity

def identify_face_top_k(input_embedding, known_embeddings, top_k=5):
    distances = []
    for name, embedding in known_embeddings.items():
        distance = torch.dist(input_embedding, embedding)
        distances.append((name, distance))
    
    # 거리 순으로 정렬
    distances.sort(key=lambda x: x[1])
    
    # 상위 top_k 반환
    top_k_results = distances[:top_k]
    return top_k_results

def identify_face(input_embedding, known_embeddings, top_k=5):
    distances = []
    for name, embedding in known_embeddings.items():
        for emb in embedding:
            distance = torch.dist(input_embedding, emb)
            distances.append((name, distance))

    # 거리 순으로 정렬
    distances.sort(key=lambda x: x[1])
    
    # 상위 top_k 반환
    top_k_results = distances[:top_k]
    return top_k_results

if __name__ == "__main__":

    # Inference Argument
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--test_dir', type=str, default='./Face_Dataset/Test', help='Test Image Directory')
    parser.add_argument('--batch_size', type=int, default='1', help='batch_size')
    parser.add_argument('--pretrained', type=str, default='None', help='batch_size')
    args = parser.parse_args()

    workers = 0 if os.name == 'nt' else 8

    # GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    # Generate Test_cropped
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    test_dir = args.test_dir
    batch_size = args.batch_size

    # Test Dataset Generation
    dataset = datasets.ImageFolder(test_dir, transform=transforms.Resize((512, 512)))
    dataset.samples = [
        (p, p.replace(test_dir, test_dir + '_cropped'))
            for p, _ in dataset.samples
    ]
    
    # DataLoader for crop
    loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )

    # Crop by MTCNN
    for i, (x, y) in enumerate(loader):
        mtcnn(x, save_path=y)
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
        
    # Remove mtcnn to reduce GPU memory usage
    del mtcnn

    if args.pretrained == 'None':
        model = InceptionResnetV1(
            classify=False,
            pretrained='vggface2',
            #num_classes=len(dataset.class_to_idx)
        ).to(device)
    else:
        model = InceptionResnetV1(
            classify=False,
            pretrained='vggface2'  # 사전 학습된 가중치를 사용하지 않음
        ).to(device)
        state_dict = torch.load(args.pretrained)
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Generate Dataset
    test_dir = test_dir + '_cropped'
    
    # Dataset 및 loader 제작
    test_dataset = datasets.ImageFolder(test_dir, transform=trans)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    class_names = test_dataset.classes
    print("Class names:", class_names)

    # Target Embedding 제작
    target_embeddings = {}
    target_dir = 'Face_Dataset/Train_cropped'

    #ipdb.set_trace()

    target_dir_list = os.listdir(target_dir)
    #ipdb.set_trace()

    for label in target_dir_list:
        
        path = os.path.join(target_dir, label)
        files = list(os.listdir(path))

        embedding = []
        for file in files:
            idx = [label, file]
            img = read_image(target_dir, idx).to(device)   #tensor [3, 160, 160]
            temp_emb = get_embedding(model, img)  
            embedding.append(temp_emb)
        
        # average embedding
        # avg_emb = torch.mean(torch.stack(embedding), dim=0)
        
        # save to target_embedding dictionary
        # target_embeddings[label] = avg_emb

        target_embeddings[label] = embedding


    correct  = 0
    total = 0
    for image, label in test_loader:
        
        image = image.to(device)
        label = class_names[int(label)]
        test_emb = get_embedding(model, image.squeeze(0))
        
        #ipdb.set_trace()
        top_5_results = identify_face(test_emb, target_embeddings, top_k=5)
        print(f"{label} result:")
        for name, distance in top_5_results:
            print(f"Name: {name}, Distance: {distance}")

        #ipdb.set_trace()
        gt = str(label)
        top_5_names = [name for name, _ in top_5_results]
        is_gt_in_top_5 = gt in top_5_names
        
        if is_gt_in_top_5:
            correct += 1
        total += 1


        #ipdb.set_trace()
        # id = identify_face(test_emb, target_embeddings)
        # gt = str(label.item())

        # print(f"predict: {id} <-> {gt} : label")

        # if id == gt:
        #     correct += 1
        # total += 1

    accuracy = correct / total
    print("\n")
    print(f"Accuracy: {accuracy:.4f}")
