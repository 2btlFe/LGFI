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

''' 
trans를 변형
Deepfake Augementation - (병철, 윤지)



'''

'''
단순 Augmentation 적용 - (찬용, 성혁)
Color jittering
Contrast
Hue
Horizontal Flip

'''



# Using Cosine_similarity
def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

# Triplet Combination will repalce the imageloader
def create_triplets(directory, folder_list, max_files=3):
    triplets = []
    folders = folder_list
    
    for folder in folders:
        path = os.path.join(directory, folder)
        files = list(os.listdir(path))[:max_files]
        num_files = len(files)
        
        for i in range(num_files-1):
            for j in range(i+1, num_files):
                anchor = (folder, files[i])
                positive = (folder, files[j])

                neg_folder = folder
                while neg_folder == folder:
                    neg_folder = random.choice(folders)

                neg_files = list(os.listdir(os.path.join(directory, neg_folder)))[:max_files]
                selected_neg = random.choice(neg_files)
                negative = (neg_folder, selected_neg)

                triplets.append((anchor, positive, negative))
    
    random.shuffle(triplets)
    return triplets

cache_img = {}
def read_image(root, index):
    path = os.path.join(root, index[0], index[1])
    if path not in cache_img:
        image = Image.open(path)
        cache_img[path] = image
    else:
        image = cache_img[path]
    return image

def get_batch(root, triplet_list, batch_size=64, preprocess=True):
    batch_steps = len(triplet_list)//batch_size
    
    for i in range(batch_steps+1):
        anchor   = []
        positive = []
        negative = []
        
        j = i*batch_size
        while j<(i+1)*batch_size and j<len(triplet_list):
            a, p, n = triplet_list[j]
            anchor.append(trans(read_image(root, a)))
            positive.append(trans(read_image(root, p)))
            negative.append(trans(read_image(root, n)))
            j+=1
        
        #ipdb.set_trace()
        anchor = torch.stack(anchor).cuda(non_blocking=True)
        positive = torch.stack(positive).cuda(non_blocking=True)
        negative = torch.stack(negative).cuda(non_blocking=True)
        
        # if preprocess:
        #     ipdb.set_trace()
        #     anchor = trans(anchor)
        #     positive = trans(positive)
        #     negative = trans(negative)
        
        yield ([anchor, positive, negative])

        del anchor, positive, negative
        torch.cuda.empty_cache()

def visualize_triplets(root, triplet, batch_size, output_file):
    num_plots = 6
    f, axes = plt.subplots(num_plots, 3, figsize=(15, 20))
    for x in get_batch(root, triplet, batch_size=num_plots, preprocess=False):
        a,p,n = x
        for i in range(num_plots):
            axes[i, 0].imshow(a[i])
            axes[i, 1].imshow(p[i])
            axes[i, 2].imshow(n[i])
            i+=1
        break
    
    plt.tight_layout()
    plt.savefig(output_file, format='jpg')
    plt.close(f)


if __name__ == "__main__":
    # Inference Argument
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--train_dir', type=str, default='./Face_Dataset/Train', help='Train Image Directory')
    parser.add_argument('--val_dir', type=str, default='./Face_Dataset/Validation', help='Validation Image Directory')
    parser.add_argument('--batch_size', type=int, default='32', help='batch_size')
    parser.add_argument('--num_epochs', type=int, default='100', help='batch_size')
    args = parser.parse_args()

    train_dir = args.train_dir
    val_dir = args.val_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    workers = 0 if os.name == 'nt' else 8

    # Load GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))


    #----------Training Set 얼굴 crop--------------------------------------------------------------------#

    # 1. MTCNN 불러오기
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )

    # 2. 얼굴 Crop을 위한 Dataset 지정
    dataset = datasets.ImageFolder(train_dir, transform=transforms.Resize((512, 512)))
    dataset.samples = [
        (p, p.replace(train_dir, train_dir + '_cropped'))
            for p, _ in dataset.samples
    ]
    
    # 3. 얼굴 Crop을 위한 데이터 로더 지정
    loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )

    # 4. 실제 얼굴 Crop 실행 및 저장 (Train_cropped에 저장됨)
    for i, (x, y) in enumerate(loader):
        mtcnn(x, save_path=y)
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
        
    #--------------------------------------------------------------------------------------------------#

    #----------Validztion Set 얼굴 crop--------------------------------------------------------------------#
    # 1. 얼굴 crop을 위한 Validation dataset 생성
    val_dataset = datasets.ImageFolder(val_dir, transform=transforms.Resize((512, 512)))
    val_dataset.samples = [
        (p, p.replace(val_dir, val_dir + '_cropped'))
            for p, _ in val_dataset.samples
    ]
    
    # 2. 데이터로더 제작
    val_loader = DataLoader(
        val_dataset,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )

    # 3. 얼굴 crop 실행
    for i, (x, y) in enumerate(val_loader):
        mtcnn(x, save_path=y)
        print('\rBatch {} of {}'.format(i + 1, len(val_loader)), end='')

    # mtcnn GPU에서 내리기
    del mtcnn
    #----------------------------------------------------------------------------------------------------#



    # InceptionResnetV1 불러오기 - vggface2로 pretrained한 모델 가져오기--------------------------------------#
    model = InceptionResnetV1(
        classify=False,
        pretrained='vggface2',
    ).to(device)
    #---------------------------------------------------------------------------------------------------#

    # Face Identification을 위한 Data List 불러오기--------------------------------------------------------#
    train_dir = train_dir + '_cropped'
    train_dir_list = next(os.walk(train_dir))[1]
    train_triplet = create_triplets(train_dir, train_dir_list)  # 이게 중요한 것인데 - Triplet을 학습 중에 골라내는 것이 아니라 갯수가 적으니 미리 random하게 지정해둔다 

    val_dir = val_dir + '_cropped'
    val_dir_list = next(os.walk(val_dir))[1]
    val_triplet  = create_triplets(val_dir, val_dir_list)
    #---------------------------------------------------------------------------------------------------#


    print("Number of training triplets:", len(train_triplet))
    print("Number of validation triplets :", len(val_triplet))

    print("\nExamples of triplets:")
    for i in range(3):
        print(train_triplet[i])


    # 학습 툴 지정 --------------------------------------------------------------------------------------#
    # Define Optimizer 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Define Scheduler 
    scheduler = MultiStepLR(optimizer, [5, 10])
    # Define Loss and Evaluation functions
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    # Tensorboard Initialization
    writer = SummaryWriter('runs/face_recognition_experiment_first_order')
    # ------------------------------------------------------------------------------------------------#


    # Training----------------------------------------------------------------------------------------#
    
    # Frozen model 지정----------------------#
    frozen_model = InceptionResnetV1(
        classify=False,
        pretrained='vggface2',
    ).to(device)

    # 모델의 모든 파라미터를 동결시킵니다.
    for param in frozen_model.parameters():
        param.requires_grad = False
    #---------------------------------------#
        
    for epoch in range(num_epochs):
        # train loss 
        train_loss = 0.0
        #reg_loss = 0.0
        num_batches = 0

        # triplet 다시 만들기 - 매번 random하게 지정될 필요가 있다


        # Triplet에 대해서 학습 적용하기
        for batch in get_batch(train_dir, train_triplet, batch_size=32, preprocess=True):
            
            #ipdb.set_trace()
            anchor, positive, negative = batch # 삼중항 선택
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)
            
            loss = criterion(anchor_embedding, positive_embedding, negative_embedding)
            
            '''
            (예은, 선재)
            Regularization Term 추가 - 여기만 Argument화 하든 class화 하든 여러번 실험하면 좋다
            ''' 
            # 이건 예시입니다
            
            # frozen_anchor_embedding = frozen_model(anchor)
            # frozen_pos_embedding = frozen_model(positive)
            # frozen_neg_embedding = frozen_model(negative)

            # frozen_anchor_dist=  anchor_embedding - frozen_anchor_embedding
            # frozen_pos_dist = positive_embedding - frozen_pos_embedding
            # frozen_neg_dist = negative_embedding - frozen_neg_embedding
            
            # reg_coeff = 0.005   # hyper parameter
            # loss_reg = torch.norm(frozen_anchor_dist, p=2) + torch.norm(frozen_pos_dist, p=2) + torch.norm(frozen_neg_dist, p=2)
            # loss += loss_reg * reg_coeff
            
            ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record train_loss
            train_loss += loss.item()
            num_batches += 1


        avg_train_loss = train_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # TensorBoard에 기록
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        #writer.add_scalar('Loss/Reg', reg_loss, epoch)

        # validation per 10 epochs
        if epoch % 10 == 0:
            val_loss = 0.0
            model.eval()
            with torch.no_grad():
                for val_batch in get_batch(val_dir, val_triplet, batch_size=32, preprocess=True):
                    anchor, positive, negative = val_batch
                    anchor_embedding = model(anchor)
                    positive_embedding = model(positive)
                    negative_embedding = model(negative)
                    
                    val_loss += criterion(anchor_embedding, positive_embedding, negative_embedding).item()

                    # reg term
                    '''
                    frozen_anchor_embedding = frozen_model(anchor)
                    frozen_pos_embedding = frozen_model(positive)
                    frozen_neg_embedding = frozen_model(negative)

                    frozen_anchor_dist=  anchor_embedding - frozen_anchor_embedding
                    frozen_pos_dist = positive_embedding - frozen_pos_embedding
                    frozen_neg_dist = negative_embedding - frozen_neg_embedding
                    
                    reg_coeff = 0.005   # hyper parameter
                    loss_reg = torch.norm(frozen_anchor_dist, p=2) + torch.norm(frozen_pos_dist, p=2) + torch.norm(frozen_neg_dist, p=2)
                    val_loss += loss_reg * reg_coeff
                    '''

            avg_val_loss = val_loss / (len(val_triplet) / 32)
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

            # TensorBoard에 기록
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)

            # Train Mode로 다시 돌아오기
            model.train()
    
    # save the model 
    os.makedirs('model', exist_ok=True)
    save_path = f'model/Facenet512_transfer_learning_{num_epochs}_{batch_size}_deepfake.pth'
    state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('logits')}
    torch.save(state_dict, save_path)
    writer.close()
