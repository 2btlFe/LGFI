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
from loss.loss import Main_Loss, Aux_Loss
from model.modified_facenet import ModifiedInceptionResnetV1

# 초기 random seed
seed_value = 5
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True

# Transform 
trans = transforms.Compose([
    # transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip the image horizontally
    # transforms.RandomRotation(degrees=15),  # Randomly rotate the image by up to 15 degrees
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, saturation, and hue
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


# Function to compute embedding
def get_embedding(model, img_tensor):
    with torch.no_grad():
        #ipdb.set_trace()
        embedding = model(img_tensor.unsqueeze(0))
    return embedding

def get_batch(root, triplet_list, batch_size=64, preprocess=True, need_meta=False):
    batch_steps = len(triplet_list)//batch_size
    
    for i in range(batch_steps+1):
        anchor   = []
        positive = []
        negative = []
        anchor_label = []
        negative_label = []
        
        j = i*batch_size
        
        while j<(i+1)*batch_size and j<len(triplet_list):
            a, p, n = triplet_list[j]
            anchor.append(trans(read_image(root, a)))
            positive.append(trans(read_image(root, p)))
            negative.append(trans(read_image(root, n)))
            
            anchor_label.append(int(a[0]))
            negative_label.append(int(n[0]))

            j+=1
        

        #ipdb.set_trace()
        anchor = torch.stack(anchor).cuda(non_blocking=True)        #[Batch, 3, 160, 160]
        positive = torch.stack(positive).cuda(non_blocking=True)    #[Batch, 3, 160, 160]
        negative = torch.stack(negative).cuda(non_blocking=True)    #[Batch, 3, 160, 160]
        anchor_label = torch.from_numpy(np.array(anchor_label))
        negative_label = torch.from_numpy(np.array(negative_label)) 

        # if preprocess:
        #     ipdb.set_trace()
        #     anchor = trans(anchor)
        #     positive = trans(positive)
        #     negative = trans(negative)
        
        if need_meta == True:
            yield ([[anchor, positive, negative], anchor_label, negative_label])
        else:
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
    parser.add_argument('--model_name', type=str, default='face_recognition')
    parser.add_argument('--main_loss', type=str, default='triplet')
    parser.add_argument('--aux_loss', type=str, default='fnp')
    parser.add_argument('--start_epoch', type=float, default=1.0)
    args = parser.parse_args()

    train_dir = args.train_dir
    val_dir = args.val_dir
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    
    reg_coeff = 0.005  # hyper parameter, you can tune this
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
    #1. 얼굴 crop을 위한 Validation dataset 생성
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
        classify=True,
        pretrained='vggface2',
        num_classes=60
    ).to(device)
    

    #------------------------------------------Freeze model -------------------------------------------#
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze some specific layers, for example, the last linear layer and last batch norm layer
    for param in model.last_linear.parameters():
        param.requires_grad = True
    for param in model.last_bn.parameters():
        param.requires_grad = True

    # Set different learning rates
    learning_rate_1 = 1e-4  # For frozen layers (though they are frozen, this shows the setup)
    learning_rate_2 = 1e-3  # For unfrozen layers

    # Define parameter groups
    param_groups = [
        {'params': model.conv2d_1a.parameters(), 'lr': learning_rate_1},
        {'params': model.conv2d_2a.parameters(), 'lr': learning_rate_1},
        {'params': model.conv2d_2b.parameters(), 'lr': learning_rate_1},
        {'params': model.last_linear.parameters(), 'lr': learning_rate_2},
        {'params': model.last_bn.parameters(), 'lr': learning_rate_2}
    ]

    # Create an optimizer
    optimizer = optim.Adam(param_groups)


    #model.logits = nn.Linear(in_features=128, out_features=60, bias=True)

    #ipdb.set_trace()

    #---------------------------------------------------------------------------------------------------#

    # Face Identification을 위한 Data List 불러오기--------------------------------------------------------#
    train_dir = train_dir + '_cropped'
    train_dir_list = next(os.walk(train_dir))[1]
    train_triplet = create_triplets(train_dir, train_dir_list)  # 이게 중요한 것인데 - Triplet을 학습 중에 골라내는 것이 아니라 갯수가 적으니 미리 random하게 지정해둔다 

    # CE loss를 위한 train_dataset
    train_dataset = datasets.ImageFolder(train_dir, transform=trans)
    train_loader = DataLoader(
        train_dataset,
        num_workers=workers,
        batch_size=batch_size,
        shuffle=True
        #collate_fn=collate_fn_pil
    )

    val_dir = val_dir + '_cropped'
    val_dir_list = next(os.walk(val_dir))[1]
    val_triplet  = create_triplets(val_dir, val_dir_list)

    # CE loss check를 위한 val_dataset
    val_dataset = datasets.ImageFolder(val_dir, transform=trans)
    val_loader = DataLoader(
        val_dataset,
        num_workers=workers,
        batch_size=batch_size,
        #collate_fn=collate_fn_pil
    )

    #---------------------------------------------------------------------------------------------------#

    print("Number of training triplets:", len(train_triplet))
    print("Number of validation triplets :", len(val_triplet))

    print("\nExamples of triplets:")
    for i in range(3):
        print(train_triplet[i])

    
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
    
    # Find Frozen Embedding Mu / Sigma    
    # class_dir = os.listdir(train_dir)

    # num_class = len(class_dir)
    # mu = [torch.zeros(512, requires_grad=False)] # Training Mean
    # sigma = [torch.ones(512, requires_grad=False)]            # Fixed Stdev

    # frozen_model.eval()
    # # Frozen Embedding 
    # with torch.no_grad():
    #     train_dir_list = os.listdir(train_dir)
    #     embedding = []
    #     total = 0
    #     for label in train_dir_list:
            
    #         path = os.path.join(train_dir, label)
    #         files = list(os.listdir(path))

    #         for file in files:
    #             idx = [label, file]
    #             img = trans(read_image(train_dir, idx)).to(device)   #tensor [3, 160, 160]
    #             temp_emb = get_embedding(frozen_model, img)  
    #             embedding.append(temp_emb)
            
    #         total += len(files)
    #         print(f"{label} : {len(files)} / {total}")


    #     # average embedding
    #     #ipdb.set_trace()
    #     avg_emb = torch.mean(torch.stack(embedding), dim=0)
    #     std_emb = torch.std(torch.stack(embedding), dim=0)
    #     #optimizer = torch.optim.SGD([param for param in mu if param.requires_grad], lr=0.01)
    # #---------------------------------------#

    # 학습 툴 지정 --------------------------------------------------------------------------------------#
    # Define Optimizer 
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Define Scheduler 
    scheduler = MultiStepLR(optimizer, [5, 10])
    # Define Loss and Evaluation functions
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    # Tensorboard Initialization
    work_dir = f'work_dir/{args.model_name}_{num_epochs}_{batch_size}'
    writer = SummaryWriter(work_dir)
    # Loss
    main1_loss = Main_Loss('ce').get_loss_function()
    main2_loss = Main_Loss(args.main_loss, criterion=criterion).get_loss_function()
    
    if args.aux_loss == 'randreg':
        #ipdb.set_trace()
        aux_loss = Aux_Loss(args.aux_loss, mu=avg_emb, sigma=std_emb).get_loss_function()
    else:
        aux_loss = Aux_Loss(args.aux_loss).get_loss_function()
    # ------------------------------------------------------------------------------------------------#
    
    # split epoch
    ce_epochs = int(num_epochs * args.start_epoch)
    triplet_epochs = int(num_epochs* (1.0 - args.start_epoch))


    # CE loss 적용
    for epoch in range(0, ce_epochs):
        # train loss 
        train_loss_sum = 0.0
        sup_loss_sum = 0.0
        num_batches = 0
        
        # triplet 다시 만들기 - 매번 random하게 지정될 필요가 있다
        train_dir_list = next(os.walk(train_dir))[1]
        train_triplet = create_triplets(train_dir, train_dir_list)  # 이게 중요한 것인데 - Triplet을 학습 중에 골라내는 것이 아니라 갯수가 적으니 미리 random하게 지정해둔다 
        
        # Triplet에 대해서 학습 적용하기
        for batch_idx, sample in enumerate(train_loader):
            # 에포크마다 시드를 변경 (옵션)
            random.seed(epoch)
            np.random.seed(epoch)
            torch.manual_seed(epoch)
            torch.cuda.manual_seed_all(epoch)
            
            # Main loss
            sup_loss = main1_loss(model, sample)
            sup_loss_sum += sup_loss.item()

            # # Aux loss
            # reg_loss = aux_loss(model, frozen_model, sample)
            # reg_loss_sum += reg_loss.item()

            # Total loss
            loss = sup_loss 
            train_loss_sum += loss.item()

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_batches += 1

        avg_train_loss = train_loss_sum / num_batches
        avg_sup_loss = sup_loss_sum / num_batches
        # avg_reg_loss = reg_loss_sum / num_batches
        print(f"Epoch {epoch+1}/{ce_epochs}, Training Loss: {avg_train_loss:.4f}")

        # TensorBoard에 기록
        writer.add_scalar('Loss/Train_CE_loss', avg_train_loss, epoch)
        writer.add_scalar('Loss/Sup_CE_loss', avg_sup_loss, epoch)
        # writer.add_scalar('Loss/Reg_loss', avg_reg_loss, epoch)

        # validation per 10 epochs
        if epoch % 5 == 0:
            val_loss_sum = 0.0
            val_sup_loss_sum = 0.0
            # val_reg_loss_sum = 0.0
            correct_predictions = 0
            total_samples = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, sample in enumerate(val_loader):

                    # Main loss
                    val_sup_loss = main1_loss(model, sample)
                    val_sup_loss_sum += val_sup_loss.item()

                    # Total loss
                    loss = val_sup_loss 
                    val_loss_sum += loss.item()

                    # Accuracy Calculation
                    # ipdb.set_trace()
                    img, target = sample
                    img = img.cuda()
                    target = target.cuda()

                    logit = model(img)
                    predict = torch.argmax(logit, dim=1)
                    correct_predictions += (predict == target).sum().item()
                    total_samples += target.size(0)
                    
            avg_val_loss = val_loss_sum / (len(val_triplet) / 32)
            avg_val_sup_loss = val_sup_loss_sum / (len(val_triplet) / 32)
            epoch_accuracy = correct_predictions / total_samples
            # avg_val_reg_loss = val_reg_loss_sum / (len(val_triplet) / 32)
            
            print(f'Epoch {epoch+1}/{ce_epochs}, Loss: {avg_val_loss:.4f}, Sup_loss : {avg_val_sup_loss:.4f} Accuracy: {epoch_accuracy * 100:.2f}%')

            # TensorBoard에 기록
            writer.add_scalar('Loss/Validation_CE_loss', avg_val_loss, epoch)
            writer.add_scalar('Loss/Validation_sup_CE_loss', avg_val_sup_loss, epoch)
            # writer.add_scalar('Loss/Validation_reg_loss', avg_val_reg_loss, epoch)

            # Train Mode로 다시 돌아오기
            model.train()


    # Model 변경
    model = ModifiedInceptionResnetV1(model)

    # Triplet loss 적용
    for epoch in range(ce_epochs, num_epochs):
        # train loss 
        train_loss_sum = 0.0
        sup_loss_sum = 0.0
        reg_loss_sum = 0.0
        num_batches = 0
        
        # triplet 다시 만들기 - 매번 random하게 지정될 필요가 있다
        train_dir_list = next(os.walk(train_dir))[1]
        train_triplet = create_triplets(train_dir, train_dir_list)  # 이게 중요한 것인데 - Triplet을 학습 중에 골라내는 것이 아니라 갯수가 적으니 미리 random하게 지정해둔다 
        
        # Triplet에 대해서 학습 적용하기
        for batch_label in get_batch(train_dir, train_triplet, batch_size=32, preprocess=True, need_meta=True):
            random.seed(epoch)
            np.random.seed(epoch)
            torch.manual_seed(epoch)
            torch.cuda.manual_seed_all(epoch)

            # Main loss
            sup_loss = main2_loss(model, batch_label)
            sup_loss_sum += sup_loss.item()

            # Aux loss
            reg_loss = aux_loss(model, frozen_model, batch_label)
            reg_loss_sum += reg_loss.item()

            # Total loss
            loss = sup_loss + reg_loss * reg_coeff
            train_loss_sum += loss.item()

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_batches += 1

        avg_train_loss = train_loss_sum / num_batches
        avg_sup_loss = sup_loss_sum / num_batches
        avg_reg_loss = reg_loss_sum / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # TensorBoard에 기록
        writer.add_scalar('Loss/Train_loss', avg_train_loss, epoch)
        writer.add_scalar('Loss/Sup_loss', avg_sup_loss, epoch)
        writer.add_scalar('Loss/Reg_loss', avg_reg_loss, epoch)

        # validation per 10 epochs
        if epoch % 5 == 0:
            val_loss_sum = 0.0
            val_sup_loss_sum = 0.0
            val_reg_loss_sum = 0.0
            model.eval()
            with torch.no_grad():
                for val_batch_label in get_batch(val_dir, val_triplet, batch_size=32, preprocess=True, need_meta=True):
                    
                    # Main loss
                    val_sup_loss = main2_loss(model, val_batch_label)
                    val_sup_loss_sum += val_sup_loss.item()

                    # Aux loss
                    val_reg_loss = aux_loss(model, frozen_model, val_batch_label)
                    val_reg_loss_sum += val_reg_loss.item()

                    # Total loss
                    loss = val_sup_loss + val_reg_loss * reg_coeff
                    val_loss_sum += loss.item()

            avg_val_loss = val_loss_sum / (len(val_triplet) / 32)
            avg_val_sup_loss = val_sup_loss_sum / (len(val_triplet) / 32)
            avg_val_reg_loss = val_reg_loss_sum / (len(val_triplet) / 32)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")

            # TensorBoard에 기록
            writer.add_scalar('Loss/Validation_loss', avg_val_loss, epoch)
            writer.add_scalar('Loss/Validation_sup_loss', avg_val_sup_loss, epoch)
            writer.add_scalar('Loss/Validation_reg_loss', avg_val_reg_loss, epoch)

            # Train Mode로 다시 돌아오기
            model.train()
    
    # save the model 
    save_path = f'{work_dir}/{args.model_name}_{num_epochs}_{batch_size}_deepfake.pth'
    state_dict = model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('logits')}
    torch.save(state_dict, save_path)
    writer.close()
