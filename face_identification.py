from deepface import DeepFace
from tqdm import tqdm
import glob
import sys
import matplotlib.pyplot as plt 
import ipdb
import os

def count_items_in_directory(directory):
    total_count = 0
    for _, dirs, files in os.walk(directory):
        total_count += 1 # 디렉토리와 파일의 수를 더함
    return total_count

models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
metrics = ["cosine", "euclidean", "euclidean_l2"]

#1. DB 이미지 리스트 불러오기 - Prototype이 될 아이들
imgs = glob.glob('test/*.jpg')

#2. 이미지 없을 때 예외처리
if not imgs:
    print("이미지가 없습니다.")
    sys.exit()

#3. DB 이미지 리스트 전처리
# for img in imgs:
    
#     print(f"{img} is processed...")
#     aligned_img = DeepFace.extract_faces(img)
    
#     #ipdb.set_trace()
#     # os.makedirs('test_crop', exist_ok=True)
#     # new_img_addr = img.replace('test', 'test_crop')

#     plt.imsave(img,aligned_img[0]['face'])

#4. 입력 이미지 전처리
real_root = '/workspace/real_photo'
total_count = count_items_in_directory(real_root)

with tqdm(total=total_count, desc="Processing items") as pbar:
    for root, dirs, files in tqdm(os.walk(real_root)):

        if len(files) == 0:
            pbar.update(1)
            continue

        for file in files:
            input_img = os.path.join(root, file)

            new_root = root.replace('real_photo', 'real_photo_crop')
            os.makedirs(new_root, exist_ok=True)

            try:
                aligned_face = DeepFace.extract_faces(input_img)[0]['face']
            except:
                print(f"Error occur! - {input_img}")
                ipdb.set_trace()

            plt.imshow(aligned_face)
            plt.show()

            new_file = os.path.join(new_root, file)
            filename, _ = os.path.splitext(file) 
            base_name = os.path.basename(new_root)
            new_txt_file = os.path.join(new_root, base_name + '.txt')
            plt.imsave(new_file, aligned_face)

            with open(new_txt_file, 'w') as f:
                for i in range(len(models)):
                    #ipdb.set_trace()
                    print(f'model: {models[i]}')
                    print(f"filename: {file}")
                    df = DeepFace.find(img_path=new_file, db_path="./test", model_name=models[i], distance_metric=metrics[2], enforce_detection=False)

                    #ipdb.set_trace()
                    print(df)
                    f.write(f'model: {models[i]}')
                    f.write(df[0].to_string())
                    f.write('------------------------------------------')
                    f.write('------------------------------------------')
                    

        pbar.update(1)

            

