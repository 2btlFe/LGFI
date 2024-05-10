from deepface import DeepFace
from tqdm import tqdm
import glob
import sys
import matplotlib.pyplot as plt 
import ipdb

models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib"]
metrics = ["cosine", "euclidean", "euclidean_l2"]

#1. DB 이미지 리스트 불러오기
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
#     plt.imsave(img,aligned_img[0]['face'])

#4. 입력 이미지 전처리
input_img = "/workspace/real_photo/0/0_5.jpg"
aligned_face = DeepFace.extract_faces(input_img)[0]['face']
plt.imshow(aligned_face)
plt.show()
plt.imsave(input_img, aligned_face)


for i in range(len(models)):
    df = DeepFace.find(img_path=input_img, db_path="./test", model_name=models[i], distance_metric=metrics[2], enforce_detection=False)
    print(df)

