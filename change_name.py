import os

# 현재 디렉토리의 모든 파일을 순회

os.chdir("./test")
for filename in os.listdir('.'):
    # 파일 확장자가 .jpg인 경우
    if filename.endswith('.jpg'):
        # 새 파일 이름 생성
        new_filename = 'img' + filename
        # 파일 이름 변경
        os.rename(filename, new_filename)
        print(f'Renamed {filename} to {new_filename}')

    elif filename.endswith('.jpeg'):
        
        # 기본 파일 이름과 확장자 분리
        base = os.path.splitext(filename)[0]

        # 새 파일 이름 생성
        new_filename = f'img{base}.jpg'
        
        # 파일 이름 변경
        os.rename(filename, new_filename)
        print(f'Renamed {filename} to {new_filename}')