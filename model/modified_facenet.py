from facenet_pytorch import InceptionResnetV1
import torch.nn as nn
import torch

class ModifiedInceptionResnetV1(nn.Module):
    def __init__(self, original_model):
        super(ModifiedInceptionResnetV1, self).__init__()
        # 원래 모델의 모든 레이어를 복사합니다.
        self.conv2d_1a = original_model.conv2d_1a
        self.conv2d_2a = original_model.conv2d_2a
        self.conv2d_2b = original_model.conv2d_2b
        self.maxpool_3a = original_model.maxpool_3a
        self.conv2d_3b = original_model.conv2d_3b
        self.conv2d_4a = original_model.conv2d_4a
        self.conv2d_4b = original_model.conv2d_4b
        #self.maxpool_5a = original_model.maxpool_5a
        #self.mixed_5b = original_model.mixed_5b
        self.repeat_1 = original_model.repeat_1
        self.mixed_6a = original_model.mixed_6a
        self.repeat_2 = original_model.repeat_2
        self.mixed_7a = original_model.mixed_7a
        self.repeat_3 = original_model.repeat_3
        self.block8 = original_model.block8
        # self.conv2d_7b = original_model.conv2d_7b
        self.avgpool_1a = original_model.avgpool_1a
        self.dropout = original_model.dropout
        self.last_linear = original_model.last_linear
        self.last_bn = original_model.last_bn
        # logits 레이어는 제외합니다.
    
    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        # x = self.maxpool_5a(x)
        # x = self.mixed_5b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        # x = self.conv2d_7b(x)
        x = self.avgpool_1a(x)
        x = torch.flatten(x, 1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.last_linear(x)
        if self.last_bn is not None:
            x = self.last_bn(x)
        # logits 레이어를 거치지 않고 last_bn의 출력을 반환합니다.
        return x

if __name__ == "__main__":

    # 원래 InceptionResnetV1 모델을 불러옵니다.
    original_model = InceptionResnetV1(
        classify=True,
        pretrained='vggface2'
    ).to(device)

    # 원래 모델을 수정된 모델로 만듭니다.
    modified_model = ModifiedInceptionResnetV1(original_model).to(device)

    # 예시 입력 텐서
    example_input = torch.randn(1, 3, 160, 160).to(device)

    # 모델의 출력 확인
    output = modified_model(example_input)
    print(output.shape)