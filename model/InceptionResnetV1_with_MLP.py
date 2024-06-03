from facenet_pytorch import InceptionResnetV1
import torch.nn as nn
import torch

class InceptionResnetV1_with_MLP(nn.Module):
    def __init__(self, num_classes):
        super(InceptionResnetV1_with_MLP, self).__init__()

        self.frozen_model = InceptionResnetV1(
            classify=False,
            pretrained='vggface2',
        )
        for param in self.frozen_model.parameters():
            param.requires_grad = False
        
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, num_classes)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        with torch.no_grad():
            x = self.frozen_model(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)

        return x


