import torch
import ipdb
import torch.nn as nn

class Main_Loss:
    def __init__(self, loss_type, **kwargs):
        self.loss_type = loss_type.lower()
        self.kwargs = kwargs

    def get_loss_function(self):
        if self.loss_type == 'triplet':
            return self.Triplet(**self.kwargs)
        # elif self.loss_type == 'ce':
        #     return self.CE(**self.kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    class Triplet(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            # FNP specific initialization here
            # For example:
            # self.some_parameter = kwargs.get('some_parameter', default_value)
            self.criterion = kwargs.get('criterion', nn.TripletMarginLoss(margin=1.0, p=2))
            
        def forward(self, model, batch_label):
            # Implement the FNP loss calculation here
            loss = 0.0

            batch, anchor_label, negative_label = batch_label
            anchor, positive, negative = batch # 삼중항 선택
            
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)
            
            loss = self.criterion(anchor_embedding, positive_embedding, negative_embedding)
            return loss

    # class CE(nn.Module):
    #     def __init__(self, **kwargs):
    #         super().__init__()
    #         # FNP specific initialization here
    #         # For example:
    #         # self.some_parameter = kwargs.get('some_parameter', default_value)
    #         self.criterion = kwargs.get('criterion', nn.TripletMarginLoss(margin=1.0, p=2))
            
    #     def forward(self, model, batch):
    #         # Implement the FNP loss calculation here
    #         loss = 0.0

    #         anchor, positive, negative = batch
    #         anchor_embedding = model(anchor)
    #         positive_embedding = model(positive)
    #         negative_embedding = model(negative)
            
    #         loss = self.criterion(anchor_embedding, positive_embedding, negative_embedding)
    #         return loss

class Aux_Loss:
    def __init__(self, loss_type, **kwargs):
        self.loss_type = loss_type.lower()
        self.kwargs = kwargs

    def get_loss_function(self):
        if self.loss_type == 'fnp':
            return self.L2(**self.kwargs)
        elif self.loss_type == 'l2sp':
            return self.L2SP(**self.kwargs)
        elif self.loss_type == 'randreg':
            return self.RandReg(**self.kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    class L2(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            # FNP specific initialization here
            # For example:
            # self.some_parameter = kwargs.get('some_parameter', default_value)
        
        def forward(self, model, frozen_model, batch_label):
            # Implement the FNP loss calculation here
            loss = 0.0

            batch, anchor_label, negative_label = batch_label
            anchor, positive, negative = batch # 삼중항 선택

            for sample in batch:
                embedding = model(sample)
                frozen_embedding = frozen_model(sample)
                dist = embedding - frozen_embedding
                loss += torch.norm(dist, p=2)
            
            return loss

    class L2SP(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            # L2SP specific initialization here
            # For example:
            # self.some_parameter = kwargs.get('some_parameter', default_value)

        def forward(self, model, frozen_model, batch):
            # Implement the L2SP loss calculation here
            loss = 0.0
            
            pretrained_params = {name: param.clone().detach() for name, param in frozen_model.named_parameters()}

            for name, param in model.named_parameters():
                if name in pretrained_params:
                    loss += torch.norm(param - pretrained_params[name], p=2)

            return loss

    class RandReg(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            # RandReg specific initialization here
            # For example:
            self.mu = kwargs.get('mu', torch.zeros(512, requires_grad=False))
            self.sigma = kwargs.get('sigma', torch.ones(512, requires_grad=False))

        def ema(self, mu_bar, mu_hat, alpha=0.5):
            return alpha*mu_bar + (1 - alpha)*mu_hat

        def update(self, mu):
            with torch.no_grad():
                self.mu = self.ema(self.mu, mu, 0.9)

        def forward(self, model, frozen_model, batch_label):
            # Implement the RandReg loss calculation here
            with torch.no_grad():
                z = torch.normal(mean=self.mu, std=self.sigma).unsqueeze(0).cuda()

            #ipdb.set_trace()

            batch, anchor_label, negative_label = batch_label
            anchor, positive, negative = batch # 삼중항 선택
            anchor_embedding = model(anchor)
            positive_embedding = model(positive)
            negative_embedding = model(negative)

            total_embedding = torch.cat((anchor_embedding, positive_embedding, negative_embedding), dim=0)
            mu_batch = torch.mean(total_embedding, dim=0)

            # update
            self.update(mu_batch)

            dist = mu_batch - z
            loss = torch.norm(dist, p=2)

            return loss

    # class AdaReg(nn.Module):
    #     def __init__(self, **kwargs):
    #         super().__init__()
    #         # RandReg specific initialization here
    #         # For example:
            
    #         self.num_class = kwarges.get('num_class')
    #         self.mu = kwarges.get('mu', [nn.Parameter(torch.zeros(512, requires_grad=True)) for _ in range(num_class+1)])
    #         self.mu_bar = kwages.get('mu_bar', [torch.ones(512, requires_grad=False) for _ in range(num_class+1)])
    #         self.sigma = kwages.get('sigma', [torch.ones(512, requires_grad=False) for _ in range(num_class+1)])

    #     def ema(self, mu_bar, mu_hat, alpha=0.5):
    #         return alpha*mu_bar + (1 - alpha)*mu_hat

    #     def update(self, mu):
    #         with torch.no_grad():
    #             for cls in range(self.num_class):
    #                 self.mu = self.ema(self.mu, mu, 0.9)

    #     def forward(self, model, frozen_model, batch_label):
    #         # Implement the RandReg loss calculation here
            
    #         loss = 0.0
    #         for cls in range(self.num_class):
    #             # Class-wise random feature 
    #             with torch.no_grad():
    #                 z = torch.normal(mean=self.mu, std=self.sigma).unsqueeze(0).cuda()  #[1, 512]

    #             # batch / label
    #             batch, anchor_label, negative_label = batch_label
    #             anchor, positive, negative = batch  

    #             # Mask
    #             anchor_mask = (anchor_label == cls)
    #             negative_mask = (negative_label == cls)

    #             # Anchor Mask Check
    #             anchor_mask_sum = anchor_mask.sum(dim=0)
    #             negative_mask_sum = negative_mask.sum(dim=0)

    #             if anchor_mask_sum > 0:
    #                 anchor_embedding = model(anchor)
    #                 positive_embedding = model(positive)
                
    #                 anchor_selected_indices = torch.nonzero(anchor_mask, as_tuple=False)
    #                 anchor_selected_tensors = anchor_embedding[anchor_selected_indices[:, 0], :]
    #                 positive_selected_tensors = positive_embedding[anchor_selected_indices[:, 0], :]
    #                 anchor_dist = (anchor_selected_tensors - z)
    #                 positive_dist = (positive_selected_tensors - z)

    #                 loss += torch.dist(anchor_dist, p=2) + torch.dist(positive_dist, p=2)

    #             if negative_mask_sum > 0:
    #                 negative_embedding = model(negative)
    #                 negative_selected_indices = torch.nonzero(negative_mask, as_tuple=False)
    #                 negative_selected_tensors = negative_embedding[negative_selected_indices[:, 0], :]
    #                 negative_dist = (negative_selected_tensors - z)

    #                 loss += torch.dist(negative_dist, p=2)

    #         # update
    #         self.update(mu_batch)

            

    #         return loss

    