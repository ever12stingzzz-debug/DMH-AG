# model_three_modalities.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18
from transformers import AutoModel, AutoTokenizer
import clip


class ThreeModalResNetWithPMR(nn.Module):
    def __init__(self, 
                 bert_model_name='bert-base-uncased',
                 clip_model_name='ViT-B/32',
                 num_classes=2, 
                 mri_feature_dim=512,
                 bert_feature_dim=512,
                 clip_feature_dim=512,
                 fused_feature_dim=512,
                 alpha=1.1, 
                 mu=0.01, 
                 epsilon=0.9, 
                 E_r=10, 
                 temperature=1.0,
                 max_seq_length=512):
        """
        PMR增强的三模态模型：
        1. MRI图像 (3D ResNet)
        2. 长文本报告 (BERT)
        3. 短文本MMSE分数描述 (CLIP)
        
        Args:
            bert_model_name: BERT模型名称
            clip_model_name: CLIP模型名称
            num_classes: 分类数量
            mri_feature_dim: MRI特征维度
            bert_feature_dim: BERT特征维度
            clip_feature_dim: CLIP特征维度
            fused_feature_dim: 融合特征维度
            alpha: PCE损失权重
            mu: PER损失权重
            epsilon: 原型更新动量
            E_r: PER应用的最大epoch数
            temperature: 温度参数
            max_seq_length: BERT最大序列长度
        """
        super(ThreeModalResNetWithPMR, self).__init__()
        
        # ========== 1. MRI Backbone ==========
        self.backbone = r3d_18(weights=None)
        self.backbone.stem[0] = nn.Conv3d(1, 64, kernel_size=(3,7,7), 
                                         stride=(1,2,2), padding=(1,3,3), bias=False)
        self.backbone.fc = nn.Identity()
        
        # MRI特征投影层
        self.mri_projection = nn.Linear(512, mri_feature_dim)
        
        # ========== 2. BERT Text Encoder ==========
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.max_seq_length = max_seq_length
        
        # 获取BERT输出维度
        bert_config = self.bert.config
        bert_hidden_size = bert_config.hidden_size
        
        # BERT特征适配层
        self.bert_adapter = nn.Sequential(
            nn.Linear(bert_hidden_size, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(768, bert_feature_dim)
        )
        
        # ========== 3. CLIP Text Encoder ==========
        self.clip_model, _ = clip.load(clip_model_name)
        self.clip_model.eval()  # 冻结CLIP参数
        
        # CLIP特征适配层
        clip_hidden_size = self.clip_model.text_projection.shape[1]  # CLIP文本特征维度
        self.clip_adapter = nn.Sequential(
            nn.Linear(clip_hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, clip_feature_dim)
        )
        
        # ========== 4. Feature Fusion ==========
        self.mri_feature_dim = mri_feature_dim
        self.bert_feature_dim = bert_feature_dim
        self.clip_feature_dim = clip_feature_dim
        self.fused_feature_dim = fused_feature_dim
        
        # 模态融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(mri_feature_dim + bert_feature_dim + clip_feature_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, fused_feature_dim),
            nn.LayerNorm(fused_feature_dim),
            nn.GELU()
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fused_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # ========== 5. PMR Module ==========
        self.pmr_module = PMRModuleThreeModal(
            mri_feature_dim=mri_feature_dim,
            bert_feature_dim=bert_feature_dim,
            clip_feature_dim=clip_feature_dim,
            num_classes=num_classes,
            alpha=alpha,
            mu=mu,
            epsilon=epsilon,
            E_r=E_r,
            temperature=temperature
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def encode_mri(self, x):
        """编码MRI图像"""
        mri_features = self.backbone(x)  # (B, 512)
        mri_features = self.mri_projection(mri_features)  # (B, mri_feature_dim)
        mri_features = F.normalize(mri_features, p=2, dim=1)  # 归一化
        return mri_features
    
    def encode_bert_text(self, input_ids, attention_mask):
        """编码BERT文本"""
        # 获取BERT输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 使用[CLS] token的特征
        cls_features = outputs.last_hidden_state[:, 0, :]  # (B, hidden_size)
        # 通过适配层
        bert_features = self.bert_adapter(cls_features)
        bert_features = F.normalize(bert_features, p=2, dim=1)  # 归一化
        return bert_features
    
    def encode_clip_text(self, clip_input_ids):
        """编码CLIP文本"""
        with torch.no_grad():
            clip_features = self.clip_model.encode_text(clip_input_ids).float()
        # 通过适配层
        clip_features = self.clip_adapter(clip_features)
        clip_features = F.normalize(clip_features, p=2, dim=1)  # 归一化
        return clip_features
    
    def forward(self, mri_data, bert_input_ids=None, bert_attention_mask=None, 
                clip_input_ids=None, labels=None):
        """
        前向传播
        
        Args:
            mri_data: MRI图像 (B, 1, D, H, W)
            bert_input_ids: BERT token IDs
            bert_attention_mask: BERT注意力掩码
            clip_input_ids: CLIP token IDs
            labels: 标签
            
        Returns:
            if labels provided: (logits, mri_features, bert_features, clip_features)
            else: logits
        """
        B = mri_data.shape[0]
        
        # 1. 编码MRI
        mri_features = self.encode_mri(mri_data)  # (B, mri_feature_dim)
        
        # 2. 编码BERT文本
        if bert_input_ids is not None and bert_attention_mask is not None:
            bert_features = self.encode_bert_text(bert_input_ids, bert_attention_mask)  # (B, bert_feature_dim)
        else:
            # 如果没有BERT输入，创建零向量
            bert_features = torch.zeros(B, self.bert_feature_dim, device=mri_data.device)
        
        # 3. 编码CLIP文本
        if clip_input_ids is not None:
            clip_features = self.encode_clip_text(clip_input_ids)  # (B, clip_feature_dim)
        else:
            # 如果没有CLIP输入，创建零向量
            clip_features = torch.zeros(B, self.clip_feature_dim, device=mri_data.device)
        
        # 4. 特征融合
        fused_features = torch.cat([mri_features, bert_features, clip_features], dim=1)  # (B, sum_dim)
        fused_features = self.fusion_layer(fused_features)  # (B, fused_feature_dim)
        
        # 5. 分类
        logits = self.classifier(fused_features)
        
        # 返回特征用于PMR计算
        if labels is not None:
            return logits, mri_features, bert_features, clip_features
        return logits
    
    def update_prototypes(self, mri_features, bert_features, clip_features, labels):
        """更新三个模态的原型"""
        self.pmr_module.update_prototypes(mri_features, bert_features, clip_features, labels)
    
    def get_pmr_loss(self, mri_features, bert_features, clip_features, labels, epoch):
        """计算PMR损失"""
        return self.pmr_module.get_pmr_loss(mri_features, bert_features, clip_features, labels, epoch)
    
    def get_prototype_info(self):
        """获取原型信息"""
        return self.pmr_module.get_prototype_info()
    
    def reset_prototypes(self):
        """重置原型"""
        self.pmr_module.reset_prototypes()


class PMRModuleThreeModal(nn.Module):
    def __init__(self, 
                 mri_feature_dim=512,
                 bert_feature_dim=512,
                 clip_feature_dim=512,
                 num_classes=2,
                 alpha=1.0,
                 mu=0.01,
                 epsilon=0.9,
                 E_r=10,
                 temperature=1.0):
        super().__init__()
        self.alpha = alpha
        self.mu = mu
        self.epsilon = epsilon
        self.E_r = E_r
        self.num_classes = num_classes
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # 三个模态的原型存储
        self.register_buffer('mri_prototypes', torch.zeros(num_classes, mri_feature_dim))
        self.register_buffer('bert_prototypes', torch.zeros(num_classes, bert_feature_dim))
        self.register_buffer('clip_prototypes', torch.zeros(num_classes, clip_feature_dim))
        self.register_buffer('proto_counts', torch.zeros(num_classes))
        
    def compute_prototype_probabilities(self, features, prototypes):
        """计算基于原型的概率分布"""
        features = F.normalize(features, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        dists = torch.cdist(features, prototypes, p=2)
        logits = -dists / self.temperature
        probs = F.softmax(logits, dim=1)
        return probs
    
    def compute_pce_loss(self, features, prototypes, labels):
        """计算原型交叉熵损失"""
        probs = self.compute_prototype_probabilities(features, prototypes)
        batch_indices = torch.arange(labels.size(0)).to(labels.device)
        true_probs = probs[batch_indices, labels]
        pce_loss = -torch.log(true_probs + 1e-8).mean()
        return pce_loss
    
    def compute_per(self, features, prototypes, labels):
        """计算原型熵正则化"""
        with torch.no_grad():
            probs = self.compute_prototype_probabilities(features, prototypes)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            per_loss = -entropy.mean()
        return per_loss
    
    def update_prototypes(self, mri_features, bert_features, clip_features, labels):
        """更新三个模态的原型"""
        mri_features = F.normalize(mri_features, p=2, dim=1)
        bert_features = F.normalize(bert_features, p=2, dim=1)
        clip_features = F.normalize(clip_features, p=2, dim=1)
        
        for k in range(self.num_classes):
            mask = labels == k
            if mask.sum() == 0:
                continue
                
            # 当前batch的特征均值
            mri_k = mri_features[mask].mean(dim=0)
            bert_k = bert_features[mask].mean(dim=0)
            clip_k = clip_features[mask].mean(dim=0)
            count_k = mask.sum().item()
            
            # 动量更新
            if self.proto_counts[k] > 0:
                self.mri_prototypes[k] = (
                    self.epsilon * self.mri_prototypes[k] + 
                    (1 - self.epsilon) * mri_k
                )
                self.bert_prototypes[k] = (
                    self.epsilon * self.bert_prototypes[k] + 
                    (1 - self.epsilon) * bert_k
                )
                self.clip_prototypes[k] = (
                    self.epsilon * self.clip_prototypes[k] + 
                    (1 - self.epsilon) * clip_k
                )
            else:
                # 第一次更新
                self.mri_prototypes[k] = mri_k
                self.bert_prototypes[k] = bert_k
                self.clip_prototypes[k] = clip_k
                
            self.proto_counts[k] += count_k
        
        # 更新后重新归一化原型
        self.mri_prototypes.data = F.normalize(self.mri_prototypes.data, p=2, dim=1)
        self.bert_prototypes.data = F.normalize(self.bert_prototypes.data, p=2, dim=1)
        self.clip_prototypes.data = F.normalize(self.clip_prototypes.data, p=2, dim=1)
    
    def compute_imbalance_ratio_matrix(self, mri_features, bert_features, clip_features, labels):
        """计算模态间的不平衡比率矩阵"""
        with torch.no_grad():
            # 计算每个模态的概率分布
            mri_probs = self.compute_prototype_probabilities(mri_features, self.mri_prototypes)
            bert_probs = self.compute_prototype_probabilities(bert_features, self.bert_prototypes)
            clip_probs = self.compute_prototype_probabilities(clip_features, self.clip_prototypes)
            
            # 获取真实类别对应的概率
            batch_size = labels.size(0)
            batch_indices = torch.arange(batch_size).to(labels.device)
            
            mri_true_probs = mri_probs[batch_indices, labels]  # (B,)
            bert_true_probs = bert_probs[batch_indices, labels]  # (B,)
            clip_true_probs = clip_probs[batch_indices, labels]  # (B,)
            
            # 计算模态间的不平衡比率
            sum_mri = mri_true_probs.sum()
            sum_bert = bert_true_probs.sum()
            sum_clip = clip_true_probs.sum()
            
            # 三个模态两两之间的比率
            rho_mri_bert = sum_mri / (sum_bert + 1e-8)
            rho_mri_clip = sum_mri / (sum_clip + 1e-8)
            rho_bert_clip = sum_bert / (sum_clip + 1e-8)
            
        return {
            'mri_bert': rho_mri_bert,
            'mri_clip': rho_mri_clip,
            'bert_clip': rho_bert_clip
        }
    
    def get_pmr_loss(self, mri_features, bert_features, clip_features, labels, epoch):
        """计算三个模态的PMR损失"""
        # 确保特征已归一化
        mri_features = F.normalize(mri_features, p=2, dim=1)
        bert_features = F.normalize(bert_features, p=2, dim=1)
        clip_features = F.normalize(clip_features, p=2, dim=1)
        
        # 计算PCE损失
        pce_mri = self.compute_pce_loss(mri_features, self.mri_prototypes, labels)
        pce_bert = self.compute_pce_loss(bert_features, self.bert_prototypes, labels)
        pce_clip = self.compute_pce_loss(clip_features, self.clip_prototypes, labels)
        
        # 计算不平衡比率矩阵
        rho_matrix = self.compute_imbalance_ratio_matrix(mri_features, bert_features, clip_features, labels)
        
        # 根据不平衡比率计算权重
        # 这里使用简单的启发式方法：对最弱的模态给予最大权重
        rho_values = torch.tensor([
            rho_matrix['mri_clip'],
            rho_matrix['bert_clip'],
            1
        ]).to(mri_features.device)
        
        # 归一化权重
        weights = F.softmax(1.0 / (rho_values + 1e-8), dim=0)
        
        # 加权PCE损失
        pce_loss = self.alpha * (
            weights[0] * pce_mri +   # MRI权重
            weights[1] * pce_bert +  # BERT权重
            weights[2] * pce_clip    # CLIP权重
        )
        
        # PER损失（只在早期epoch应用）
        if epoch < self.E_r:
            per_mri = self.compute_per(mri_features, self.mri_prototypes, labels)
            per_bert = self.compute_per(bert_features, self.bert_prototypes, labels)
            per_clip = self.compute_per(clip_features, self.clip_prototypes, labels)
            
            # PER使用与PCE相反的权重（鼓励不确定性）
            per_weights = F.softmax(rho_values, dim=0)
            per_loss = self.mu * (
                per_weights[0] * per_mri +
                per_weights[1] * per_bert +
                per_weights[2] * per_clip
            )
        else:
            per_loss = torch.tensor(0.0, device=pce_loss.device)
        
        # 收集不平衡比率用于监控
        imbalance_info = {
            'rho_mri_bert': rho_matrix['mri_bert'].item(),
            'rho_mri_clip': rho_matrix['mri_clip'].item(),
            'rho_bert_clip': rho_matrix['bert_clip'].item()
        }
        
        return pce_loss, per_loss, imbalance_info
    
    def get_prototype_info(self):
        """获取原型信息"""
        info = {
            'mri_prototypes_norm': torch.norm(self.mri_prototypes, dim=1).mean().item(),
            'bert_prototypes_norm': torch.norm(self.bert_prototypes, dim=1).mean().item(),
            'clip_prototypes_norm': torch.norm(self.clip_prototypes, dim=1).mean().item(),
            'proto_counts': self.proto_counts.cpu().numpy()
        }
        return info
    
    def reset_prototypes(self):
        """重置原型"""
        self.mri_prototypes.zero_()
        self.bert_prototypes.zero_()
        self.clip_prototypes.zero_()
        self.proto_counts.zero_()


# 使用示例
def create_three_modal_model(bert_model_name='bert-base-uncased',
                            clip_model_name='ViT-B/32',
                            num_classes=2):
    """创建三模态模型"""
    model = ThreeModalResNetWithPMR(
        bert_model_name=bert_model_name,
        clip_model_name=clip_model_name,
        num_classes=num_classes,
        mri_feature_dim=512,
        bert_feature_dim=512,
        clip_feature_dim=512,
        fused_feature_dim=512
    )
    return model


# 训练循环示例
def train_step_three_modal(model, batch, optimizer, criterion, epoch, device):
    """训练步骤示例"""
    model.train()
    
    # 将数据移到设备
    mri_data = batch['mri'].to(device)
    bert_input_ids = batch['bert_input_ids'].to(device)
    bert_attention_mask = batch['bert_attention_mask'].to(device)
    clip_input_ids = batch['clip_input_ids'].to(device)
    labels = batch['label'].to(device)
    
    # 前向传播
    logits, mri_features, bert_features, clip_features = model(
        mri_data=mri_data,
        bert_input_ids=bert_input_ids,
        bert_attention_mask=bert_attention_mask,
        clip_input_ids=clip_input_ids,
        labels=labels
    )
    
    # 计算分类损失
    cls_loss = criterion(logits, labels)
    
    # 更新原型
    model.update_prototypes(mri_features, bert_features, clip_features, labels)
    
    # 计算PMR损失
    pce_loss, per_loss, imbalance_info = model.get_pmr_loss(
        mri_features, bert_features, clip_features, labels, epoch
    )
    
    # 总损失
    total_loss = cls_loss + pce_loss + per_loss
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'cls_loss': cls_loss.item(),
        'pce_loss': pce_loss.item() if torch.is_tensor(pce_loss) else pce_loss,
        'per_loss': per_loss.item() if torch.is_tensor(per_loss) else per_loss,
        'imbalance_info': imbalance_info
    }


import torch

if __name__ == "__main__":
    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 创建模型并放到device
    model = create_three_modal_model(
        bert_model_name='bert-base-uncased',
        clip_model_name='ViT-B/32',
        num_classes=2
    )
    model = model.to(device)
    model.eval()  # 如果只是测试前向的话

    # 示例输入（注意都要 to(device)）
    batch_size = 2
    mri_data = torch.randn(batch_size, 1, 64, 128, 128, device=device)  # 直接在device上创建
    bert_input_ids = torch.randint(0, 1000, (batch_size, 512), device=device)  # 在device上
    # BERT 的 attention_mask 一般是 long 类型
    bert_attention_mask = torch.ones(batch_size, 512, dtype=torch.long, device=device)
    # 你之前用的 clip_input_ids (若是 token ids)，也放到 device
    _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)  # device 不重要，这里只是为了确保 tokenizer 可用
    dummy_texts = ["The patient has MMSE score: 28."] * batch_size
    clip_input_ids = clip.tokenize(dummy_texts).to(device)  # 已经是 long dtype
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)

    # 前向传播（确保模型内部没有创建新的 CPU tensor）
    with torch.no_grad():
        logits, mri_features, bert_features, clip_features = model(
            mri_data=mri_data,
            bert_input_ids=bert_input_ids,
            bert_attention_mask=bert_attention_mask,
            clip_input_ids=clip_input_ids,
            labels=labels
        )

    print(f"Logits device: {logits.device}, shape: {logits.shape}")
    print(f"MRI features device: {mri_features.device}, shape: {mri_features.shape}")
    print(f"BERT features device: {bert_features.device}, shape: {bert_features.shape}")
    print(f"CLIP features device: {clip_features.device}, shape: {clip_features.shape}")

    # 更新原型（会在 model 内部做 tensor 运算，确保实现中使用了同一device）
    model.update_prototypes(mri_features, bert_features, clip_features, labels)

    # 计算PMR损失
    pce_loss, per_loss, imbalance_info = model.get_pmr_loss(
        mri_features, bert_features, clip_features, labels, epoch=5
    )

    print(f"PCE Loss: {pce_loss}")
    print(f"PER Loss: {per_loss}")
    print(f"Imbalance Info: {imbalance_info}")
