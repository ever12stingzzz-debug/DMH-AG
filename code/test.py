import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import json
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from torchvision.models.video import r3d_18
from tqdm import tqdm
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import clip
import pandas as pd
import time
import logging
import random
import re
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 导入新的三模态模型和数据集
from ResNet3DWithPMRAALMMSE import create_three_modal_model, train_step_three_modal
from dataset import MultiModalMRIDataset, collate_fn, AALAttentionProcessor, generate_mmse_text

log_dir = "/public/home/wangdongjing/zby/zhubinyu/AD/log/resnet+MMSE+AAL+PMR"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(log_dir, 'three_modal2.log'),
    filemode='a',
)
main_logger = logging.getLogger('main')

def train_with_pmr_three_modal(model, dataloader, criterion, optimizer, device, epoch):
    """Training function with PMR integration - 三模态版本"""
    model.train()
    train_loss, correct, total = 0, 0, 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    # PMR损失统计
    total_pce_loss = 0.0
    total_per_loss = 0.0
    total_cls_loss = 0.0
    
    # 不平衡比率统计
    imbalance_ratios = {
        'mri_bert': [],
        'mri_clip': [],
        'bert_clip': []
    }
    
    # 获取初始原型信息
    prototype_info_initial = model.get_prototype_info()
    main_logger.info(f"Epoch {epoch} - Initial prototypes: "
                    f"MRI: {prototype_info_initial['mri_prototypes_norm']:.4f}, "
                    f"BERT: {prototype_info_initial['bert_prototypes_norm']:.4f}, "
                    f"CLIP: {prototype_info_initial['clip_prototypes_norm']:.4f}")

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch}")):
        # 跳过空batch
        if not batch:
            continue
            
        # 获取三个模态的数据
        imgs = batch['mri'].to(device)
        bert_input_ids = batch['bert_input_ids'].to(device)
        bert_attention_mask = batch['bert_attention_mask'].to(device)
        clip_input_ids = batch['clip_input_ids'].to(device)
        labels = batch['label'].to(device)
        
        # 前向传播获取三个模态的特征
        outputs, mri_features, bert_features, clip_features = model(
            mri_data=imgs,
            bert_input_ids=bert_input_ids,
            bert_attention_mask=bert_attention_mask,
            clip_input_ids=clip_input_ids,
            labels=labels
        )
        
        # 标准交叉熵损失
        cls_loss = criterion(outputs, labels)
        
        # PMR损失（三模态版本）
        pce_loss, per_loss, imbalance_info = model.get_pmr_loss(
            mri_features, bert_features, clip_features, labels, epoch
        )
        
        # 总损失 = 分类损失 + PCE + PER
        total_batch_loss = cls_loss + pce_loss + per_loss
        # total_batch_loss = cls_loss

        optimizer.zero_grad()
        total_batch_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 更新三个模态的原型
        model.update_prototypes(mri_features.detach(), bert_features.detach(), clip_features.detach(), labels)

        # 统计指标
        batch_size = imgs.size(0)
        train_loss += total_batch_loss.item() * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_pce_loss += pce_loss.item() * batch_size
        total_per_loss += per_loss.item() * batch_size
        
        # 记录不平衡比率
        imbalance_ratios['mri_bert'].append(imbalance_info['rho_mri_bert'])
        imbalance_ratios['mri_clip'].append(imbalance_info['rho_mri_clip'])
        imbalance_ratios['bert_clip'].append(imbalance_info['rho_bert_clip'])
        
        # 预测和准确率
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)
        
        correct += (preds == labels).sum().item()
        total += batch_size
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        
        # 每10个batch打印一次调试信息
        if batch_idx % 10 == 0:
            main_logger.debug(f"Epoch {epoch} Batch {batch_idx}: "
                            f"CLS: {cls_loss.item():.4f}, "
                            f"PCE: {pce_loss.item():.4f}, "
                            f"PER: {per_loss.item():.4f}, "
                            f"ρ_mri_bert: {imbalance_info['rho_mri_bert']:.4f}")

    # 计算平均指标
    if total > 0:
        epoch_loss = train_loss / total
        avg_cls_loss = total_cls_loss / total
        avg_pce_loss = total_pce_loss / total
        avg_per_loss = total_per_loss / total
        epoch_acc = correct / total
    else:
        epoch_loss = avg_cls_loss = avg_pce_loss = avg_per_loss = epoch_acc = 0.0
    
    # 计算平均不平衡比率
    avg_imbalance = {
        'mri_bert': np.mean(imbalance_ratios['mri_bert']) if imbalance_ratios['mri_bert'] else 1.0,
        'mri_clip': np.mean(imbalance_ratios['mri_clip']) if imbalance_ratios['mri_clip'] else 1.0,
        'bert_clip': np.mean(imbalance_ratios['bert_clip']) if imbalance_ratios['bert_clip'] else 1.0
    }
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    if len(all_labels) > 0:
        epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
        epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
        epoch_roc_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    else:
        epoch_precision = epoch_recall = epoch_f1 = epoch_roc_auc = 0.0
    
    # 记录最终原型信息
    prototype_info_final = model.get_prototype_info()
    main_logger.info(f"Epoch {epoch} - Final prototypes: "
                    f"MRI: {prototype_info_final['mri_prototypes_norm']:.4f}, "
                    f"BERT: {prototype_info_final['bert_prototypes_norm']:.4f}, "
                    f"CLIP: {prototype_info_final['clip_prototypes_norm']:.4f}")

    return (epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_roc_auc, 
            avg_cls_loss, avg_pce_loss, avg_per_loss, avg_imbalance)

def evaluate_with_pmr(model, dataloader, criterion, device, mode="Validation"):
    """Evaluation function - 三模态版本"""
    model.eval()
    val_loss, correct, total = 0, 0, 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=mode):
            # 跳过空batch
            if not batch:
                continue
                
            # 获取三个模态的数据
            imgs = batch['mri'].to(device)
            bert_input_ids = batch['bert_input_ids'].to(device)
            bert_attention_mask = batch['bert_attention_mask'].to(device)
            clip_input_ids = batch['clip_input_ids'].to(device)
            labels = batch['label'].to(device)
            
            # 评估时不使用PMR
            outputs = model(
                mri_data=imgs,
                bert_input_ids=bert_input_ids,
                bert_attention_mask=bert_attention_mask,
                clip_input_ids=clip_input_ids
            )
            
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算指标
    if total > 0:
        epoch_loss = val_loss / total
        epoch_acc = correct / total
    else:
        epoch_loss = epoch_acc = 0.0
        
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    if len(all_labels) > 0:
        epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
        epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
        epoch_roc_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
    else:
        epoch_precision = epoch_recall = epoch_f1 = epoch_roc_auc = 0.0

    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_roc_auc

def initialize_prototypes(model, train_loader, device):
    """初始化三个模态的原型"""
    main_logger.info("Initializing prototypes with training data...")
    model.eval()
    
    with torch.no_grad():
        # 收集特征和标签
        mri_features_list = []
        bert_features_list = []
        clip_features_list = []
        labels_list = []
        
        # 收集3-5个batch的数据
        for i, batch in enumerate(train_loader):
            if i >= 8 or not batch:  # 最多5个batch
                break
                
            imgs = batch['mri'].to(device)
            bert_input_ids = batch['bert_input_ids'].to(device)
            bert_attention_mask = batch['bert_attention_mask'].to(device)
            clip_input_ids = batch['clip_input_ids'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播获取特征
            _, mri_features, bert_features, clip_features = model(
                mri_data=imgs,
                bert_input_ids=bert_input_ids,
                bert_attention_mask=bert_attention_mask,
                clip_input_ids=clip_input_ids,
                labels=labels
            )
            
            mri_features_list.append(mri_features)
            bert_features_list.append(bert_features)
            clip_features_list.append(clip_features)
            labels_list.append(labels)
        
        if mri_features_list:  # 确保有数据
            # 合并所有batch的特征
            all_mri_features = torch.cat(mri_features_list, dim=0)
            all_bert_features = torch.cat(bert_features_list, dim=0)
            all_clip_features = torch.cat(clip_features_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            
            # 手动初始化原型
            pmr_module = model.pmr_module
            for k in range(pmr_module.num_classes):
                mask = all_labels == k
                if mask.sum() > 0:
                    pmr_module.mri_prototypes[k] = all_mri_features[mask].mean(dim=0)
                    pmr_module.bert_prototypes[k] = all_bert_features[mask].mean(dim=0)
                    pmr_module.clip_prototypes[k] = all_clip_features[mask].mean(dim=0)
                    pmr_module.proto_counts[k] = mask.sum().item()
                    main_logger.info(f"Class {k}: {mask.sum().item()} samples")
                else:
                    # 如果某个类别没有样本，使用随机初始化
                    pmr_module.mri_prototypes[k] = torch.randn_like(pmr_module.mri_prototypes[k])
                    pmr_module.bert_prototypes[k] = torch.randn_like(pmr_module.bert_prototypes[k])
                    pmr_module.clip_prototypes[k] = torch.randn_like(pmr_module.clip_prototypes[k])
                    pmr_module.proto_counts[k] = 0
                    main_logger.warning(f"Class {k}: No samples found, using random initialization")
            
            # 归一化原型
            pmr_module.mri_prototypes.data = F.normalize(pmr_module.mri_prototypes.data, p=2, dim=1)
            pmr_module.bert_prototypes.data = F.normalize(pmr_module.bert_prototypes.data, p=2, dim=1)
            pmr_module.clip_prototypes.data = F.normalize(pmr_module.clip_prototypes.data, p=2, dim=1)
        else:
            main_logger.warning("No data available for prototype initialization")
    
    # 检查初始化结果
    init_info = model.get_prototype_info()
    main_logger.info(f"After initialization - "
                    f"MRI norm: {init_info['mri_prototypes_norm']:.4f}, "
                    f"BERT norm: {init_info['bert_prototypes_norm']:.4f}, "
                    f"CLIP norm: {init_info['clip_prototypes_norm']:.4f}")
    main_logger.info(f"Prototype counts: {init_info['proto_counts']}")

def evaluate_with_pmr(model, dataloader, criterion, device, mode="Validation"):
    """Evaluation function - 三模态版本，包含混淆矩阵"""
    model.eval()
    val_loss, correct, total = 0, 0, 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=mode):
            # 跳过空batch
            if not batch:
                continue
                
            # 获取三个模态的数据
            imgs = batch['mri'].to(device)
            bert_input_ids = batch['bert_input_ids'].to(device)
            bert_attention_mask = batch['bert_attention_mask'].to(device)
            clip_input_ids = batch['clip_input_ids'].to(device)
            labels = batch['label'].to(device)
            
            # 评估时不使用PMR
            outputs = model(
                mri_data=imgs,
                bert_input_ids=bert_input_ids,
                bert_attention_mask=bert_attention_mask,
                clip_input_ids=clip_input_ids
            )
            
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 计算指标
    if total > 0:
        epoch_loss = val_loss / total
        epoch_acc = correct / total
    else:
        epoch_loss = epoch_acc = 0.0
        
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算混淆矩阵
    if len(all_labels) > 0:
        cm = confusion_matrix(all_labels, all_preds)
        
        # 计算各项指标
        epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
        epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
        epoch_roc_auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5
        
        # 计算混淆矩阵的详细指标
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        cm_metrics = {
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,  # 召回率
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        }
    else:
        cm = np.array([[0, 0], [0, 0]])
        cm_metrics = {}
        epoch_precision = epoch_recall = epoch_f1 = epoch_roc_auc = 0.0

    return epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1, epoch_roc_auc, cm, cm_metrics

def plot_confusion_matrix(cm, class_names, save_path=None, title="Confusion Matrix"):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 需要两个tokenizer：BERT和CLIP
    from transformers import AutoTokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # 加载CLIP模型和tokenizer
    clip_model_name = "ViT-B/32"
    clip_model, clip_preprocess = clip.load(clip_model_name)
    clip_model.eval()  # 注意：CLIP在训练时是冻结的
    
    main_logger.info(f"Using device: {device}")

    # 设置文本文件目录路径
    text_dir = "/public/home/wangdongjing/zby/zhubinyu/AAL/results/2-class-textv1"
    
    main_logger.info("Loading datasets...")
    
    # 创建AAL注意力处理器（可选）
    aal_processor = AALAttentionProcessor(
        aal3_dir="/public/home/wangdongjing/zby/data/AAL3", 
        cache_dir="/tmp/aal_cache"
    )
    
    # 创建三模态数据集
    train_dataset = MultiModalMRIDataset(
        root_dir="/public/home/wangdongjing/zby/data/ADNI_2",
        text_dir=text_dir,
        csv_path="/public/home/wangdongjing/zby/data/ADNI_2_MMSE.csv",
        bert_tokenizer=bert_tokenizer,
        clip_model_name=clip_model_name,
        target_shape=(128, 128, 128),
        split="train",
        train_ratio=0.7,
        val_ratio=0.2,
        max_seq_length=512,
        apply_augmentation=True,
        use_attention=False,
        aal_processor=aal_processor,
        attention_strength=0.5,
        rng_seed=42
    )
    
    val_dataset = MultiModalMRIDataset(
        root_dir="/public/home/wangdongjing/zby/data/ADNI_2",
        text_dir=text_dir,
        csv_path="/public/home/wangdongjing/zby/data/ADNI_2_MMSE.csv",
        bert_tokenizer=bert_tokenizer,
        clip_model_name=clip_model_name,
        target_shape=(128, 128, 128),
        split="val",
        train_ratio=0.7,
        val_ratio=0.2,
        max_seq_length=512,
        apply_augmentation=False,
        use_attention=False,
        aal_processor=aal_processor,
        attention_strength=0.5,
        rng_seed=42
    )
    
    test_dataset = MultiModalMRIDataset(
        root_dir="/public/home/wangdongjing/zby/data/ADNI_2",
        text_dir=text_dir,
        csv_path="/public/home/wangdongjing/zby/data/ADNI_2_MMSE.csv",
        bert_tokenizer=bert_tokenizer,
        clip_model_name=clip_model_name,
        target_shape=(128, 128, 128),
        split="test",
        train_ratio=0.7,
        val_ratio=0.2,
        max_seq_length=512,
        apply_augmentation=False,
        use_attention=False,
        aal_processor=aal_processor,
        attention_strength=0.5,
        rng_seed=42
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=3, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=3, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=3, pin_memory=True, collate_fn=collate_fn)

    main_logger.info("Creating Three-Modal PMR model...")
    
    # 创建三模态PMR模型
    model = create_three_modal_model(
        bert_model_name='bert-base-uncased',
        clip_model_name=clip_model_name,
        num_classes=2
    )

    main_logger.info("Starting Three-Modal PMR training")
    model.to(device)

    # === 关键修复：在训练前初始化原型 ===
    initialize_prototypes(model, train_loader, device)
    # === 初始化完成 ===

    criterion = nn.CrossEntropyLoss()
    
    # 优化器配置 - 对不同部分使用不同学习率
    # optimizer = optim.AdamW([
    #     # MRI组件 - 正常学习率
    #     {'params': model.backbone.parameters(), 'lr': 1e-4},
    #     {'params': model.mri_projection.parameters(), 'lr': 1e-4},
        
    #     # BERT组件 - 较低学习率（通常建议）
    #     {'params': model.bert.parameters(), 'lr': 2e-5},
    #     {'params': model.bert_adapter.parameters(), 'lr': 5e-4},
        
    #     # CLIP适配器（CLIP模型本身是冻结的）
    #     {'params': model.clip_adapter.parameters(), 'lr': 1e-4},
        
    #     # 融合和分类组件
    #     {'params': model.fusion_layer.parameters(), 'lr': 1e-4},
    #     {'params': model.classifier.parameters(), 'lr': 1e-4},
        
    #     # PMR模块参数（如果可学习）
    #     {'params': model.pmr_module.parameters(), 'lr': 1e-4}
    # ], weight_decay=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    num_epochs = 100
    best_val_acc = 0.0
    best_val_f1 = 0.0

    # 训练历史记录
    train_history = []
    val_history = []

    # for epoch in range(num_epochs):
    #     # 使用 PMR 训练（三模态版本）
    #     (train_loss, train_acc, train_precision, train_recall, train_f1, 
    #      train_roc_auc, train_cls, train_pce, train_per, train_imbalance) = train_with_pmr_three_modal(
    #         model, train_loader, criterion, optimizer, device, epoch
    #     )
        
    #     # 验证（不使用 PMR）
    #     val_loss, val_acc, val_precision, val_recall, val_f1, val_roc_auc = evaluate_with_pmr(
    #         model, val_loader, criterion, device
    #     )
        
    #     # 学习率调度
    #     scheduler.step()
    #     current_lr = scheduler.get_last_lr()[0]

    #     # 记录历史
    #     train_history.append({
    #         'epoch': epoch, 
    #         'loss': train_loss, 
    #         'acc': train_acc, 
    #         'f1': train_f1, 
    #         'auc': train_roc_auc,
    #         'cls_loss': train_cls, 
    #         'pce_loss': train_pce, 
    #         'per_loss': train_per, 
    #         'imbalance': train_imbalance
    #     })
    #     val_history.append({
    #         'epoch': epoch, 
    #         'loss': val_loss, 
    #         'acc': val_acc, 
    #         'f1': val_f1, 
    #         'auc': val_roc_auc
    #     })

    #     main_logger.info(f"Epoch {epoch+1}/{num_epochs} (LR: {current_lr:.6f})")
    #     main_logger.info(f"Train - Total Loss: {train_loss:.4f}, CLS: {train_cls:.4f}, "
    #                     f"PCE: {train_pce:.4f}, PER: {train_per:.4f}")
    #     main_logger.info(f"Train - Acc: {train_acc:.4f}, F1: {train_f1:.4f}, AUC: {train_roc_auc:.4f}")
    #     main_logger.info(f"Train - Imbalance: MRI/BERT: {train_imbalance['mri_bert']:.4f}, "
    #                     f"MRI/CLIP: {train_imbalance['mri_clip']:.4f}, "
    #                     f"BERT/CLIP: {train_imbalance['bert_clip']:.4f}")
    #     main_logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
    #                     f"F1: {val_f1:.4f}, AUC: {val_roc_auc:.4f}")

    #     # 保存最佳模型（基于准确率和F1分数）
    #     if val_acc > best_val_acc or (val_acc == best_val_acc and val_f1 > best_val_f1):
    #         best_val_acc = max(val_acc, best_val_acc)
    #         best_val_f1 = max(val_f1, best_val_f1)
    #         torch.save({
    #             'epoch': epoch,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'val_acc': val_acc,
    #             'val_f1': val_f1,
    #             'train_history': train_history,
    #             'val_history': val_history
    #         }, "/public/home/wangdongjing/zby/zhubinyu/AD/pt/resnet+MMSE+AAL+PMR2.pth")
    #         main_logger.info(f"New best model saved: Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
    
    # # 加载最佳模型进行测试
    # checkpoint = torch.load("/public/home/wangdongjing/zby/zhubinyu/AD/pt/resnet+MMSE+AAL+PMR2.pth")
    # model.load_state_dict(checkpoint['model_state_dict'])
    
    # test_loss, test_acc, test_precision, test_recall, test_f1, test_roc_auc = evaluate_with_pmr(
    #     model, test_loader, criterion, device, mode="Test"
    # )
    
    # main_logger.info("=== Final Test Results ===")
    # main_logger.info(f"Test Loss: {test_loss:.4f}")
    # main_logger.info(f"Test Acc: {test_acc:.4f}")
    # main_logger.info(f"Test Precision: {test_precision:.4f}")
    # main_logger.info(f"Test Recall: {test_recall:.4f}")
    # main_logger.info(f"Test F1: {test_f1:.4f}")
    # main_logger.info(f"Test ROC-AUC: {test_roc_auc:.4f}")
    
    # # 保存训练历史
    # history = {
    #     'train': train_history,
    #     'val': val_history,
    #     'test_results': {
    #         'loss': test_loss,
    #         'acc': test_acc,
    #         'precision': test_precision,
    #         'recall': test_recall,
    #         'f1': test_f1,
    #         'roc_auc': test_roc_auc
    #     }
    # }
    
    # with open("/public/home/wangdongjing/zby/zhubinyu/AD/log/resnet+MMSE+AAL+PMR/training_history_three_modal.json", 'w') as f:
    #     json.dump(history, f, indent=2)
    
    # main_logger.info("Training completed successfully!")

    # 加载最佳模型进行测试
    checkpoint = torch.load("/public/home/wangdongjing/zby/zhubinyu/AD/pt/resnet+MMSE+AAL+PMR2.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 修改测试调用，接收混淆矩阵
    test_loss, test_acc, test_precision, test_recall, test_f1, test_roc_auc, test_cm, test_cm_metrics = evaluate_with_pmr(
        model, test_loader, criterion, device, mode="Test"
    )
    
    main_logger.info("=== Final Test Results ===")
    main_logger.info(f"Test Loss: {test_loss:.4f}")
    main_logger.info(f"Test Acc: {test_acc:.4f}")
    main_logger.info(f"Test Precision: {test_precision:.4f}")
    main_logger.info(f"Test Recall: {test_recall:.4f}")
    main_logger.info(f"Test F1: {test_f1:.4f}")
    main_logger.info(f"Test ROC-AUC: {test_roc_auc:.4f}")
    
    # 打印混淆矩阵和详细指标
    main_logger.info("=== Confusion Matrix ===")
    main_logger.info(f"Confusion Matrix:\n{test_cm}")
    main_logger.info(f"True Negative: {test_cm_metrics.get('tn', 0)}")
    main_logger.info(f"False Positive: {test_cm_metrics.get('fp', 0)}")
    main_logger.info(f"False Negative: {test_cm_metrics.get('fn', 0)}")
    main_logger.info(f"True Positive: {test_cm_metrics.get('tp', 0)}")
    main_logger.info(f"Sensitivity (Recall): {test_cm_metrics.get('sensitivity', 0):.4f}")
    main_logger.info(f"Specificity: {test_cm_metrics.get('specificity', 0):.4f}")
    main_logger.info(f"Precision: {test_cm_metrics.get('precision', 0):.4f}")
    main_logger.info(f"F1-Score: {test_cm_metrics.get('f1_score', 0):.4f}")
    
    # 绘制并保存混淆矩阵
    class_names = ['Normal', 'AD']  # 根据你的类别调整
    cm_save_path = os.path.join(log_dir, "confusion_matrix_test.png")
    plot_confusion_matrix(test_cm, class_names, save_path=cm_save_path, title="Test Set Confusion Matrix")
    main_logger.info(f"Confusion matrix saved to: {cm_save_path}")
    
    # 保存训练历史
    history = {
        'train': train_history,
        'val': val_history,
        'test_results': {
            'loss': test_loss,
            'acc': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1,
            'roc_auc': test_roc_auc,
            'confusion_matrix': test_cm.tolist(),  # 将numpy数组转为list以便JSON序列化
            'cm_metrics': test_cm_metrics
        }
    }
    
    with open("/public/home/wangdongjing/zby/zhubinyu/AD/log/resnet+MMSE+AAL+PMR/training_history_three_modal.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    main_logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()