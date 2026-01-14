# dataset.py (optimized with MMSE text)
import os
import re
import logging
import random
from typing import Optional, Dict, Any, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage import rotate, zoom
from nilearn.image import resample_to_img
from transformers import AutoTokenizer
import clip

# -------------------------
# Config logging
# -------------------------
logger = logging.getLogger("MultiModalMRIDataset")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# -------------------------
# Utility transforms
# -------------------------
def random_rotation_3d(img: np.ndarray, max_angle: float = 15.0, rng: Optional[random.Random] = None) -> np.ndarray:
    """Apply random 3D rotation around z-axis (axes=(1,2))."""
    if rng is None:
        rng = random
    angle = rng.uniform(-max_angle, max_angle)
    return rotate_3d(img, angle)


def rotate_3d(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the 3D image around z-axis. Assumes img shape is (D, H, W)."""
    return rotate(img, angle, axes=(1, 2), reshape=False, mode='constant', cval=0.0)


def random_flip_3d(img: np.ndarray, prob: float = 0.5, rng: Optional[random.Random] = None) -> np.ndarray:
    """Randomly flip the 3D image along x and y axes with probability `prob`."""
    if rng is None:
        rng = random
    out = img
    if rng.random() < prob:
        out = np.flip(out, axis=0).copy()
    if rng.random() < prob:
        out = np.flip(out, axis=1).copy()
    return out


# -------------------------
# Report parsing (robust)
# -------------------------
def extract_report_text(txt_path: str) -> Optional[str]:
    """
    Extract all brain regions data from report text.
    """
    if not os.path.exists(txt_path):
        return None

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        logger.warning("Failed to open report %s: %s", txt_path, e)
        return None

    region_mapping = {
        '海马体': 'HPS',
        '杏仁核': 'Aga',
        '颞叶': 'TL',
        '额叶': 'FL',
        '顶叶': 'PL',
        '基底节': 'BG',
        '丘脑': 'Tls',
        '小脑': 'Cbm',
        '眶额皮质': 'OC',
        '前扣带回': 'AC',
        '脑干核团': 'BN'
    }

    simplified_lines = []
    for line in lines:
        line = line.strip()
        if '|' not in line:
            continue
        if not any(region in line for region in region_mapping.keys()):
            continue

        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) < 2:
            continue

        brain_region_cn = parts[0]
        if brain_region_cn not in region_mapping:
            continue
        brain_region_en = region_mapping[brain_region_cn]

        volume = None
        template_volume = None
        signal = None
        fill_ratio = None
        fill_status = None

        for part in parts[1:]:
            if 'AAL空间体积' in part and ':' in part:
                try:
                    volume = part.split(':', 1)[1].strip().split()[0]
                except Exception:
                    volume = None
            elif 'AAL模板体积' in part and ':' in part:
                try:
                    template_volume = part.split(':', 1)[1].strip().split()[0]
                except Exception:
                    template_volume = None
            elif 'MRI信号强度' in part and ':' in part:
                try:
                    signal = part.split(':', 1)[1].strip().split()[0]
                except Exception:
                    signal = None
            elif '填充信息' in part and ':' in part:
                try:
                    fill_info = part.split(':', 1)[1].strip()
                except Exception:
                    fill_info = ''
                m = re.search(r'([\d.]+)\s*%', fill_info)
                if m:
                    fill_ratio = m.group(1)
                if '严重填充不足' in fill_info:
                    fill_status = 's_underfill'
                elif '中度填充不足' in fill_info:
                    fill_status = 'm_underfill'
                elif '轻度填充不足' in fill_info:
                    fill_status = 'n_underfill'
                elif '正常填充' in fill_info:
                    fill_status = 'n_fill'
                elif '轻度过度填充' in fill_info:
                    fill_status = 'm_overfill'
                elif '过度填充' in fill_info:
                    fill_status = 'overfill'
                else:
                    if fill_status is None:
                        fill_status = 'unknown'

        if volume is not None and signal is not None and fill_ratio is not None:
            simplified_line = (
                f"{brain_region_en}:Vol{volume}mL_Temp{template_volume}mL_"
                f"Sig{signal}_Fill{fill_ratio}%_{fill_status}"
            )
            simplified_lines.append(simplified_line)

    if simplified_lines:
        return ' '.join(simplified_lines)
    else:
        return None


# -------------------------
# MMSE text generator for CLIP
# -------------------------
def generate_mmse_text(mm_score: float) -> str:
    """Generate text description for MMSE score."""
    mm_score = float(mm_score)
    if mm_score >= 27:
        return "The patient's MMSE score is {} indicating normal cognitive function.".format(mm_score)
    elif mm_score >= 24:
        return "The patient's MMSE score is {} suggesting mild cognitive impairment.".format(mm_score)
    elif mm_score >= 18:
        return "The patient's MMSE score is {} indicating moderate cognitive impairment.".format(mm_score)
    elif mm_score >= 10:
        return "The patient's MMSE score is {} suggesting severe cognitive impairment.".format(mm_score)
    else:
        return "The patient's MMSE score is {} indicating very severe cognitive impairment.".format(mm_score)


# -------------------------
# AAL Attention Processor (with cache)
# -------------------------
class AALAttentionProcessor:
    """AAL atlas based attention generator with optional caching (per-image)."""

    def __init__(self, aal3_dir: str, cache_dir: Optional[str] = None):
        self.aal3_dir = aal3_dir
        self.cache_dir = cache_dir
        self.aal3_img = None
        self.aal3_data = None
        self.region_definitions = {
            'HPS': [41, 42],
            'Aga': [45, 46],
            'TL': [41, 42, 43, 44, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94],
            'FL': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
            'PL': [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
            'BG': [75, 76, 77, 78, 79, 80],
            'Tls': list(range(121, 151)),
            'Cbm': list(range(95, 113)),
            'OC': [25, 26, 27, 28, 29, 30, 31, 32],
            'AC': [151, 152, 153, 154, 155, 156],
            'BN': [159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170]
        }
        self.load_aal3_atlas()
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def load_aal3_atlas(self) -> bool:
        """Load AAL3 atlas (NIfTI)."""
        try:
            aal3_path = os.path.join(self.aal3_dir, 'AAL3v1.nii')
            if not os.path.exists(aal3_path):
                aal3_path = os.path.join(self.aal3_dir, 'AAL3v1_1mm.nii')
            self.aal3_img = nib.load(aal3_path)
            self.aal3_data = self.aal3_img.get_fdata().astype(np.int16)
            logger.info("AAL3 atlas loaded, shape: %s", self.aal3_data.shape)
            return True
        except Exception as e:
            logger.error("Failed to load AAL3 atlas: %s", e)
            return False

    def create_region_weights_from_report(self, report_text: Optional[str]) -> Dict[str, float]:
        """Create region-wise weights from report text (default 1.0 if missing)."""
        region_weights = {}
        if report_text is None:
            for region in self.region_definitions.keys():
                region_weights[region] = 1.0
            return region_weights

        for region_abbr in self.region_definitions.keys():
            pattern = rf"{region_abbr}.*?Fill([\d.]+)%"
            match = re.search(pattern, report_text)
            if match:
                filling_ratio = float(match.group(1))
                if filling_ratio < 70:
                    weight = 2.0
                elif filling_ratio < 85:
                    weight = 1.5
                elif filling_ratio < 95:
                    weight = 1.2
                elif filling_ratio > 110:
                    weight = 0.8
                elif filling_ratio > 105:
                    weight = 0.9
                else:
                    weight = 1.0
                region_weights[region_abbr] = weight
            else:
                region_weights[region_abbr] = 1.0
        return region_weights

    def _make_aal_attention(self, region_weights: Dict[str, float]) -> np.ndarray:
        """Create an attention volume in AAL atlas space (same shape as aal3_data)."""
        aal_attention = np.ones_like(self.aal3_data, dtype=np.float32)
        aal_attention[:] = 1.0
        for region_abbr, labels in self.region_definitions.items():
            mask = np.isin(self.aal3_data, labels)
            weight = float(region_weights.get(region_abbr, 1.0))
            aal_attention[mask] = weight
        return aal_attention

    def _cache_path_for(self, img_id: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{img_id}_attention.npy")

    def generate_attention_map(self, brain_img: nib.Nifti1Image, region_weights: Dict[str, float],
                               img_id: Optional[str] = None, use_cache: bool = True) -> np.ndarray:
        """
        Generate attention map aligned with original brain_img space.
        """
        cache_path = self._cache_path_for(img_id) if img_id else None
        if cache_path and use_cache and os.path.exists(cache_path):
            try:
                att = np.load(cache_path)
                if att.ndim == 3:
                    return att
            except Exception as e:
                logger.debug("Failed loading attention cache %s: %s", cache_path, e)

        aal_attention = self._make_aal_attention(region_weights)

        try:
            temp_img = nib.Nifti1Image(aal_attention, self.aal3_img.affine)
            attention_original = resample_to_img(temp_img, brain_img)
            att_data = attention_original.get_fdata().astype(np.float32)
            if cache_path and use_cache:
                try:
                    np.save(cache_path, att_data)
                except Exception as e:
                    logger.debug("Could not save attention cache %s: %s", cache_path, e)
            return att_data
        except Exception as e:
            logger.warning("Attention generation failed, returning ones map: %s", e)
            return np.ones(brain_img.get_fdata().shape, dtype=np.float32)

    def apply_attention_weighting(self, brain_img: nib.Nifti1Image, attention_map: np.ndarray,
                                  attention_strength: float = 0.5, clip_max: float = 10.0) -> np.ndarray:
        """
        Apply attention weighting to brain image data.
        """
        brain_data = brain_img.get_fdata().astype(np.float32)
        if brain_data.shape != attention_map.shape:
            attention_map_resized = self._resize_to_match(attention_map, brain_data.shape)
        else:
            attention_map_resized = attention_map

        attention_map_resized = np.clip(attention_map_resized, -clip_max, clip_max)
        weighted_brain = brain_data * (1.0 + attention_strength * (np.exp(np.clip(attention_map_resized - 1.0, -10.0, 10.0))))
        weighted_brain = np.nan_to_num(weighted_brain, nan=0.0, posinf=0.0, neginf=0.0)
        return weighted_brain.astype(np.float32)

    def _resize_to_match(self, data: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Resize 3D data to target_shape using scipy zoom (order=1)."""
        factors = [float(target_shape[i]) / float(data.shape[i]) for i in range(3)]
        resized = zoom(data, factors, order=1)
        return resized.astype(np.float32)


# -------------------------
# Dataset with Three Modalities
# -------------------------
class MultiModalMRIDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 text_dir: str,
                 csv_path: str,
                 bert_tokenizer: AutoTokenizer,
                 clip_model_name: str = "ViT-B/32",
                 target_shape: tuple = (128, 128, 128),
                 split: str = "train",
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.2,
                 max_seq_length: int = 512,
                 apply_augmentation: bool = True,
                 use_attention: bool = False,
                 aal_processor: Optional[AALAttentionProcessor] = None,
                 attention_strength: float = 0.5,
                 rng_seed: Optional[int] = None,
                 attention_cache_dir: Optional[str] = None):
        """
        MultiModal MRI dataset with three modalities:
        1. MRI images (3D)
        2. Long text reports (BERT)
        3. Short MMSE text descriptions (CLIP)
        """
        self.root_dir = root_dir
        self.text_dir = text_dir
        self.target_shape = tuple(target_shape)
        self.bert_tokenizer = bert_tokenizer
        self.split = split
        self.apply_augmentation = bool(apply_augmentation and split == "train")
        self.use_attention = use_attention
        self.max_seq_length = max_seq_length
        self.attention_strength = attention_strength

        # randomness control
        self.rng = random.Random(rng_seed) if rng_seed is not None else random

        # CLIP model for MMSE text encoding
        try:
            self.clip_model, self.clip_preprocess = clip.load(clip_model_name)
            self.clip_model.eval()  # Freeze CLIP
            logger.info(f"Loaded CLIP model: {clip_model_name}")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

        # AAL processor (optionally attach cache dir)
        if use_attention and aal_processor is None:
            self.aal_processor = AALAttentionProcessor(
                aal3_dir=os.environ.get("AAL3_DIR", "/data/zby/AAL3"),
                cache_dir=attention_cache_dir
            )
        else:
            if aal_processor and attention_cache_dir:
                aal_processor.cache_dir = attention_cache_dir
                os.makedirs(attention_cache_dir, exist_ok=True)
            self.aal_processor = aal_processor

        # Load CSV
        self.data = pd.read_csv(csv_path)
        # drop samples without MMSCORE
        if "MMSCORE" in self.data.columns:
            self.data = self.data.dropna(subset=["MMSCORE"])
        else:
            logger.warning("CSV missing MMSCORE column; continuing without dropping.")

        self.label_map = {'CN': 0, 'AD': 1}
        # shuffle reproducibly
        self.data = self.data.sample(frac=1, random_state=(rng_seed if rng_seed is not None else 42)).reset_index(drop=True)

        total_len = len(self.data)
        train_end = int(total_len * train_ratio)
        val_end = train_end + int(total_len * val_ratio)

        if split == "train":
            self.data = self.data.iloc[:train_end]
        elif split == "val":
            self.data = self.data.iloc[train_end:val_end]
        else:
            self.data = self.data.iloc[val_end:]

        logger.info("Loaded %s split, %d samples (target_shape=%s, apply_augmentation=%s, use_attention=%s)",
                    split, len(self.data), self.target_shape, self.apply_augmentation, self.use_attention)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        row = self.data.iloc[idx]
        img_id = str(row['Image Data ID'])
        try:
            label = self.label_map.get(row['Group'], 0)
            mm_score = float(row.get('MMSCORE', 0.0))
        except Exception as e:
            logger.warning("Bad row format at idx %d: %s", idx, e)
            return None

        # locate nii file
        nii_path = os.path.join(self.root_dir, f"{img_id}.nii")
        if not os.path.exists(nii_path):
            nii_path = os.path.join(self.root_dir, f"{img_id}.nii.gz")
            if not os.path.exists(nii_path):
                logger.warning("NIfTI not found for id %s (checked .nii and .nii.gz)", img_id)
                return None

        try:
            mri_img = nib.load(nii_path)
            img = mri_img.get_fdata().astype(np.float32)
        except Exception as e:
            logger.warning("Failed loading NIfTI %s: %s", nii_path, e)
            return None

        # read report for BERT
        txt_path = os.path.join(self.text_dir, f"{img_id}_report.txt")
        report_text = extract_report_text(txt_path)

        # Optionally apply attention
        if self.use_attention and self.aal_processor:
            try:
                region_weights = self.aal_processor.create_region_weights_from_report(report_text)
                attention_map = self.aal_processor.generate_attention_map(mri_img, region_weights, img_id=img_id, use_cache=True)
                img = self.aal_processor.apply_attention_weighting(mri_img, attention_map, attention_strength=self.attention_strength)
            except Exception as e:
                logger.warning("Attention processing failed for %s: %s", img_id, e)

        # Normalize (safe)
        mean = np.mean(img)
        std = np.std(img)
        if std < 1e-6:
            std = 1.0
        img = (img - mean) / (std + 1e-8)

        # augmentation (if train)
        if self.apply_augmentation:
            img = random_rotation_3d(img, max_angle=15.0, rng=self.rng)
            img = random_flip_3d(img, prob=0.5, rng=self.rng)

        # resize to target shape
        img = self._resize_3d(img, self.target_shape)

        # add channel dim (C, D, H, W)
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # Generate MMSE text for CLIP
        mmse_text = generate_mmse_text(mm_score)
        
        # If report_text is required to be present, skip if missing
        if report_text is None:
            logger.warning("report_text failed")
            return None

        # Tokenize BERT text
        try:
            bert_tokens = self.bert_tokenizer(
                report_text,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors='pt'
            )
        except Exception as e:
            logger.warning("BERT tokenization failed for %s: %s", img_id, e)
            return None

        # Tokenize CLIP text (MMSE)
        try:
            with torch.no_grad():
                clip_tokens = clip.tokenize([mmse_text])
        except Exception as e:
            logger.warning("CLIP tokenization failed for %s: %s", img_id, e)
            return None

        # Convert to tensors
        mri_tensor = torch.from_numpy(img).to(dtype=torch.float32)
        bert_input_ids = bert_tokens['input_ids'].squeeze(0)
        bert_attention_mask = bert_tokens['attention_mask'].squeeze(0)
        clip_input_ids = clip_tokens.squeeze(0)

        return {
            'mri': mri_tensor,
            'bert_input_ids': bert_input_ids,
            'bert_attention_mask': bert_attention_mask,
            'clip_input_ids': clip_input_ids,
            'label': torch.tensor(label, dtype=torch.long),
            'mmse': torch.tensor(mm_score, dtype=torch.float32),
            'id': img_id
        }

    def _resize_3d(self, img: np.ndarray, target_shape: tuple) -> np.ndarray:
        """
        Robust resize: crop centrally if larger; pad centrally if smaller.
        """
        data = img
        for axis in range(3):
            cur = data.shape[axis]
            tgt = int(target_shape[axis])
            if cur == tgt:
                continue
            elif cur > tgt:
                start = (cur - tgt) // 2
                end = start + tgt
                sl = [slice(None)] * 3
                sl[axis] = slice(start, end)
                data = data[tuple(sl)]
            else:
                pad_before = (tgt - cur) // 2
                pad_after = tgt - cur - pad_before
                pad_width = [(0, 0)] * 3
                pad_width[axis] = (pad_before, pad_after)
                data = np.pad(data, pad_width, mode='constant', constant_values=0)
        assert data.shape == tuple(target_shape), f"resize failed: got {data.shape}, want {target_shape}"
        return data.astype(np.float32)


# -------------------------
# Collate fn
# -------------------------
def collate_fn(batch):
    """Filter out None samples and collate."""
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}
    return torch.utils.data.default_collate(batch)