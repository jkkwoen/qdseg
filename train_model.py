#!/usr/bin/env python3
"""
QDSeg Model Training Pipeline

Trains a segmentation model using AFM quantum dot data.
Supported models: Cellulus (unsupervised), StarDist, Cellpose (planned for future support)

Usage:
    # Train Cellulus model (default)
    python train_model.py

    # Specify options
    python train_model.py --model cellulus --epochs 10000 --batch-size 8

    # Custom data path
    python train_model.py --data-dir ./my_data --output-dir ./my_models

Requirements:
    pip install torch zarr toml

    # Cellulus official version (optional):
    pip install git+https://github.com/funkelab/cellulus.git

Notes:
    - Cellulus is unsupervised and does not require labels
    - GPU usage is recommended (CUDA or MPS)
    - Trained models are saved to output-dir
"""

import os
import sys
import argparse
import warnings
import platform
import multiprocessing
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np

# Add project root (parent directory of grain_analyzer folder)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    warnings.warn("python-dotenv is not installed. Run 'pip install python-dotenv' to use environment variables.", ImportWarning)
    pass


# ============================================================
# Configuration class
# ============================================================

@dataclass
class TrainingConfig:
    """Training configuration"""
    model_type: str = "cellulus"
    num_epochs: int = 10000
    batch_size: int = 4
    learning_rate: float = 1e-4
    patch_size: int = 128
    num_embeddings: int = 8
    num_fmaps: int = 32
    checkpoint_interval: int = 1000
    log_interval: int = 100
    
    # Paths
    data_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    # GPU settings
    use_gpu: bool = True
    device: Optional[str] = None
    
    # Hardware optimization settings
    num_workers: int = -1  # -1 for auto-detection
    prefetch_factor: int = 2
    pin_memory: bool = True
    
    # CUDA optimization settings
    use_amp: bool = True  # Automatic Mixed Precision
    cudnn_benchmark: bool = True  # cuDNN benchmark mode
    compile_model: bool = False  # torch.compile (PyTorch 2.0+)
    
    def __post_init__(self):
        # Read default paths from environment variables
        if self.data_dir is None:
            env_data_dir = os.getenv('QDSEG_DATA_DIR')
            if env_data_dir:
                self.data_dir = Path(env_data_dir)
            else:
                # Default: relative path from project root
                project_root = Path(__file__).parent.parent
                self.data_dir = project_root / "tests" / "input_data" / "xqd"
        
        if self.output_dir is None:
            env_output_dir = os.getenv('QDSEG_OUTPUT_DIR')
            if env_output_dir:
                self.output_dir = Path(env_output_dir) / self.model_type
            else:
                # Default: relative path from project root
                project_root = Path(__file__).parent.parent
                self.output_dir = project_root / "tests" / "model_data" / self.model_type
        
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        
        # Auto-configure num_workers (device-specific optimization)
        # Apple Silicon MPS: unified memory makes num_workers=0 faster
        # CUDA: num_workers > 0 is effective
        # Here we only set defaults; actual adjustment happens after device detection
        if self.num_workers < 0:
            self.num_workers = 0  # Default, adjusted after device detection


# ============================================================
# Dataset class (global - multiprocessing pickle compatible)
# ============================================================

class AFMDataset:
    """AFM image dataset (PyTorch Dataset compatible)"""
    
    def __init__(self, zarr_path, dataset_name='train/raw', patch_size=128):
        import zarr
        root = zarr.open_group(str(zarr_path), mode='r')
        self.data = np.array(root[dataset_name][:])
        self.patch_size = patch_size
        
    def __len__(self):
        return len(self.data) * 16
    
    def __getitem__(self, idx):
        import torch
        img_idx = idx % len(self.data)
        img = self.data[img_idx]
        
        h, w = img.shape
        y = np.random.randint(0, max(1, h - self.patch_size))
        x = np.random.randint(0, max(1, w - self.patch_size))
        
        patch = img[y:y+self.patch_size, x:x+self.patch_size]
        
        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
            padded = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            padded[:patch.shape[0], :patch.shape[1]] = patch
            patch = padded
        
        return torch.from_numpy(patch).unsqueeze(0)


# ============================================================
# Utility functions
# ============================================================

def get_hardware_info() -> Dict[str, Any]:
    """Collect system hardware information"""
    import torch
    
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
    }
    
    # Memory info (macOS)
    try:
        import subprocess
        result = subprocess.run(
            ['sysctl', '-n', 'hw.memsize'], 
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            mem_bytes = int(result.stdout.strip())
            info['memory_gb'] = mem_bytes / (1024 ** 3)
    except Exception:
        info['memory_gb'] = None
    
    # GPU info
    if torch.cuda.is_available():
        info['gpu_type'] = 'cuda'
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['gpu_type'] = 'mps'
        info['gpu_name'] = 'Apple Silicon GPU'
        # MPS uses unified memory
        info['gpu_memory_gb'] = info.get('memory_gb', 'Unified Memory')
    else:
        info['gpu_type'] = 'cpu'
        info['gpu_name'] = None
        info['gpu_memory_gb'] = None
    
    return info


def print_hardware_info():
    """Print hardware information"""
    info = get_hardware_info()
    
    print("\nHardware Info:")
    print(f"   System: {info['platform']}")
    print(f"   Processor: {info['processor']}")
    print(f"   CPU Cores: {info['cpu_count']}")
    if info.get('memory_gb'):
        print(f"   System Memory: {info['memory_gb']:.1f} GB")
    print(f"   Python: {info['python_version']}")
    print(f"   PyTorch: {info['torch_version']}")
    
    if info['gpu_type'] == 'cuda':
        print(f"   GPU: {info['gpu_name']} ({info['gpu_memory_gb']:.1f} GB)")
    elif info['gpu_type'] == 'mps':
        print(f"   GPU: {info['gpu_name']} (Unified Memory)")
    else:
        print("   GPU: Not available")
    
    return info


def get_device(use_gpu: bool = True) -> "torch.device":
    """Auto-detect the optimal device"""
    import torch
    
    if not use_gpu:
        print("   Warning: Running in CPU mode")
        return torch.device('cpu')
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"   ✓ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return device
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("   ✓ Using Apple Silicon MPS acceleration")
        return device
    
    print("   Warning: GPU not found, running in CPU mode")
    return torch.device('cpu')


def setup_environment(config: Optional[TrainingConfig] = None):
    """Environment setup - CUDA/MPS optimization"""
    import torch
    
    # Set multiprocessing start method on macOS (use spawn instead of fork)
    if platform.system() == 'Darwin':
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set
    
    # ============================================================
    # CUDA optimization
    # ============================================================
    if torch.cuda.is_available():
        # cuDNN benchmark mode - selects optimal algorithm when input size is constant
        if config is None or config.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            print("   ✓ cuDNN benchmark mode enabled")
        
        # cuDNN deterministic mode (when reproducibility is needed)
        # torch.backends.cudnn.deterministic = True
        
        # Enable TF32 (performance improvement on Ampere+ GPUs)
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        
        # CUDA memory management
        if os.environ.get('PYTORCH_CUDA_ALLOC_CONF') is None:
            # Memory fragmentation optimization
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # ============================================================
    # MPS optimization (Apple Silicon)
    # ============================================================
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS memory optimization - maximize unified memory usage
        if os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO') is None:
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.9'  # Use up to 90%
        if os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO') is None:
            os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'
        
        # MPS operation optimization
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None:
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # Fallback unsupported ops to CPU
    
    # Suppress warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*MPS.*')
    warnings.filterwarnings('ignore', message='.*beta.*')


def load_jpg_data(data_dir: Path, target_size: int = 512) -> List[np.ndarray]:
    """Load JPG image files

    Args:
        data_dir: Directory containing JPG files
        target_size: Output image size (square, default 512)
    """
    from skimage import io
    from skimage.transform import resize
    from skimage.color import rgb2gray
    
    jpg_files = sorted(list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.jpeg")) + list(data_dir.glob("*.png")))
    
    if not jpg_files:
        raise FileNotFoundError(f"No JPG files found in: {data_dir}")

    print(f"\n{len(jpg_files)} JPG files found")
    
    images = []
    for jpg_file in jpg_files:
        try:
            img = io.imread(str(jpg_file))
            
            # RGB → Grayscale
            if len(img.shape) == 3:
                img = rgb2gray(img)
            
            # Resize
            if img.shape[0] != target_size or img.shape[1] != target_size:
                img = resize(img, (target_size, target_size), 
                            preserve_range=True, anti_aliasing=True)
            
            # Normalize to [0, 1]
            pmin, pmax = np.percentile(img, [1, 99])
            if pmax - pmin > 1e-10:
                img_norm = np.clip((img - pmin) / (pmax - pmin), 0, 1)
            else:
                img_norm = np.zeros_like(img)
            
            # Gamma correction (contrast enhancement)
            gamma = 0.7
            img_norm = np.power(img_norm, gamma)

            images.append(img_norm.astype(np.float32))
            print(f"   ✓ {jpg_file.name}: {img.shape}")
        except Exception as e:
            print(f"   FAILED {jpg_file.name}: {e}")
    
    return images


def load_afm_data(data_dir: Path, target_size: int = 512) -> List[np.ndarray]:
    """Load AFM XQD files

    Args:
        data_dir: Directory containing XQD files
        target_size: Output image size (square, default 512)
    """
    from grain_analyzer import AFMData
    from skimage.transform import resize
    
    xqd_files = sorted(data_dir.glob("*.xqd"))
    
    if not xqd_files:
        raise FileNotFoundError(f"No XQD files found in: {data_dir}")

    print(f"\n{len(xqd_files)} XQD files found")
    
    images = []
    for xqd_file in xqd_files:
        try:
            data = AFMData(str(xqd_file))
            data.first_correction().second_correction().third_correction()
            data.flat_correction("line_by_line").baseline_correction("min_to_zero")
            
            height = data.get_data()
            
            # Resize to target size if needed
            if height.shape[0] != target_size or height.shape[1] != target_size:
                height = resize(height, (target_size, target_size), 
                               preserve_range=True, anti_aliasing=True)
            
            # Normalize to [0, 1]
            pmin, pmax = np.percentile(height, [1, 99])
            if pmax - pmin > 1e-10:
                height_norm = np.clip((height - pmin) / (pmax - pmin), 0, 1)
            else:
                height_norm = np.zeros_like(height)
            
            # Gamma correction (contrast enhancement - sharper boundaries)
            gamma = 0.7
            height_norm = np.power(height_norm, gamma)
            
            images.append(height_norm.astype(np.float32))
            print(f"   ✓ {xqd_file.name}: {height.shape}")
        except Exception as e:
            print(f"   FAILED {xqd_file.name}: {e}")
    
    return images


# ============================================================
# Cellulus training
# ============================================================

class CellulusTrainer:
    """Cellulus model trainer"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = None
        self.model = None
        self.optimizer = None
        self.scaler = None  # AMP GradScaler
        self.use_amp = False  # Whether to use AMP
        
    def prepare_data(self, images: List[np.ndarray]) -> Path:
        """Prepare data in Zarr format"""
        import zarr
        
        zarr_path = self.config.output_dir / "afm_grains.zarr"
        
        print(f"\nCreating Zarr dataset: {zarr_path}")
        
        stacked = np.stack(images, axis=0).astype(np.float32)
        
        n_train = max(1, len(images) - 2)
        train_data = stacked[:n_train]
        test_data = stacked[n_train:]
        
        root = zarr.open_group(str(zarr_path), mode='w')
        
        try:
            train_group = root.create_group('train')
            test_group = root.create_group('test')
            train_group.create_array('raw', data=train_data, chunks=(1, 256, 256))
            test_group.create_array('raw', data=test_data, chunks=(1, 256, 256))
        except (TypeError, AttributeError):
            root['train/raw'] = train_data
            root['test/raw'] = test_data
        
        print(f"   ✓ train/raw: {train_data.shape}")
        print(f"   ✓ test/raw: {test_data.shape}")
        
        return zarr_path
    
    def build_model(self):
        """Build model"""
        import torch
        import torch.nn as nn
        
        class SimpleUNet(nn.Module):
            """Simple U-Net based embedding network"""
            
            def __init__(self, in_channels=1, num_embeddings=8, num_fmaps=32):
                super().__init__()
                
                self.enc1 = self._block(in_channels, num_fmaps)
                self.enc2 = self._block(num_fmaps, num_fmaps * 2)
                self.enc3 = self._block(num_fmaps * 2, num_fmaps * 4)
                
                self.dec2 = self._block(num_fmaps * 4 + num_fmaps * 2, num_fmaps * 2)
                self.dec1 = self._block(num_fmaps * 2 + num_fmaps, num_fmaps)
                
                self.out = nn.Conv2d(num_fmaps, num_embeddings, 1)
                
                self.pool = nn.MaxPool2d(2)
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            
            def _block(self, in_ch, out_ch):
                return nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            
            def forward(self, x):
                e1 = self.enc1(x)
                e2 = self.enc2(self.pool(e1))
                e3 = self.enc3(self.pool(e2))
                
                d2 = self.dec2(torch.cat([self.up(e3), e2], dim=1))
                d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
                
                return self.out(d1)
        
        self.model = SimpleUNet(
            num_embeddings=self.config.num_embeddings,
            num_fmaps=self.config.num_fmaps,
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        
        return self.model
    
    def contrastive_loss(self, embeddings, temperature=0.1):
        """Object-centric contrastive loss"""
        import torch
        import torch.nn.functional as F
        
        b, c, h, w = embeddings.shape
        device = embeddings.device
        
        embeddings = F.normalize(embeddings, dim=1)
        
        num_anchors = min(256, h * w)
        indices = torch.randperm(h * w, device=device)[:num_anchors]
        
        embeddings_flat = embeddings.view(b, c, -1)
        anchors = embeddings_flat[:, :, indices]
        
        similarity = torch.bmm(anchors.transpose(1, 2), embeddings_flat) / temperature
        
        anchor_y = indices // w
        anchor_x = indices % w
        all_y = torch.arange(h, device=device).view(-1, 1).expand(h, w).reshape(-1)
        all_x = torch.arange(w, device=device).view(1, -1).expand(h, w).reshape(-1)
        
        dist = ((anchor_y.view(-1, 1) - all_y.view(1, -1)) ** 2 + 
                (anchor_x.view(-1, 1) - all_x.view(1, -1)) ** 2).float().sqrt()
        
        positive_mask = (dist < 5).float()
        negative_mask = (dist > 20).float()
        
        exp_sim = torch.exp(similarity)
        
        positive_sim = (exp_sim * positive_mask.unsqueeze(0)).sum(dim=-1)
        negative_sim = (exp_sim * negative_mask.unsqueeze(0)).sum(dim=-1)
        
        loss = -torch.log(positive_sim / (positive_sim + negative_sim + 1e-10) + 1e-10)
        
        return loss.mean()
    
    def train(self, zarr_path: Path):
        """Train model"""
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        
        # Device setup
        self.device = get_device(self.config.use_gpu)
        
        # ============================================================
        # AMP (Automatic Mixed Precision) setup
        # ============================================================
        self.use_amp = False
        self.scaler = None
        
        if self.config.use_amp:
            if self.device.type == 'cuda':
                # CUDA AMP - using FP16
                self.use_amp = True
                self.scaler = torch.cuda.amp.GradScaler()
                print("   ✓ CUDA AMP (FP16) enabled")
            elif self.device.type == 'mps':
                # Limited AMP support on MPS
                # Partially supported in PyTorch 2.0+
                try:
                    # Test autocast on MPS
                    with torch.autocast(device_type='mps', dtype=torch.float16):
                        pass
                    self.use_amp = True
                    print("   ✓ MPS AMP enabled")
                except Exception:
                    self.use_amp = False
                    print("   Warning: MPS AMP not supported - using FP32")
        
        # Build model
        self.build_model()
        
        # ============================================================
        # torch.compile (PyTorch 2.0+) - additional optimization
        # ============================================================
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                print("   ✓ torch.compile applied")
            except Exception as e:
                print(f"   Warning: torch.compile failed: {e}")
        
        # DataLoader - utilize multi-core CPU
        dataset = AFMDataset(zarr_path, patch_size=self.config.patch_size)
        
        # ============================================================
        # Device-specific DataLoader optimization
        # ============================================================
        num_workers = self.config.num_workers
        pin_memory = self.config.pin_memory
        prefetch_factor = self.config.prefetch_factor
        
        # Apple Silicon MPS optimization
        # - Unified memory: pin_memory not needed
        # - Main process loading is faster (no inter-process copy overhead)
        if self.device.type == 'mps':
            num_workers = 0  # Single process is faster on MPS
            pin_memory = False
            print("   ✓ MPS optimized: num_workers=0 (unified memory)")
        
        # CUDA optimization
        elif self.device.type == 'cuda':
            if num_workers == 0:
                cpu_count = os.cpu_count() or 4
                num_workers = min(cpu_count // 2, 8)
            pin_memory = True
            print(f"   ✓ CUDA optimized: num_workers={num_workers}, pin_memory=True")
        
        persistent_workers = num_workers > 0
        
        print(f"\nDataLoader settings:")
        print(f"   num_workers: {num_workers}")
        print(f"   pin_memory: {pin_memory}")
        
        dataloader_kwargs = {
            'batch_size': self.config.batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }
        
        # Apply additional options only when num_workers > 0
        if num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = prefetch_factor
            dataloader_kwargs['persistent_workers'] = persistent_workers
        
        dataloader = DataLoader(dataset, **dataloader_kwargs)
        
        print(f"\nStarting training: {self.config.num_epochs} epochs, batch_size={self.config.batch_size}\n")
        
        best_loss = float('inf')
        checkpoint_path = self.config.output_dir / 'best_loss.pth'
        
        # tqdm progress bar
        epoch_pbar = tqdm(
            range(self.config.num_epochs),
            desc="Training",
            unit="epoch",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        for epoch in epoch_pbar:
            self.model.train()
            total_loss = 0
            
            # Per-batch progress (within epoch)
            batch_pbar = tqdm(
                dataloader,
                desc=f"  Epoch {epoch+1}",
                leave=False,
                ncols=80,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
            )
            
            for batch in batch_pbar:
                batch = batch.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True)  # Memory efficiency
                
                # Apply AMP
                if self.use_amp and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        embeddings = self.model(batch)
                        loss = self.contrastive_loss(embeddings)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                elif self.use_amp and self.device.type == 'mps':
                    with torch.autocast(device_type='mps', dtype=torch.float16):
                        embeddings = self.model(batch)
                        loss = self.contrastive_loss(embeddings)
                    loss.backward()
                    self.optimizer.step()
                else:
                    embeddings = self.model(batch)
                    loss = self.contrastive_loss(embeddings)
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            # Update progress bar
            epoch_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'best': f'{best_loss:.4f}'
            })

            # Save checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss,
                    'config': {
                        'num_embeddings': self.config.num_embeddings,
                        'num_fmaps': self.config.num_fmaps,
                    }
                }, checkpoint_path)
                # Update progress bar
                epoch_pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'best': f'{best_loss:.4f}',
                    'saved': '✓'
                })
            
            # Periodic checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                periodic_path = self.config.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss,
                }, periodic_path)
                tqdm.write(f"   Checkpoint saved: {periodic_path.name}")
        
        epoch_pbar.close()
        print(f"\n✓ Training complete! Best loss: {best_loss:.4f}")
        print(f"✓ Model saved: {checkpoint_path}")
        
        return checkpoint_path


# ============================================================
# Trainer classes for future extension
# ============================================================

class StarDistTrainer:
    """StarDist model trainer (future implementation)"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def train(self, images: List[np.ndarray], labels: List[np.ndarray]):
        """
        StarDist training (requires labels)

        TODO: Implement custom StarDist training
        """
        raise NotImplementedError(
            "Custom StarDist training is not yet implemented.\n"
            "Reference: https://github.com/stardist/stardist"
        )


class CellposeTrainer:
    """Cellpose model trainer (future implementation)"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def train(self, images: List[np.ndarray], labels: List[np.ndarray]):
        """
        Cellpose training (requires labels)

        TODO: Implement custom Cellpose training
        """
        raise NotImplementedError(
            "Custom Cellpose training is not yet implemented.\n"
            "Reference: https://github.com/MouseLand/cellpose"
        )


# ============================================================
# Main function
# ============================================================

def get_trainer(config: TrainingConfig):
    """Return the appropriate Trainer for the given config"""
    trainers = {
        'cellulus': CellulusTrainer,
        'stardist': StarDistTrainer,
        'cellpose': CellposeTrainer,
    }
    
    if config.model_type not in trainers:
        raise ValueError(f"Unsupported model: {config.model_type}\n"
                        f"Supported models: {list(trainers.keys())}")
    
    return trainers[config.model_type](config)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='QDSeg Model Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default Cellulus training
  python train_model.py

  # Specify number of epochs
  python train_model.py --epochs 5000

  # Custom paths
  python train_model.py --data-dir ./my_data --output-dir ./my_models

  # Adjust batch size and learning rate
  python train_model.py --batch-size 8 --lr 0.0005
"""
    )
    
    # Model settings
    parser.add_argument('--model', type=str, default='cellulus',
                        choices=['cellulus', 'stardist', 'cellpose'],
                        help='Model type to train (default: cellulus)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of training epochs (default: 10000)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--patch-size', type=int, default=128,
                        help='Training patch size (default: 128)')

    # Model architecture
    parser.add_argument('--num-embeddings', type=int, default=8,
                        help='Number of embedding channels (default: 8)')
    parser.add_argument('--num-fmaps', type=int, default=32,
                        help='Number of base feature maps (default: 32)')

    # Paths
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Training data directory (default: $QDSEG_DATA_DIR or tests/input_data/xqd)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: $QDSEG_OUTPUT_DIR/{model} or tests/model_data/{model})')

    # GPU settings
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU usage')
    parser.add_argument('--device', type=str, default=None,
                        help='Specify device (cuda, mps, cpu)')

    # Checkpoint
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                        help='Checkpoint save interval (default: 1000)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Log output interval (default: 100)')

    # Hardware optimization
    parser.add_argument('--num-workers', type=int, default=-1,
                        help='Number of DataLoader workers (-1: auto, 0: disabled)')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help='Number of batches to prefetch (default: 2)')
    parser.add_argument('--no-pin-memory', action='store_true',
                        help='Disable pin_memory')

    # CUDA/GPU optimization
    parser.add_argument('--no-amp', action='store_true',
                        help='Disable Automatic Mixed Precision')
    parser.add_argument('--no-cudnn-benchmark', action='store_true',
                        help='Disable cuDNN benchmark mode')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile (PyTorch 2.0+)')
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        model_type=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patch_size=args.patch_size,
        num_embeddings=args.num_embeddings,
        num_fmaps=args.num_fmaps,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        use_gpu=not args.no_gpu,
        device=args.device,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=not args.no_pin_memory,
        use_amp=not args.no_amp,
        cudnn_benchmark=not args.no_cudnn_benchmark,
        compile_model=args.compile,
    )
    
    # Environment setup
    setup_environment(config)
    
    # Print header
    print("=" * 70)
    print(f"QDSeg Model Training: {config.model_type.upper()}")
    print("=" * 70)
    
    # Print hardware info
    hw_info = print_hardware_info()
    
    print(f"\nTraining Config:")
    print(f"   Model: {config.model_type}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   DataLoader workers: {config.num_workers}")
    print(f"   Data: {config.data_dir}")
    print(f"   Output: {config.output_dir}")
    
    # M3 Ultra optimization recommendations
    if hw_info.get('cpu_count', 0) >= 20 and hw_info.get('memory_gb', 0) >= 64:
        print(f"\nHigh-performance hardware detected! Recommended settings:")
        print(f"   --batch-size 16 or 32 (current: {config.batch_size})")
        print(f"   --num-workers 8~12 (current: {config.num_workers})")
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "=" * 70)
    print("1. Load Data")
    print("=" * 70)
    
    images = load_afm_data(config.data_dir)
    
    if not images:
        print("ERROR: No images loaded.")
        return 1
    
    # Create trainer and start training
    print("\n" + "=" * 70)
    print(f"2. {config.model_type.upper()} Model Training")
    print("=" * 70)
    
    trainer = get_trainer(config)
    
    if config.model_type == 'cellulus':
        zarr_path = trainer.prepare_data(images)
        checkpoint_path = trainer.train(zarr_path)
    else:
        # StarDist, Cellpose require labels
        trainer.train(images, labels=None)
        checkpoint_path = None
    
    # Complete
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    if checkpoint_path and checkpoint_path.exists():
        print(f"\n✓ Trained model: {checkpoint_path}")
        print("\nUsage:")
        print("   from qdseg import segment_cellulus")
        print(f"   labels = segment_cellulus(height, checkpoint_path='{checkpoint_path}')")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

