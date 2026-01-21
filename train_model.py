#!/usr/bin/env python3
"""
QDSeg 모델 학습 파이프라인

AFM quantum dot 데이터로 세그멘테이션 모델을 학습합니다.
지원 모델: Cellulus (비지도 학습), StarDist, Cellpose (향후 지원 예정)

사용법:
    # Cellulus 모델 학습 (기본)
    python train_model.py
    
    # 옵션 지정
    python train_model.py --model cellulus --epochs 10000 --batch-size 8
    
    # 커스텀 데이터 경로
    python train_model.py --data-dir ./my_data --output-dir ./my_models

요구사항:
    pip install torch zarr toml
    
    # Cellulus 공식 버전 (선택):
    pip install git+https://github.com/funkelab/cellulus.git

참고:
    - Cellulus는 비지도 학습 기반으로 라벨이 필요 없습니다
    - GPU 사용을 권장합니다 (CUDA 또는 MPS)
    - 학습된 모델은 output-dir에 저장됩니다
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

# 프로젝트 루트 추가 (grain_analyzer 폴더의 상위 디렉토리)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    warnings.warn("python-dotenv가 설치되지 않았습니다. 환경변수를 사용하려면 'pip install python-dotenv'를 실행하세요.", ImportWarning)
    pass


# ============================================================
# 설정 클래스
# ============================================================

@dataclass
class TrainingConfig:
    """학습 설정"""
    model_type: str = "cellulus"
    num_epochs: int = 10000
    batch_size: int = 4
    learning_rate: float = 1e-4
    patch_size: int = 128
    num_embeddings: int = 8
    num_fmaps: int = 32
    checkpoint_interval: int = 1000
    log_interval: int = 100
    
    # 경로
    data_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    
    # GPU 설정
    use_gpu: bool = True
    device: Optional[str] = None
    
    # 하드웨어 최적화 설정
    num_workers: int = -1  # -1이면 자동 감지
    prefetch_factor: int = 2
    pin_memory: bool = True
    
    # CUDA 최적화 설정
    use_amp: bool = True  # Automatic Mixed Precision
    cudnn_benchmark: bool = True  # cuDNN 벤치마크 모드
    compile_model: bool = False  # torch.compile (PyTorch 2.0+)
    
    def __post_init__(self):
        # 환경변수에서 기본 경로 읽기
        if self.data_dir is None:
            env_data_dir = os.getenv('QDSEG_DATA_DIR')
            if env_data_dir:
                self.data_dir = Path(env_data_dir)
            else:
                # 기본값: 프로젝트 루트 기준 상대 경로
                project_root = Path(__file__).parent.parent
                self.data_dir = project_root / "tests" / "input_data" / "xqd"
        
        if self.output_dir is None:
            env_output_dir = os.getenv('QDSEG_OUTPUT_DIR')
            if env_output_dir:
                self.output_dir = Path(env_output_dir) / self.model_type
            else:
                # 기본값: 프로젝트 루트 기준 상대 경로
                project_root = Path(__file__).parent.parent
                self.output_dir = project_root / "tests" / "model_data" / self.model_type
        
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        
        # num_workers 자동 설정 (디바이스별 최적화)
        # Apple Silicon MPS: 통합 메모리라 num_workers=0이 더 빠름
        # CUDA: num_workers > 0이 효과적
        # 여기서는 일단 기본값만 설정, 실제 디바이스 감지 후 조정됨
        if self.num_workers < 0:
            self.num_workers = 0  # 기본값, 디바이스 감지 후 조정


# ============================================================
# 데이터셋 클래스 (전역 - multiprocessing pickle 호환)
# ============================================================

class AFMDataset:
    """AFM 이미지 데이터셋 (PyTorch Dataset 호환)"""
    
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
# 유틸리티 함수
# ============================================================

def get_hardware_info() -> Dict[str, Any]:
    """시스템 하드웨어 정보 수집"""
    import torch
    
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
    }
    
    # 메모리 정보 (macOS)
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
    
    # GPU 정보
    if torch.cuda.is_available():
        info['gpu_type'] = 'cuda'
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['gpu_type'] = 'mps'
        info['gpu_name'] = 'Apple Silicon GPU'
        # MPS는 통합 메모리 사용
        info['gpu_memory_gb'] = info.get('memory_gb', 'Unified Memory')
    else:
        info['gpu_type'] = 'cpu'
        info['gpu_name'] = None
        info['gpu_memory_gb'] = None
    
    return info


def print_hardware_info():
    """하드웨어 정보 출력"""
    info = get_hardware_info()
    
    print("\n💻 하드웨어 정보:")
    print(f"   시스템: {info['platform']}")
    print(f"   프로세서: {info['processor']}")
    print(f"   CPU 코어: {info['cpu_count']}개")
    if info.get('memory_gb'):
        print(f"   시스템 메모리: {info['memory_gb']:.1f} GB")
    print(f"   Python: {info['python_version']}")
    print(f"   PyTorch: {info['torch_version']}")
    
    if info['gpu_type'] == 'cuda':
        print(f"   GPU: {info['gpu_name']} ({info['gpu_memory_gb']:.1f} GB)")
    elif info['gpu_type'] == 'mps':
        print(f"   GPU: {info['gpu_name']} (통합 메모리)")
    else:
        print("   GPU: 사용 불가")
    
    return info


def get_device(use_gpu: bool = True) -> "torch.device":
    """최적의 디바이스 자동 감지"""
    import torch
    
    if not use_gpu:
        print("   ⚠️ CPU 모드로 실행")
        return torch.device('cpu')
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"   ✓ CUDA GPU 사용: {torch.cuda.get_device_name(0)}")
        return device
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("   ✓ Apple Silicon MPS 가속 사용")
        return device
    
    print("   ⚠️ GPU를 찾을 수 없음, CPU 모드로 실행")
    return torch.device('cpu')


def setup_environment(config: Optional[TrainingConfig] = None):
    """환경 설정 - CUDA/MPS 최적화"""
    import torch
    
    # macOS에서 멀티프로세싱 시작 방법 설정 (fork 대신 spawn 사용)
    if platform.system() == 'Darwin':
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # 이미 설정된 경우
    
    # ============================================================
    # CUDA 최적화
    # ============================================================
    if torch.cuda.is_available():
        # cuDNN 벤치마크 모드 - 입력 크기가 일정할 때 최적 알고리즘 선택
        if config is None or config.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            print("   ✓ cuDNN benchmark 모드 활성화")
        
        # cuDNN deterministic 모드 (재현성 필요시)
        # torch.backends.cudnn.deterministic = True
        
        # TF32 활성화 (Ampere 이상 GPU에서 성능 향상)
        if hasattr(torch.backends.cuda, 'matmul'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        
        # CUDA 메모리 관리
        if os.environ.get('PYTORCH_CUDA_ALLOC_CONF') is None:
            # 메모리 분할 최적화
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
    # ============================================================
    # MPS 최적화 (Apple Silicon)
    # ============================================================
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS 메모리 최적화 - 통합 메모리 최대 활용
        if os.environ.get('PYTORCH_MPS_HIGH_WATERMARK_RATIO') is None:
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.9'  # 90%까지 사용
        if os.environ.get('PYTORCH_MPS_LOW_WATERMARK_RATIO') is None:
            os.environ['PYTORCH_MPS_LOW_WATERMARK_RATIO'] = '0.0'
        
        # MPS 연산 최적화
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') is None:
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 미지원 연산은 CPU로 폴백
    
    # 경고 숨김
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', message='.*MPS.*')
    warnings.filterwarnings('ignore', message='.*beta.*')


def load_jpg_data(data_dir: Path, target_size: int = 512) -> List[np.ndarray]:
    """JPG 이미지 파일들 로드
    
    Args:
        data_dir: JPG 파일들이 있는 디렉토리
        target_size: 출력 이미지 크기 (정사각형, 기본값 512)
    """
    from skimage import io
    from skimage.transform import resize
    from skimage.color import rgb2gray
    
    jpg_files = sorted(list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.jpeg")) + list(data_dir.glob("*.png")))
    
    if not jpg_files:
        raise FileNotFoundError(f"JPG 파일을 찾을 수 없습니다: {data_dir}")
    
    print(f"\n📂 {len(jpg_files)}개의 JPG 파일 발견")
    
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
            
            # 감마 보정 (대비 강화)
            gamma = 0.7
            img_norm = np.power(img_norm, gamma)
            
            images.append(img_norm.astype(np.float32))
            print(f"   ✓ {jpg_file.name}: {img.shape}")
        except Exception as e:
            print(f"   ❌ {jpg_file.name}: {e}")
    
    return images


def load_afm_data(data_dir: Path, target_size: int = 512) -> List[np.ndarray]:
    """AFM XQD 파일들 로드
    
    Args:
        data_dir: XQD 파일들이 있는 디렉토리
        target_size: 출력 이미지 크기 (정사각형, 기본값 512)
    """
    from grain_analyzer import AFMData
    from skimage.transform import resize
    
    xqd_files = sorted(data_dir.glob("*.xqd"))
    
    if not xqd_files:
        raise FileNotFoundError(f"XQD 파일을 찾을 수 없습니다: {data_dir}")
    
    print(f"\n📂 {len(xqd_files)}개의 XQD 파일 발견")
    
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
            
            # 감마 보정 (대비 강화 - 경계를 더 명확하게)
            gamma = 0.7
            height_norm = np.power(height_norm, gamma)
            
            images.append(height_norm.astype(np.float32))
            print(f"   ✓ {xqd_file.name}: {height.shape}")
        except Exception as e:
            print(f"   ❌ {xqd_file.name}: {e}")
    
    return images


# ============================================================
# Cellulus 학습
# ============================================================

class CellulusTrainer:
    """Cellulus 모델 학습기"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = None
        self.model = None
        self.optimizer = None
        self.scaler = None  # AMP GradScaler
        self.use_amp = False  # AMP 사용 여부
        
    def prepare_data(self, images: List[np.ndarray]) -> Path:
        """데이터를 Zarr 형식으로 준비"""
        import zarr
        
        zarr_path = self.config.output_dir / "afm_grains.zarr"
        
        print(f"\n📦 Zarr 데이터셋 생성: {zarr_path}")
        
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
        """모델 구축"""
        import torch
        import torch.nn as nn
        
        class SimpleUNet(nn.Module):
            """간단한 U-Net 기반 임베딩 네트워크"""
            
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
        """모델 학습"""
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        
        # 디바이스 설정
        self.device = get_device(self.config.use_gpu)
        
        # ============================================================
        # AMP (Automatic Mixed Precision) 설정
        # ============================================================
        self.use_amp = False
        self.scaler = None
        
        if self.config.use_amp:
            if self.device.type == 'cuda':
                # CUDA AMP - FP16 사용
                self.use_amp = True
                self.scaler = torch.cuda.amp.GradScaler()
                print("   ✓ CUDA AMP (FP16) 활성화")
            elif self.device.type == 'mps':
                # MPS에서는 AMP 제한적 지원
                # PyTorch 2.0+ 에서 일부 지원
                try:
                    # MPS에서 autocast 테스트
                    with torch.autocast(device_type='mps', dtype=torch.float16):
                        pass
                    self.use_amp = True
                    print("   ✓ MPS AMP 활성화")
                except Exception:
                    self.use_amp = False
                    print("   ⚠️ MPS AMP 미지원 - FP32 사용")
        
        # 모델 구축
        self.build_model()
        
        # ============================================================
        # torch.compile (PyTorch 2.0+) - 추가 최적화
        # ============================================================
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                print("   ✓ torch.compile 적용")
            except Exception as e:
                print(f"   ⚠️ torch.compile 실패: {e}")
        
        # 데이터로더 - 멀티코어 CPU 활용
        dataset = AFMDataset(zarr_path, patch_size=self.config.patch_size)
        
        # ============================================================
        # 디바이스별 DataLoader 최적화
        # ============================================================
        num_workers = self.config.num_workers
        pin_memory = self.config.pin_memory
        prefetch_factor = self.config.prefetch_factor
        
        # Apple Silicon MPS 최적화
        # - 통합 메모리: pin_memory 불필요
        # - 메인 프로세스 로딩이 더 빠름 (프로세스간 복사 오버헤드 없음)
        if self.device.type == 'mps':
            num_workers = 0  # MPS에서는 단일 프로세스가 더 빠름
            pin_memory = False
            print("   ✓ MPS 최적화: num_workers=0 (통합 메모리)")
        
        # CUDA 최적화
        elif self.device.type == 'cuda':
            if num_workers == 0:
                cpu_count = os.cpu_count() or 4
                num_workers = min(cpu_count // 2, 8)
            pin_memory = True
            print(f"   ✓ CUDA 최적화: num_workers={num_workers}, pin_memory=True")
        
        persistent_workers = num_workers > 0
        
        print(f"\n⚡ DataLoader 설정:")
        print(f"   num_workers: {num_workers}")
        print(f"   pin_memory: {pin_memory}")
        
        dataloader_kwargs = {
            'batch_size': self.config.batch_size,
            'shuffle': True,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
        }
        
        # num_workers > 0일 때만 추가 옵션 적용
        if num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = prefetch_factor
            dataloader_kwargs['persistent_workers'] = persistent_workers
        
        dataloader = DataLoader(dataset, **dataloader_kwargs)
        
        print(f"\n🚀 학습 시작: {self.config.num_epochs} epochs, batch_size={self.config.batch_size}\n")
        
        best_loss = float('inf')
        checkpoint_path = self.config.output_dir / 'best_loss.pth'
        
        # tqdm 진행 표시줄
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
            
            # 배치별 진행 표시 (에폭 내)
            batch_pbar = tqdm(
                dataloader,
                desc=f"  Epoch {epoch+1}",
                leave=False,
                ncols=80,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}'
            )
            
            for batch in batch_pbar:
                batch = batch.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad(set_to_none=True)  # 메모리 효율성
                
                # AMP 적용
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
            
            # 진행 표시줄 업데이트
            epoch_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'best': f'{best_loss:.4f}'
            })
            
            # 체크포인트 저장
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
                # 진행 표시줄 업데이트
                epoch_pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'best': f'{best_loss:.4f}',
                    'saved': '✓'
                })
            
            # 주기적 체크포인트
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                periodic_path = self.config.output_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': avg_loss,
                }, periodic_path)
                tqdm.write(f"   💾 체크포인트 저장: {periodic_path.name}")
        
        epoch_pbar.close()
        print(f"\n✓ 학습 완료! Best loss: {best_loss:.4f}")
        print(f"✓ 모델 저장됨: {checkpoint_path}")
        
        return checkpoint_path


# ============================================================
# 향후 확장용 Trainer 클래스들
# ============================================================

class StarDistTrainer:
    """StarDist 모델 학습기 (향후 구현)"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def train(self, images: List[np.ndarray], labels: List[np.ndarray]):
        """
        StarDist 학습 (라벨 필요)
        
        TODO: StarDist 커스텀 학습 구현
        """
        raise NotImplementedError(
            "StarDist 커스텀 학습은 아직 구현되지 않았습니다.\n"
            "참고: https://github.com/stardist/stardist"
        )


class CellposeTrainer:
    """Cellpose 모델 학습기 (향후 구현)"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def train(self, images: List[np.ndarray], labels: List[np.ndarray]):
        """
        Cellpose 학습 (라벨 필요)
        
        TODO: Cellpose 커스텀 학습 구현
        """
        raise NotImplementedError(
            "Cellpose 커스텀 학습은 아직 구현되지 않았습니다.\n"
            "참고: https://github.com/MouseLand/cellpose"
        )


# ============================================================
# 메인 함수
# ============================================================

def get_trainer(config: TrainingConfig):
    """설정에 맞는 Trainer 반환"""
    trainers = {
        'cellulus': CellulusTrainer,
        'stardist': StarDistTrainer,
        'cellpose': CellposeTrainer,
    }
    
    if config.model_type not in trainers:
        raise ValueError(f"지원하지 않는 모델: {config.model_type}\n"
                        f"지원 모델: {list(trainers.keys())}")
    
    return trainers[config.model_type](config)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description='QDSeg 모델 학습',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 Cellulus 학습
  python train_model.py
  
  # 에폭 수 지정
  python train_model.py --epochs 5000
  
  # 커스텀 경로
  python train_model.py --data-dir ./my_data --output-dir ./my_models
  
  # 배치 크기 및 학습률 조정
  python train_model.py --batch-size 8 --lr 0.0005
"""
    )
    
    # 모델 설정
    parser.add_argument('--model', type=str, default='cellulus',
                        choices=['cellulus', 'stardist', 'cellpose'],
                        help='학습할 모델 타입 (기본: cellulus)')
    
    # 학습 파라미터
    parser.add_argument('--epochs', type=int, default=10000,
                        help='학습 에폭 수 (기본: 10000)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='배치 크기 (기본: 4)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='학습률 (기본: 0.0001)')
    parser.add_argument('--patch-size', type=int, default=128,
                        help='학습 패치 크기 (기본: 128)')
    
    # 모델 구조
    parser.add_argument('--num-embeddings', type=int, default=8,
                        help='임베딩 채널 수 (기본: 8)')
    parser.add_argument('--num-fmaps', type=int, default=32,
                        help='기본 특성 맵 수 (기본: 32)')
    
    # 경로
    parser.add_argument('--data-dir', type=str, default=None,
                        help='학습 데이터 디렉토리 (기본: $QDSEG_DATA_DIR 또는 tests/input_data/xqd)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='출력 디렉토리 (기본: $QDSEG_OUTPUT_DIR/{model} 또는 tests/model_data/{model})')
    
    # GPU 설정
    parser.add_argument('--no-gpu', action='store_true',
                        help='GPU 사용 비활성화')
    parser.add_argument('--device', type=str, default=None,
                        help='특정 디바이스 지정 (cuda, mps, cpu)')
    
    # 체크포인트
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                        help='체크포인트 저장 간격 (기본: 1000)')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='로그 출력 간격 (기본: 100)')
    
    # 하드웨어 최적화
    parser.add_argument('--num-workers', type=int, default=-1,
                        help='DataLoader 워커 수 (-1: 자동, 0: 비활성화)')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help='미리 로드할 배치 수 (기본: 2)')
    parser.add_argument('--no-pin-memory', action='store_true',
                        help='pin_memory 비활성화')
    
    # CUDA/GPU 최적화
    parser.add_argument('--no-amp', action='store_true',
                        help='Automatic Mixed Precision 비활성화')
    parser.add_argument('--no-cudnn-benchmark', action='store_true',
                        help='cuDNN benchmark 모드 비활성화')
    parser.add_argument('--compile', action='store_true',
                        help='torch.compile 사용 (PyTorch 2.0+)')
    
    args = parser.parse_args()
    
    # 설정 생성
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
    
    # 환경 설정
    setup_environment(config)
    
    # 헤더 출력
    print("=" * 70)
    print(f"🧠 QDSeg 모델 학습: {config.model_type.upper()}")
    print("=" * 70)
    
    # 하드웨어 정보 출력
    hw_info = print_hardware_info()
    
    print(f"\n📋 학습 설정:")
    print(f"   모델: {config.model_type}")
    print(f"   에폭: {config.num_epochs}")
    print(f"   배치 크기: {config.batch_size}")
    print(f"   학습률: {config.learning_rate}")
    print(f"   DataLoader 워커: {config.num_workers}")
    print(f"   데이터: {config.data_dir}")
    print(f"   출력: {config.output_dir}")
    
    # M3 Ultra 최적화 권장사항
    if hw_info.get('cpu_count', 0) >= 20 and hw_info.get('memory_gb', 0) >= 64:
        print(f"\n💡 고성능 하드웨어 감지! 권장 설정:")
        print(f"   --batch-size 16 또는 32 (현재: {config.batch_size})")
        print(f"   --num-workers 8~12 (현재: {config.num_workers})")
    
    # 출력 디렉토리 생성
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 데이터 로드
    print("\n" + "=" * 70)
    print("1️⃣  데이터 로드")
    print("=" * 70)
    
    images = load_afm_data(config.data_dir)
    
    if not images:
        print("❌ 로드된 이미지가 없습니다.")
        return 1
    
    # Trainer 생성 및 학습
    print("\n" + "=" * 70)
    print(f"2️⃣  {config.model_type.upper()} 모델 학습")
    print("=" * 70)
    
    trainer = get_trainer(config)
    
    if config.model_type == 'cellulus':
        zarr_path = trainer.prepare_data(images)
        checkpoint_path = trainer.train(zarr_path)
    else:
        # StarDist, Cellpose는 라벨이 필요
        trainer.train(images, labels=None)
        checkpoint_path = None
    
    # 완료
    print("\n" + "=" * 70)
    print("✅ 학습 완료!")
    print("=" * 70)
    
    if checkpoint_path and checkpoint_path.exists():
        print(f"\n✓ 학습된 모델: {checkpoint_path}")
        print("\n사용 방법:")
        print("   from qdseg import segment_cellulus")
        print(f"   labels = segment_cellulus(height, checkpoint_path='{checkpoint_path}')")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

