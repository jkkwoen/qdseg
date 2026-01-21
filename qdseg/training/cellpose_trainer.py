"""
Cellpose Trainer for AFM Grain Images

Cellpose 모델을 커스텀 데이터셋으로 학습시키는 모듈입니다.
지도 학습(Supervised Learning)이므로 레이블링된 마스크가 필요합니다.

Requirements:
    - cellpose
    - torch
    - numpy
    - labeled masks (from napari, ImageJ, etc.)

Usage:
    from grain_analyzer.training import CellposeTrainer, CellposeConfig
    
    config = CellposeConfig(
        data_dir=Path("./labeled_data"),
        output_dir=Path("./models/cellpose_afm"),
        n_epochs=100,
    )
    trainer = CellposeTrainer(config)
    trainer.train()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class CellposeConfig:
    """Cellpose 학습 설정"""
    
    # 데이터 경로
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    output_dir: Path = field(default_factory=lambda: Path("./models/cellpose"))
    
    # 학습 파라미터
    n_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 0.2  # Cellpose 기본값
    weight_decay: float = 1e-5
    
    # 모델 파라미터
    model_type: str = "cyto3"  # 베이스 모델 ('cyto', 'cyto2', 'cyto3', 'nuclei')
    
    # 증강
    min_train_masks: int = 1  # 최소 객체 수
    
    # 기타
    model_name: str = "cellpose_afm"
    use_gpu: bool = True
    
    # 모델 아키텍처
    nclasses: int = 3  # 기본값 (cytoplasm + nuclei + background)


class CellposeTrainer:
    """
    Cellpose 모델 학습 클래스
    
    지도 학습을 위해 다음 데이터가 필요합니다:
    - images/: 원본 AFM 이미지 (npy, tiff, png 등)
    - masks/: 레이블 마스크 (각 객체가 고유 정수로 표시)
    
    Example:
        config = CellposeConfig(
            data_dir=Path("./labeled_data"),
            output_dir=Path("./models/cellpose_afm"),
        )
        trainer = CellposeTrainer(config)
        
        # 데이터 로드
        images, masks = trainer.load_data()
        
        # 학습
        trainer.train(images, masks)
        
        # 모델 경로
        print(f"Model saved to: {trainer.get_model_path()}")
    """
    
    def __init__(self, config: CellposeConfig):
        self.config = config
        self._check_dependencies()
        self.device = None
        
    def _check_dependencies(self):
        """필요한 패키지 확인"""
        try:
            import cellpose
            import torch
        except ImportError as e:
            raise ImportError(
                "Cellpose 학습을 위해서는 cellpose가 필요합니다.\n"
                "설치: pip install cellpose\n"
                f"오류: {e}"
            )
    
    def _setup_device(self):
        """디바이스 설정"""
        import torch
        
        if not self.config.use_gpu:
            self.device = torch.device('cpu')
            print("   디바이스: CPU")
            return
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"   디바이스: CUDA ({torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("   디바이스: MPS (Apple Silicon)")
        else:
            self.device = torch.device('cpu')
            print("   디바이스: CPU")
    
    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        학습 데이터 로드
        
        데이터 구조:
            data_dir/
                images/
                    image_001.npy (또는 .tiff, .png)
                    image_002.npy
                    ...
                masks/
                    mask_001.npy (또는 .tiff, .png)
                    mask_002.npy
                    ...
        
        Returns:
            images: 이미지 리스트 (각각 2D numpy array)
            masks: 마스크 리스트 (각각 2D numpy array, int labels)
        """
        from skimage.io import imread
        
        images_dir = self.config.data_dir / "images"
        masks_dir = self.config.data_dir / "masks"
        
        if not images_dir.exists() or not masks_dir.exists():
            raise FileNotFoundError(
                f"데이터 디렉토리 구조가 올바르지 않습니다.\n"
                f"필요한 구조:\n"
                f"  {self.config.data_dir}/\n"
                f"    images/\n"
                f"    masks/\n"
            )
        
        images = []
        masks = []
        
        # 지원하는 확장자
        extensions = ['*.npy', '*.tiff', '*.tif', '*.png']
        image_files = []
        for ext in extensions:
            image_files.extend(sorted(images_dir.glob(ext)))
        
        print(f"\n📂 {len(image_files)}개의 이미지 파일 발견")
        
        for img_path in image_files:
            # 매칭되는 마스크 찾기
            mask_path = None
            for ext in ['.npy', '.tiff', '.tif', '.png']:
                candidate = masks_dir / (img_path.stem + ext)
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if mask_path is None:
                print(f"   ⚠️ {img_path.name}: 매칭되는 마스크 없음")
                continue
            
            try:
                # 이미지 로드
                if img_path.suffix == '.npy':
                    img = np.load(img_path)
                else:
                    try:
                        from tifffile import imread as tiff_imread
                        if img_path.suffix in ['.tiff', '.tif']:
                            img = tiff_imread(img_path)
                        else:
                            img = imread(img_path)
                    except ImportError:
                        img = imread(img_path)
                
                # 마스크 로드
                if mask_path.suffix == '.npy':
                    mask = np.load(mask_path)
                else:
                    try:
                        from tifffile import imread as tiff_imread
                        if mask_path.suffix in ['.tiff', '.tif']:
                            mask = tiff_imread(mask_path)
                        else:
                            mask = imread(mask_path)
                    except ImportError:
                        mask = imread(mask_path)
                
                # 정규화 (Cellpose는 0-255 범위 선호)
                if img.max() <= 1:
                    img = (img * 255).astype(np.uint8)
                elif img.max() > 255:
                    pmin, pmax = np.percentile(img, [1, 99])
                    img = np.clip((img - pmin) / (pmax - pmin + 1e-10) * 255, 0, 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                
                # 마스크가 정수형인지 확인
                mask = mask.astype(np.int32)
                
                images.append(img)
                masks.append(mask)
                print(f"   ✓ {img_path.name}: {img.shape}, {mask.max()} objects")
                
            except Exception as e:
                print(f"   ❌ {img_path.name}: {e}")
        
        if not images:
            raise ValueError("로드된 이미지가 없습니다.")
        
        print(f"\n✓ 총 {len(images)}개의 이미지-마스크 쌍 로드됨")
        return images, masks
    
    def train(
        self,
        images: Optional[List[np.ndarray]] = None,
        masks: Optional[List[np.ndarray]] = None,
    ) -> Path:
        """
        Cellpose 모델 학습
        
        Parameters:
            images: 이미지 리스트 (제공하지 않으면 load_data() 호출)
            masks: 마스크 리스트
            
        Returns:
            model_path: 학습된 모델 경로
        """
        from cellpose import models, train
        
        # 데이터 로드
        if images is None or masks is None:
            images, masks = self.load_data()
        
        # 디바이스 설정
        print("\n🔧 환경 설정 중...")
        self._setup_device()
        
        # 출력 디렉토리 생성
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 학습/테스트 분할 (Cellpose는 별도의 검증 세트 필요 없음)
        print(f"\n🚀 Cellpose 학습 시작...")
        print(f"   베이스 모델: {self.config.model_type}")
        print(f"   에폭 수: {self.config.n_epochs}")
        print(f"   배치 크기: {self.config.batch_size}")
        print(f"   학습률: {self.config.learning_rate}")
        
        # 베이스 모델 로드
        model = models.CellposeModel(
            model_type=self.config.model_type,
            gpu=self.config.use_gpu,
            device=self.device,
        )
        
        # 모델 파일 경로
        model_path = self.config.output_dir / f"{self.config.model_name}"
        
        # 학습 (Cellpose 4.x API)
        # Cellpose의 train 함수 사용
        new_model_path, train_losses, test_losses = train.train_seg(
            model.net,
            train_data=images,
            train_labels=masks,
            test_data=None,
            test_labels=None,
            channels=[0, 0],  # 그레이스케일
            save_path=str(self.config.output_dir),
            save_every=10,
            n_epochs=self.config.n_epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            batch_size=self.config.batch_size,
            min_train_masks=self.config.min_train_masks,
            model_name=self.config.model_name,
        )
        
        print(f"\n✅ 학습 완료!")
        print(f"   모델 경로: {new_model_path}")
        print(f"   최종 학습 손실: {train_losses[-1]:.4f}")
        
        return Path(new_model_path)
    
    def get_model_path(self) -> Path:
        """학습된 모델 경로 반환"""
        return self.config.output_dir / self.config.model_name


def create_labeled_data_from_rule_based(
    xqd_files: List[Path],
    output_dir: Path,
    review_with_napari: bool = True,
) -> None:
    """
    Rule-based 세그멘테이션을 사용하여 초기 레이블 데이터 생성
    
    이 함수는 Rule-based 방법으로 초기 마스크를 생성하고,
    napari로 리뷰/수정할 수 있게 합니다.
    
    Parameters:
        xqd_files: XQD 파일 경로 리스트
        output_dir: 출력 디렉토리 (images/, masks/ 하위 폴더 생성)
        review_with_napari: napari로 리뷰 여부
    
    Example:
        from pathlib import Path
        xqd_files = list(Path("./input_data/xqd").glob("*.xqd"))
        create_labeled_data_from_rule_based(
            xqd_files,
            Path("./labeled_data"),
            review_with_napari=True,
        )
    """
    from grain_analyzer import AFMData, segment_rule_based
    
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 {len(xqd_files)}개 파일 처리 중...")
    
    for xqd_file in xqd_files:
        try:
            # AFM 데이터 로드
            data = AFMData(str(xqd_file))
            data.first_correction().second_correction().third_correction()
            data.flat_correction("line_by_line").baseline_correction("min_to_zero")
            
            height = data.get_data()
            meta = data.get_meta_data()
            
            # Cellpose용 이미지 준비 (0-255 범위)
            pmin, pmax = np.percentile(height, [1, 99])
            height_uint8 = np.clip((height - pmin) / (pmax - pmin) * 255, 0, 255).astype(np.uint8)
            
            # Rule-based 세그멘테이션
            labels = segment_rule_based(height, meta)
            
            # 저장
            img_path = images_dir / f"{xqd_file.stem}.npy"
            mask_path = masks_dir / f"{xqd_file.stem}.npy"
            
            np.save(img_path, height_uint8)
            np.save(mask_path, labels)
            
            print(f"   ✓ {xqd_file.name}: {labels.max()} objects")
            
        except Exception as e:
            print(f"   ❌ {xqd_file.name}: {e}")
    
    print(f"\n✅ 초기 레이블 데이터 생성 완료: {output_dir}")
    
    if review_with_napari:
        print("\n📝 napari로 레이블 검토/수정을 권장합니다:")
        print("   pip install napari[all]")
        print("   napari")
        print(f"   - 이미지 폴더: {images_dir}")
        print(f"   - 마스크 폴더: {masks_dir}")
        print("\n   napari에서 마스크를 수정한 후 같은 경로에 저장하세요.")

