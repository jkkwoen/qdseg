"""
StarDist Trainer for AFM Grain Images

StarDist 모델을 커스텀 데이터셋으로 학습시키는 모듈입니다.
지도 학습(Supervised Learning)이므로 레이블링된 마스크가 필요합니다.

Requirements:
    - stardist
    - tensorflow
    - numpy
    - labeled masks (from napari, ImageJ, etc.)

Usage:
    from grain_analyzer.training import StarDistTrainer, StarDistConfig
    
    config = StarDistConfig(
        data_dir=Path("./labeled_data"),
        output_dir=Path("./models/stardist_afm"),
        epochs=100,
    )
    trainer = StarDistTrainer(config)
    trainer.train()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class StarDistConfig:
    """StarDist 학습 설정"""
    
    # 데이터 경로
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    output_dir: Path = field(default_factory=lambda: Path("./models/stardist"))
    
    # 학습 파라미터
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-4
    
    # 모델 파라미터
    n_rays: int = 32  # StarDist의 ray 개수 (32가 일반적)
    grid: Tuple[int, int] = (2, 2)  # 예측 그리드 스케일
    
    # 증강
    use_augmentation: bool = True
    
    # 검증
    validation_split: float = 0.1
    
    # 기타
    model_name: str = "stardist_afm"


class StarDistTrainer:
    """
    StarDist 모델 학습 클래스
    
    지도 학습을 위해 다음 데이터가 필요합니다:
    - images/: 원본 AFM 이미지 (npy, tiff, png 등)
    - masks/: 레이블 마스크 (각 객체가 고유 정수로 표시)
    
    Example:
        config = StarDistConfig(
            data_dir=Path("./labeled_data"),
            output_dir=Path("./models/stardist_afm"),
        )
        trainer = StarDistTrainer(config)
        
        # 데이터 로드
        images, masks = trainer.load_data()
        
        # 학습
        trainer.train(images, masks)
        
        # 모델 경로
        print(f"Model saved to: {trainer.get_model_path()}")
    """
    
    def __init__(self, config: StarDistConfig):
        self.config = config
        self._check_dependencies()
        
    def _check_dependencies(self):
        """필요한 패키지 확인"""
        try:
            import stardist
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "StarDist 학습을 위해서는 stardist와 tensorflow가 필요합니다.\n"
                "설치: pip install stardist tensorflow\n"
                f"오류: {e}"
            )
    
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
        from tifffile import imread as tiff_imread
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
                elif img_path.suffix in ['.tiff', '.tif']:
                    img = tiff_imread(img_path)
                else:
                    img = imread(img_path)
                
                # 마스크 로드
                if mask_path.suffix == '.npy':
                    mask = np.load(mask_path)
                elif mask_path.suffix in ['.tiff', '.tif']:
                    mask = tiff_imread(mask_path)
                else:
                    mask = imread(mask_path)
                
                # 정규화
                if img.max() > 1:
                    img = img.astype(np.float32)
                    pmin, pmax = np.percentile(img, [1, 99])
                    img = np.clip((img - pmin) / (pmax - pmin + 1e-10), 0, 1)
                
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
        StarDist 모델 학습
        
        Parameters:
            images: 이미지 리스트 (제공하지 않으면 load_data() 호출)
            masks: 마스크 리스트
            
        Returns:
            model_path: 학습된 모델 경로
        """
        from stardist import fill_label_holes, random_label_cmap
        from stardist.models import Config2D, StarDist2D
        from csbdeep.utils import normalize
        
        # 데이터 로드
        if images is None or masks is None:
            images, masks = self.load_data()
        
        # 레이블 정리 (구멍 채우기)
        print("\n🔧 레이블 전처리 중...")
        masks = [fill_label_holes(mask) for mask in masks]
        
        # 학습/검증 분할
        n_val = max(1, int(len(images) * self.config.validation_split))
        X_train, Y_train = images[:-n_val], masks[:-n_val]
        X_val, Y_val = images[-n_val:], masks[-n_val:]
        
        print(f"   학습: {len(X_train)}개, 검증: {len(X_val)}개")
        
        # 모델 설정
        print("\n🏗️ 모델 설정 중...")
        conf = Config2D(
            n_rays=self.config.n_rays,
            grid=self.config.grid,
            n_channel_in=1,
            train_epochs=self.config.epochs,
            train_batch_size=self.config.batch_size,
            train_learning_rate=self.config.learning_rate,
            use_gpu=True,
        )
        print(f"   n_rays: {conf.n_rays}")
        print(f"   grid: {conf.grid}")
        print(f"   epochs: {conf.train_epochs}")
        
        # 출력 디렉토리 생성
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 생성
        model = StarDist2D(
            conf,
            name=self.config.model_name,
            basedir=str(self.config.output_dir),
        )
        
        # 증강 설정
        augmenter = None
        if self.config.use_augmentation:
            from csbdeep.data import Normalizer, normalize_mi_ma
            
            def simple_augmenter(x, y):
                """간단한 데이터 증강 (회전, 플립)"""
                # 랜덤 회전 (0, 90, 180, 270도)
                k = np.random.randint(0, 4)
                x = np.rot90(x, k)
                y = np.rot90(y, k)
                
                # 랜덤 수평 플립
                if np.random.rand() > 0.5:
                    x = np.fliplr(x)
                    y = np.fliplr(y)
                
                # 랜덤 수직 플립
                if np.random.rand() > 0.5:
                    x = np.flipud(x)
                    y = np.flipud(y)
                
                return x, y
            
            augmenter = simple_augmenter
        
        # 학습
        print(f"\n🚀 StarDist 학습 시작...")
        print(f"   모델 저장 경로: {self.config.output_dir / self.config.model_name}")
        
        # 데이터를 numpy array로 스택 (stardist 요구사항)
        X_train_arr = np.array(X_train)
        Y_train_arr = np.array(Y_train)
        X_val_arr = np.array(X_val)
        Y_val_arr = np.array(Y_val)
        
        # 채널 차원 추가 (필요시)
        if X_train_arr.ndim == 3:
            X_train_arr = X_train_arr[..., np.newaxis]
        if X_val_arr.ndim == 3:
            X_val_arr = X_val_arr[..., np.newaxis]
        
        model.train(
            X_train_arr, Y_train_arr,
            validation_data=(X_val_arr, Y_val_arr),
            augmenter=augmenter,
        )
        
        # 임계값 최적화
        print("\n🔧 임계값 최적화 중...")
        model.optimize_thresholds(X_val_arr, Y_val_arr)
        
        model_path = self.config.output_dir / self.config.model_name
        print(f"\n✅ 학습 완료!")
        print(f"   모델 경로: {model_path}")
        
        return model_path
    
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
            
            # 정규화
            pmin, pmax = np.percentile(height, [1, 99])
            height_norm = np.clip((height - pmin) / (pmax - pmin), 0, 1).astype(np.float32)
            
            # Rule-based 세그멘테이션
            labels = segment_rule_based(height, meta)
            
            # 저장
            img_path = images_dir / f"{xqd_file.stem}.npy"
            mask_path = masks_dir / f"{xqd_file.stem}.npy"
            
            np.save(img_path, height_norm)
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

