"""
StarDist Trainer for AFM Grain Images

Module for training StarDist models on custom datasets.
Requires labeled masks as this is supervised learning.

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
    """StarDist training configuration"""

    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    output_dir: Path = field(default_factory=lambda: Path("./models/stardist"))
    
    # Training parameters
    epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-4
    
    # Model parameters
    n_rays: int = 32  # Number of StarDist rays (32 is typical)
    grid: Tuple[int, int] = (2, 2)  # Prediction grid scale

    # Augmentation
    use_augmentation: bool = True
    
    # Validation
    validation_split: float = 0.1
    
    # Misc
    model_name: str = "stardist_afm"


class StarDistTrainer:
    """
    StarDist model training class

    The following data is required for supervised learning:
    - images/: Original AFM images (npy, tiff, png, etc.)
    - masks/: Label masks (each object marked with a unique integer)

    Example:
        config = StarDistConfig(
            data_dir=Path("./labeled_data"),
            output_dir=Path("./models/stardist_afm"),
        )
        trainer = StarDistTrainer(config)

        # Load data
        images, masks = trainer.load_data()

        # Train
        trainer.train(images, masks)

        # Model path
        print(f"Model saved to: {trainer.get_model_path()}")
    """
    
    def __init__(self, config: StarDistConfig):
        self.config = config
        self._check_dependencies()
        
    def _check_dependencies(self):
        """Check required packages"""
        try:
            import stardist
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "stardist and tensorflow are required for StarDist training.\n"
                "Install: pip install stardist tensorflow\n"
                f"Error: {e}"
            )
    
    def load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Load training data

        Data structure:
            data_dir/
                images/
                    image_001.npy (or .tiff, .png)
                    image_002.npy
                    ...
                masks/
                    mask_001.npy (or .tiff, .png)
                    mask_002.npy
                    ...

        Returns:
            images: List of images (each a 2D numpy array)
            masks: List of masks (each a 2D numpy array with int labels)
        """
        from tifffile import imread as tiff_imread
        from skimage.io import imread
        
        images_dir = self.config.data_dir / "images"
        masks_dir = self.config.data_dir / "masks"
        
        if not images_dir.exists() or not masks_dir.exists():
            raise FileNotFoundError(
                f"Invalid data directory structure.\n"
                f"Required structure:\n"
                f"  {self.config.data_dir}/\n"
                f"    images/\n"
                f"    masks/\n"
            )
        
        images = []
        masks = []
        
        # Supported extensions
        extensions = ['*.npy', '*.tiff', '*.tif', '*.png']
        image_files = []
        for ext in extensions:
            image_files.extend(sorted(images_dir.glob(ext)))
        
        print(f"\n📂 Found {len(image_files)} image files")
        
        for img_path in image_files:
            # Find matching mask
            mask_path = None
            for ext in ['.npy', '.tiff', '.tif', '.png']:
                candidate = masks_dir / (img_path.stem + ext)
                if candidate.exists():
                    mask_path = candidate
                    break
            
            if mask_path is None:
                print(f"   ⚠️ {img_path.name}: No matching mask found")
                continue
            
            try:
                # Load image
                if img_path.suffix == '.npy':
                    img = np.load(img_path)
                elif img_path.suffix in ['.tiff', '.tif']:
                    img = tiff_imread(img_path)
                else:
                    img = imread(img_path)
                
                # Load mask
                if mask_path.suffix == '.npy':
                    mask = np.load(mask_path)
                elif mask_path.suffix in ['.tiff', '.tif']:
                    mask = tiff_imread(mask_path)
                else:
                    mask = imread(mask_path)
                
                # Normalize
                if img.max() > 1:
                    img = img.astype(np.float32)
                    pmin, pmax = np.percentile(img, [1, 99])
                    img = np.clip((img - pmin) / (pmax - pmin + 1e-10), 0, 1)
                
                # Ensure mask is integer type
                mask = mask.astype(np.int32)
                
                images.append(img)
                masks.append(mask)
                print(f"   ✓ {img_path.name}: {img.shape}, {mask.max()} objects")
                
            except Exception as e:
                print(f"   ❌ {img_path.name}: {e}")
        
        if not images:
            raise ValueError("No images loaded.")
        
        print(f"\n✓ Loaded {len(images)} image-mask pairs")
        return images, masks
    
    def train(
        self,
        images: Optional[List[np.ndarray]] = None,
        masks: Optional[List[np.ndarray]] = None,
    ) -> Path:
        """
        Train StarDist model

        Parameters:
            images: List of images (calls load_data() if not provided)
            masks: List of masks

        Returns:
            model_path: Path to the trained model
        """
        from stardist import fill_label_holes, random_label_cmap
        from stardist.models import Config2D, StarDist2D
        from csbdeep.utils import normalize
        
        # Load data
        if images is None or masks is None:
            images, masks = self.load_data()

        # Clean up labels (fill holes)
        print("\n🔧 Preprocessing labels...")
        masks = [fill_label_holes(mask) for mask in masks]
        
        # Train/validation split
        n_val = max(1, int(len(images) * self.config.validation_split))
        X_train, Y_train = images[:-n_val], masks[:-n_val]
        X_val, Y_val = images[-n_val:], masks[-n_val:]
        
        print(f"   Train: {len(X_train)}, Validation: {len(X_val)}")
        
        # Model configuration
        print("\n🏗️ Configuring model...")
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
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create model
        model = StarDist2D(
            conf,
            name=self.config.model_name,
            basedir=str(self.config.output_dir),
        )
        
        # Augmentation setup
        augmenter = None
        if self.config.use_augmentation:
            from csbdeep.data import Normalizer, normalize_mi_ma
            
            def simple_augmenter(x, y):
                """Simple data augmentation (rotation, flip)"""
                # Random rotation (0, 90, 180, 270 degrees)
                k = np.random.randint(0, 4)
                x = np.rot90(x, k)
                y = np.rot90(y, k)
                
                # Random horizontal flip
                if np.random.rand() > 0.5:
                    x = np.fliplr(x)
                    y = np.fliplr(y)
                
                # Random vertical flip
                if np.random.rand() > 0.5:
                    x = np.flipud(x)
                    y = np.flipud(y)
                
                return x, y
            
            augmenter = simple_augmenter
        
        # Training
        print(f"\n🚀 Starting StarDist training...")
        print(f"   Model save path: {self.config.output_dir / self.config.model_name}")
        
        # Stack data into numpy arrays (stardist requirement)
        X_train_arr = np.array(X_train)
        Y_train_arr = np.array(Y_train)
        X_val_arr = np.array(X_val)
        Y_val_arr = np.array(Y_val)
        
        # Add channel dimension (if needed)
        if X_train_arr.ndim == 3:
            X_train_arr = X_train_arr[..., np.newaxis]
        if X_val_arr.ndim == 3:
            X_val_arr = X_val_arr[..., np.newaxis]
        
        model.train(
            X_train_arr, Y_train_arr,
            validation_data=(X_val_arr, Y_val_arr),
            augmenter=augmenter,
        )
        
        # Threshold optimization
        print("\n🔧 Optimizing thresholds...")
        model.optimize_thresholds(X_val_arr, Y_val_arr)
        
        model_path = self.config.output_dir / self.config.model_name
        print(f"\n✅ Training complete!")
        print(f"   Model path: {model_path}")
        
        return model_path
    
    def get_model_path(self) -> Path:
        """Return path to the trained model"""
        return self.config.output_dir / self.config.model_name


def create_labeled_data_from_rule_based(
    xqd_files: List[Path],
    output_dir: Path,
    review_with_napari: bool = True,
) -> None:
    """
    Generate initial label data using rule-based segmentation

    This function creates initial masks using a rule-based method,
    allowing review/correction with napari.

    Parameters:
        xqd_files: List of XQD file paths
        output_dir: Output directory (creates images/, masks/ subfolders)
        review_with_napari: Whether to review with napari

    Example:
        from pathlib import Path
        xqd_files = list(Path("./input_data/xqd").glob("*.xqd"))
        create_labeled_data_from_rule_based(
            xqd_files,
            Path("./labeled_data"),
            review_with_napari=True,
        )
    """
    from grain_analyzer import AFMData, segment_advanced

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Processing {len(xqd_files)} files...")

    for xqd_file in xqd_files:
        try:
            # Load AFM data
            data = AFMData(str(xqd_file))
            data.first_correction().second_correction().third_correction()
            data.flat_correction("line_by_line").baseline_correction("min_to_zero")

            height = data.get_data()
            meta = data.get_meta_data()

            # Normalize
            pmin, pmax = np.percentile(height, [1, 99])
            height_norm = np.clip((height - pmin) / (pmax - pmin), 0, 1).astype(np.float32)

            # Advanced segmentation
            labels = segment_advanced(height, meta)
            
            # Save
            img_path = images_dir / f"{xqd_file.stem}.npy"
            mask_path = masks_dir / f"{xqd_file.stem}.npy"
            
            np.save(img_path, height_norm)
            np.save(mask_path, labels)
            
            print(f"   ✓ {xqd_file.name}: {labels.max()} objects")
            
        except Exception as e:
            print(f"   ❌ {xqd_file.name}: {e}")
    
    print(f"\n✅ Initial label data generation complete: {output_dir}")

    if review_with_napari:
        print("\n📝 It is recommended to review/edit labels with napari:")
        print("   pip install napari[all]")
        print("   napari")
        print(f"   - Image folder: {images_dir}")
        print(f"   - Mask folder: {masks_dir}")
        print("\n   After editing masks in napari, save them to the same path.")

