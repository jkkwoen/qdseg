"""
Cellpose Trainer for AFM Grain Images

Module for training Cellpose models on custom datasets.
Requires labeled masks as this is supervised learning.

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
    """Cellpose training configuration"""

    # Data paths
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    output_dir: Path = field(default_factory=lambda: Path("./models/cellpose"))
    
    # Training parameters
    n_epochs: int = 100
    batch_size: int = 8
    learning_rate: float = 0.2  # Cellpose default
    weight_decay: float = 1e-5
    
    # Model parameters
    model_type: str = "cyto3"  # Base model ('cyto', 'cyto2', 'cyto3', 'nuclei')

    # Augmentation
    min_train_masks: int = 1  # Minimum number of objects

    # Misc
    model_name: str = "cellpose_afm"
    use_gpu: bool = True
    
    # Model architecture
    nclasses: int = 3  # Default (cytoplasm + nuclei + background)


class CellposeTrainer:
    """
    Cellpose model training class

    The following data is required for supervised learning:
    - images/: Original AFM images (npy, tiff, png, etc.)
    - masks/: Label masks (each object marked with a unique integer)

    Example:
        config = CellposeConfig(
            data_dir=Path("./labeled_data"),
            output_dir=Path("./models/cellpose_afm"),
        )
        trainer = CellposeTrainer(config)

        # Load data
        images, masks = trainer.load_data()

        # Train
        trainer.train(images, masks)

        # Model path
        print(f"Model saved to: {trainer.get_model_path()}")
    """
    
    def __init__(self, config: CellposeConfig):
        self.config = config
        self._check_dependencies()
        self.device = None
        
    def _check_dependencies(self):
        """Check required packages"""
        try:
            import cellpose
            import torch
        except ImportError as e:
            raise ImportError(
                "cellpose is required for Cellpose training.\n"
                "Install: pip install cellpose\n"
                f"Error: {e}"
            )
    
    def _setup_device(self):
        """Setup device"""
        import torch
        
        if not self.config.use_gpu:
            self.device = torch.device('cpu')
            print("   Device: CPU")
            return

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"   Device: CUDA ({torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("   Device: MPS (Apple Silicon)")
        else:
            self.device = torch.device('cpu')
            print("   Device: CPU")
    
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
                else:
                    try:
                        from tifffile import imread as tiff_imread
                        if img_path.suffix in ['.tiff', '.tif']:
                            img = tiff_imread(img_path)
                        else:
                            img = imread(img_path)
                    except ImportError:
                        img = imread(img_path)
                
                # Load mask
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
                
                # Normalize (Cellpose prefers 0-255 range)
                if img.max() <= 1:
                    img = (img * 255).astype(np.uint8)
                elif img.max() > 255:
                    pmin, pmax = np.percentile(img, [1, 99])
                    img = np.clip((img - pmin) / (pmax - pmin + 1e-10) * 255, 0, 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
                
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
        Train Cellpose model

        Parameters:
            images: List of images (calls load_data() if not provided)
            masks: List of masks

        Returns:
            model_path: Path to the trained model
        """
        from cellpose import models, train
        
        # Load data
        if images is None or masks is None:
            images, masks = self.load_data()

        # Device setup
        print("\n🔧 Setting up environment...")
        self._setup_device()
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Train/test split (Cellpose does not require a separate validation set)
        print(f"\n🚀 Starting Cellpose training...")
        print(f"   Base model: {self.config.model_type}")
        print(f"   Epochs: {self.config.n_epochs}")
        print(f"   Batch size: {self.config.batch_size}")
        print(f"   Learning rate: {self.config.learning_rate}")
        
        # Load base model
        model = models.CellposeModel(
            model_type=self.config.model_type,
            gpu=self.config.use_gpu,
            device=self.device,
        )
        
        # Model file path
        model_path = self.config.output_dir / f"{self.config.model_name}"
        
        # Training (Cellpose 4.x API)
        # Use Cellpose's train function
        new_model_path, train_losses, test_losses = train.train_seg(
            model.net,
            train_data=images,
            train_labels=masks,
            test_data=None,
            test_labels=None,
            channels=[0, 0],  # Grayscale
            save_path=str(self.config.output_dir),
            save_every=10,
            n_epochs=self.config.n_epochs,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            batch_size=self.config.batch_size,
            min_train_masks=self.config.min_train_masks,
            model_name=self.config.model_name,
        )
        
        print(f"\n✅ Training complete!")
        print(f"   Model path: {new_model_path}")
        print(f"   Final training loss: {train_losses[-1]:.4f}")
        
        return Path(new_model_path)
    
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
    from qdseg import AFMData, segment

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

            # Prepare image for Cellpose (0-255 range)
            pmin, pmax = np.percentile(height, [1, 99])
            height_uint8 = np.clip((height - pmin) / (pmax - pmin) * 255, 0, 255).astype(np.uint8)

            # Advanced segmentation
            labels = segment(height, meta)
            
            # Save
            img_path = images_dir / f"{xqd_file.stem}.npy"
            mask_path = masks_dir / f"{xqd_file.stem}.npy"
            
            np.save(img_path, height_uint8)
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

