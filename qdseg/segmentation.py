"""
Grain Segmentation Methods

각 함수는 동일한 입출력 형식을 따름:
    Input: height (2D array), meta (dict)
    Output: labels (2D int array, 0=background, 1,2,3...=grains)

사용 예시:
    >>> from grain_analyzer.segmentation import segment_rule_based, segment_stardist, segment_cellpose
    >>> labels = segment_rule_based(height, meta)
    >>> labels = segment_stardist(height, meta)
    >>> labels = segment_cellpose(height, meta)
"""

import warnings
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.feature import peak_local_max


def segment_rule_based(
    height: np.ndarray,
    meta: Optional[Dict] = None,
    *,
    gaussian_sigma: float = 1.0,
    min_area_px: int = 10,
    min_peak_separation_nm: float = 10.0,
) -> np.ndarray:
    """
    Rule-based segmentation: Otsu + Distance Transform + Voronoi
    
    Parameters
    ----------
    height : np.ndarray
        Height image (2D, nm units)
    meta : dict, optional
        Metadata with 'pixel_nm' key for pixel size in nm
    gaussian_sigma : float
        Gaussian smoothing sigma (default: 1.0)
    min_area_px : int
        Minimum grain area in pixels (default: 10)
    min_peak_separation_nm : float
        Minimum separation between peaks in nm (default: 10.0)
    
    Returns
    -------
    labels : np.ndarray
        Label image (int32, 0=background, 1,2,3...=grains)
    
    Examples
    --------
    >>> labels = segment_rule_based(height, meta, gaussian_sigma=1.5)
    >>> num_grains = labels.max()
    """
    pixel_nm = meta.get("pixel_nm", (1.0, 1.0)) if meta else (1.0, 1.0)
    xp_nm, yp_nm = float(pixel_nm[0]), float(pixel_nm[1])
    
    # 1. Smoothing + Thresholding
    h_smooth = gaussian(height, sigma=gaussian_sigma, preserve_range=True)
    binary = h_smooth > threshold_otsu(h_smooth)
    
    # Remove small objects (suppress deprecation warning for min_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        binary = remove_small_objects(binary, min_size=min_area_px)
    
    if not np.any(binary):
        return np.zeros(height.shape, dtype=np.int32)
    
    # 2. Distance transform and peak detection
    distance = ndi.distance_transform_edt(binary)
    avg_px_nm = float(np.mean(pixel_nm)) if np.all(np.isfinite(pixel_nm)) else 1.0
    min_dist_px = max(1, int(round((min_peak_separation_nm / avg_px_nm) / 2.0)))
    
    coords = peak_local_max(
        distance,
        labels=binary,
        min_distance=min_dist_px,
        exclude_border=False,
    )
    
    if coords.size == 0:
        return np.zeros(height.shape, dtype=np.int32)
    
    # 3. Cluster nearby peaks using DBSCAN
    try:
        from sklearn.cluster import DBSCAN
        
        coords_nm = np.column_stack([coords[:, 0] * yp_nm, coords[:, 1] * xp_nm])
        clustering = DBSCAN(eps=min_peak_separation_nm, min_samples=1).fit(coords_nm)
        
        rep_coords = []
        for lab in np.unique(clustering.labels_):
            idxs = np.where(clustering.labels_ == lab)[0]
            best = idxs[np.argmax(distance[coords[idxs, 0], coords[idxs, 1]])]
            rep_coords.append(coords[best])
        rep_coords = np.array(rep_coords)
    except ImportError:
        # sklearn 없으면 모든 좌표 사용
        rep_coords = coords
    
    # 4. Voronoi segmentation from markers
    labels = _voronoi_from_markers(height.shape, rep_coords, binary, pixel_nm)
    
    return labels


def segment_stardist(
    height: np.ndarray,
    meta: Optional[Dict] = None,
    *,
    model_name: str = "2D_versatile_fluo",
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.4,
    model_path: Optional[str] = None,
    normalize_input: bool = True,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    StarDist-based segmentation using deep learning
    
    Uses star-convex polygon detection for grain segmentation.
    Requires stardist and tensorflow to be installed.
    
    GPU 가속은 환경에 따라 자동 감지됩니다:
    - NVIDIA GPU: CUDA (tensorflow-gpu 필요)
    - Apple Silicon: Metal (tensorflow-metal 필요)
    - 그 외: CPU
    
    Parameters
    ----------
    height : np.ndarray
        Height image (2D, nm units)
    meta : dict, optional
        Metadata (not used, for API consistency)
    model_name : str
        Pretrained model name (default: '2D_versatile_fluo')
        Options: '2D_versatile_fluo', '2D_versatile_he', '2D_paper_dsb2018'
    prob_thresh : float
        Probability threshold for detection (0-1, default: 0.5)
    nms_thresh : float
        Non-maximum suppression threshold (0-1, default: 0.4)
    model_path : str, optional
        Path to custom trained model directory
    normalize_input : bool
        Whether to normalize input image (default: True)
    use_gpu : bool
        Whether to use GPU if available (default: True)
        GPU type is auto-detected (CUDA/Metal/CPU)
    
    Returns
    -------
    labels : np.ndarray
        Label image (int32, 0=background, 1,2,3...=grains)
    
    Raises
    ------
    ImportError
        If stardist or tensorflow is not installed
    
    Examples
    --------
    >>> labels = segment_stardist(height, prob_thresh=0.6)
    >>> # With custom model
    >>> labels = segment_stardist(height, model_path="./my_model")
    >>> # CPU only
    >>> labels = segment_stardist(height, use_gpu=False)
    
    Notes
    -----
    - AFM height images may require custom trained models for best results
    - Pretrained models are optimized for fluorescence microscopy images
    - Install with: pip install stardist tensorflow
    - For Apple Silicon: pip install tensorflow-metal
    - For NVIDIA GPU: pip install tensorflow[cuda]
    
    References
    ----------
    StarDist: https://github.com/stardist/stardist
    """
    # Lazy import - StarDist 미설치 시 다른 함수는 사용 가능
    try:
        from stardist.models import StarDist2D
    except ImportError:
        raise ImportError(
            "StarDist가 설치되지 않았습니다. "
            "설치: pip install stardist tensorflow"
        )
    
    # GPU 환경 설정
    if use_gpu:
        try:
            from .utils import setup_gpu_environment, check_tensorflow_gpu
            setup_gpu_environment()
            check_tensorflow_gpu(verbose=False)
        except ImportError:
            pass
    
    # Load model
    if model_path:
        # 커스텀 모델 로드
        # model_path가 디렉토리 경로이면, 그 안의 모델을 로드
        # 예: model_path="/path/to/models/my_model" -> basedir="/path/to/models", name="my_model"
        import os
        model_dir = os.path.dirname(model_path)
        model_name_custom = os.path.basename(model_path)
        if model_dir == '':
            model_dir = '.'
        model = StarDist2D(None, name=model_name_custom, basedir=model_dir)
    else:
        model = StarDist2D.from_pretrained(model_name)
    
    # Normalize input
    if normalize_input:
        img_norm = _normalize_percentile(height, 1, 99.8)
    else:
        img_norm = height
    
    # Predict
    labels, _ = model.predict_instances(
        img_norm,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
    )
    
    return labels.astype(np.int32)


def segment_cellpose(
    height: np.ndarray,
    meta: Optional[Dict] = None,
    *,
    model_type: str = "cyto3",
    diameter: Optional[float] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    model_path: Optional[str] = None,
    normalize_input: bool = True,
    gpu: bool = True,
    device: Optional["torch.device"] = None,
) -> np.ndarray:
    """
    Cellpose-based segmentation using deep learning
    
    Uses gradient flow-based cell segmentation.
    Requires cellpose to be installed.
    
    GPU 가속은 환경에 따라 자동 감지됩니다:
    - NVIDIA GPU: CUDA
    - Apple Silicon: MPS (Metal Performance Shaders)
    - 그 외: CPU
    
    Parameters
    ----------
    height : np.ndarray
        Height image (2D, nm units)
    meta : dict, optional
        Metadata with 'pixel_nm' key (used for diameter estimation)
    model_type : str
        Model type (default: 'cyto3')
        Options: 'cyto', 'cyto2', 'cyto3', 'nuclei', 'tissuenet_cp3', etc.
    diameter : float, optional
        Expected cell diameter in pixels. If None, auto-estimated.
    flow_threshold : float
        Flow error threshold (default: 0.4)
    cellprob_threshold : float
        Cell probability threshold (default: 0.0)
    model_path : str, optional
        Path to custom trained model file (.pth or model directory)
        If provided, model_type is ignored.
    normalize_input : bool
        Whether to normalize input image (default: True)
    gpu : bool
        Whether to use GPU if available (default: True)
        GPU type is auto-detected (CUDA/MPS/CPU)
    device : torch.device, optional
        Specific device to use. If None, auto-detected.
    
    Returns
    -------
    labels : np.ndarray
        Label image (int32, 0=background, 1,2,3...=grains)
    
    Raises
    ------
    ImportError
        If cellpose is not installed
    
    Examples
    --------
    >>> labels = segment_cellpose(height, diameter=30)
    >>> labels = segment_cellpose(height, model_type='nuclei')
    >>> # With custom trained model
    >>> labels = segment_cellpose(height, model_path="./models/afm_model")
    >>> # CPU only
    >>> labels = segment_cellpose(height, gpu=False)
    >>> # Specific device
    >>> import torch
    >>> labels = segment_cellpose(height, device=torch.device('mps'))
    
    Notes
    -----
    - Cellpose is a generalist algorithm for cellular segmentation
    - 'cyto3' model is recommended for general use (but slower, uses Transformer)
    - 'cyto2' is faster with similar accuracy
    - AFM images may require diameter tuning for best results
    - Custom models can be trained with grain_analyzer.training.CellposeTrainer
    - Install with: pip install cellpose
    
    References
    ----------
    Cellpose: https://github.com/mouseland/cellpose
    """
    # Lazy import - Cellpose 미설치 시 다른 함수는 사용 가능
    try:
        from cellpose import models
    except ImportError:
        raise ImportError(
            "Cellpose가 설치되지 않았습니다. "
            "설치: pip install cellpose"
        )
    
    # 디바이스 자동 감지
    if device is None and gpu:
        try:
            from .utils import setup_gpu_environment, get_torch_device
            setup_gpu_environment()
            device = get_torch_device(verbose=False)
            gpu = device.type != 'cpu'
        except ImportError:
            # utils 없으면 기본값 사용
            pass
    
    # Load model (Cellpose 4.x uses CellposeModel)
    if model_path:
        # 커스텀 모델 로드
        model_kwargs = {'pretrained_model': model_path, 'gpu': gpu}
        if device is not None:
            model_kwargs['device'] = device
        model = models.CellposeModel(**model_kwargs)
    else:
        # 사전 학습 모델 사용
        model_kwargs = {'model_type': model_type, 'gpu': gpu}
        if device is not None:
            model_kwargs['device'] = device
        model = models.CellposeModel(**model_kwargs)
    
    # Normalize input
    if normalize_input:
        img_norm = _normalize_percentile(height, 1, 99.8)
        # Cellpose expects 0-255 range for better performance
        img_input = (img_norm * 255).astype(np.float32)
    else:
        img_input = height.astype(np.float32)
    
    # Run segmentation (Cellpose 4.x returns 3 values: masks, flows, styles)
    result = model.eval(
        img_input,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    # result[0] is masks
    masks = result[0]
    
    return masks.astype(np.int32)


def _voronoi_from_markers(
    shape: Tuple[int, int],
    markers: np.ndarray,
    mask: np.ndarray,
    pixel_nm: Tuple[float, float],
) -> np.ndarray:
    """
    Voronoi segmentation from marker coordinates
    
    Uses distance transform with anisotropic metric (nm units)
    to assign each pixel to nearest marker.
    
    Parameters
    ----------
    shape : Tuple[int, int]
        Output shape (height, width)
    markers : np.ndarray
        Marker coordinates (N, 2) as (row, col)
    mask : np.ndarray
        Binary mask limiting segmentation region
    pixel_nm : Tuple[float, float]
        Pixel size (x_nm, y_nm)
    
    Returns
    -------
    labels : np.ndarray
        Label image
    """
    h, w = shape
    labels = np.zeros((h, w), dtype=np.int32)
    
    if markers.size == 0:
        return labels
    
    xp_nm, yp_nm = float(pixel_nm[0]), float(pixel_nm[1])
    
    # Create seed mask
    seeds = np.zeros((h, w), dtype=bool)
    valid_markers = []
    for r, c in np.asarray(markers, dtype=int):
        if 0 <= r < h and 0 <= c < w:
            seeds[r, c] = True
            valid_markers.append((r, c))
    
    if not valid_markers:
        return labels
    
    # Distance transform with nm metric (anisotropic)
    sampling = (yp_nm, xp_nm)
    _, (iy, ix) = ndi.distance_transform_edt(
        ~seeds, sampling=sampling, return_indices=True
    )
    
    # Assign labels based on nearest marker
    seed_labels = np.zeros_like(labels)
    for i, (r, c) in enumerate(valid_markers, start=1):
        seed_labels[r, c] = i
    
    labels = seed_labels[iy, ix]
    
    # Apply mask
    labels = np.where(mask, labels, 0)
    
    return labels


def _normalize_percentile(
    img: np.ndarray,
    pmin: float = 1,
    pmax: float = 99.8,
) -> np.ndarray:
    """
    Percentile-based normalization to [0, 1]
    
    Parameters
    ----------
    img : np.ndarray
        Input image
    pmin : float
        Lower percentile (default: 1)
    pmax : float
        Upper percentile (default: 99.8)
    
    Returns
    -------
    np.ndarray
        Normalized image in [0, 1]
    """
    mi = np.percentile(img, pmin)
    ma = np.percentile(img, pmax)
    
    if ma - mi < 1e-10:
        return np.zeros_like(img, dtype=np.float32)
    
    return np.clip((img - mi) / (ma - mi), 0, 1).astype(np.float32)


def segment_cellulus(
    height: np.ndarray,
    meta: Optional[Dict] = None,
    *,
    model_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    normalize_input: bool = True,
    gpu: bool = True,
    device: Optional["torch.device"] = None,
    use_official: bool = True,
    # 공식 Cellulus 파라미터
    num_fmaps: int = 24,
    fmap_inc_factor: int = 3,
    p_salt_pepper: float = 0.01,
    num_infer_iterations: int = 16,
    bandwidth: Optional[float] = None,
    clustering: str = "meanshift",
    grow_distance: int = 3,
    shrink_distance: int = 6,
    min_size: Optional[int] = None,
) -> np.ndarray:
    """
    Cellulus-based segmentation using unsupervised deep learning
    
    Cellulus는 비지도 학습 기반 인스턴스 세그멘테이션 방법입니다.
    Object-centric embeddings를 학습하여 라벨 없이 세그멘테이션을 수행합니다.
    
    이 함수는 두 가지 모델 형식을 지원합니다:
    1. 공식 Cellulus 모델 (use_official=True, cellulus 패키지 필요)
    2. 간소화된 모델 (use_official=False, PyTorch만 필요)
    
    GPU 가속은 환경에 따라 자동 감지됩니다:
    - NVIDIA GPU: CUDA
    - Apple Silicon: MPS
    - 그 외: CPU
    
    Parameters
    ----------
    height : np.ndarray
        Height image (2D, nm units)
    meta : dict, optional
        Metadata (not used, for API consistency)
    model_path : str, optional
        Path to trained Cellulus model directory
    checkpoint_path : str, optional
        Path to model checkpoint file (.pth)
    normalize_input : bool
        Whether to normalize input image (default: True)
    gpu : bool
        Whether to use GPU if available (default: True)
    device : torch.device, optional
        Specific device to use. If None, auto-detected.
    use_official : bool
        Whether to use official Cellulus model (default: True)
    num_fmaps : int
        Number of feature maps in first layer (default: 24)
    fmap_inc_factor : int
        Feature map increase factor (default: 3)
    p_salt_pepper : float
        Salt-pepper noise probability for uncertainty (default: 0.01)
    num_infer_iterations : int
        Number of inference iterations (default: 16)
    bandwidth : float, optional
        MeanShift bandwidth. If None, auto-detected.
    clustering : str
        Clustering method: "meanshift" or "greedy" (default: "meanshift")
    grow_distance : int
        Morphological grow distance (default: 3)
    shrink_distance : int
        Morphological shrink distance (default: 6)
    min_size : int, optional
        Minimum object size filter. If None, no filtering.
    
    Returns
    -------
    labels : np.ndarray
        Label image (int32, 0=background, 1,2,3...=grains)
    
    Raises
    ------
    ImportError
        If neither cellulus nor torch is installed
    ValueError
        If no model path or checkpoint is provided
    
    Examples
    --------
    >>> # With official Cellulus model
    >>> labels = segment_cellulus(height, checkpoint_path="./models/best_loss.pth")
    
    >>> # With simple model (faster)
    >>> labels = segment_cellulus(height, checkpoint_path="./models/best_loss.pth", use_official=False)
    
    Notes
    -----
    - Train your model with: python train_model.py or cellulus CLI
    - Cellulus requires training on your own data (unsupervised)
    - Best suited for microscopy images with consistent object appearance
    
    References
    ----------
    Cellulus: https://github.com/funkelab/cellulus
    Paper: "Unsupervised Learning of Object-Centric Embeddings for Cell 
           Instance Segmentation in Microscopy Images" (ICCV 2023)
    """
    import torch
    
    if checkpoint_path is None and model_path is None:
        raise ValueError(
            "Cellulus는 학습된 모델이 필요합니다. "
            "checkpoint_path를 지정하세요. "
            "모델 학습: python train_model.py"
        )
    
    # 디바이스 자동 감지
    if device is None and gpu:
        try:
            from .utils import setup_gpu_environment, get_torch_device
            setup_gpu_environment()
            device = get_torch_device(verbose=False)
        except ImportError:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
    elif device is None:
        device = torch.device('cpu')
    
    # Normalize input
    if normalize_input:
        img_norm = _normalize_percentile(height, 1, 99.8)
    else:
        img_norm = height.astype(np.float32)
    
    # 공식 Cellulus 시도
    if use_official:
        try:
            device_str = str(device) if device else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
            
            labels = _segment_with_official_cellulus(
                img_norm,
                checkpoint_path,
                device_str,
                num_fmaps=num_fmaps,
                fmap_inc_factor=fmap_inc_factor,
                p_salt_pepper=p_salt_pepper,
                num_infer_iterations=num_infer_iterations,
                bandwidth=bandwidth,
                clustering=clustering,
                grow_distance=grow_distance,
                shrink_distance=shrink_distance,
                min_size=min_size,
            )
            if labels is not None:
                return labels.astype(np.int32)
        except ImportError:
            pass  # 공식 Cellulus 없으면 간소화 모델 시도
        except Exception as e:
            import traceback
            traceback.print_exc()
            pass  # 실패하면 간소화 모델 시도
    
    # 간소화된 모델 시도 (PyTorch 체크포인트)
    labels = _segment_with_simple_cellulus(img_norm, checkpoint_path, device)
    if labels is not None:
        return labels.astype(np.int32)
    
    raise RuntimeError(
        "모델을 로드할 수 없습니다. "
        "python train_model.py로 모델을 학습하세요."
    )


def _segment_with_official_cellulus(
    img_norm: np.ndarray,
    checkpoint_path: str,
    device: str,
    *,
    num_fmaps: int = 24,
    fmap_inc_factor: int = 3,
    crop_size: int = 252,
    p_salt_pepper: float = 0.01,
    num_infer_iterations: int = 16,
    bandwidth: Optional[float] = None,
    num_bandwidths: int = 1,
    reduction_probability: float = 0.1,
    clustering: str = "meanshift",
    post_processing: str = "cell",
    grow_distance: int = 3,
    shrink_distance: int = 6,
    min_size: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    공식 Cellulus CLI를 사용한 세그멘테이션
    
    공식 Cellulus의 infer 파이프라인을 호출합니다:
    1. 입력 이미지를 zarr 형식으로 변환
    2. ExperimentConfig 생성
    3. infer_experiment 호출 (predict → detect → segment)
    4. 결과 zarr에서 segmentation 로드
    
    Parameters
    ----------
    img_norm : np.ndarray
        정규화된 입력 이미지 (H, W), 값 범위 [0, 1]
    checkpoint_path : str
        학습된 Cellulus 모델 체크포인트 경로
    device : str
        디바이스 ("cuda:0", "mps", "cpu")
    num_fmaps : int
        첫 번째 레이어의 feature map 개수
    fmap_inc_factor : int
        레이어 간 feature map 증가 배수
    crop_size : int
        추론에 사용할 crop 크기
    p_salt_pepper : float
        Salt-pepper noise 비율
    num_infer_iterations : int
        추론 반복 횟수
    bandwidth : float, optional
        MeanShift bandwidth (None이면 자동)
    num_bandwidths : int
        사용할 bandwidth 개수
    reduction_probability : float
        클러스터링에 사용할 픽셀 비율
    clustering : str
        클러스터링 방법 ("meanshift" 또는 "greedy")
    post_processing : str
        후처리 방법 ("cell" 또는 "nucleus")
    grow_distance : int
        Morphological grow 거리
    shrink_distance : int
        Morphological shrink 거리
    min_size : int, optional
        최소 객체 크기 (None이면 필터링 안 함)
    
    Returns
    -------
    labels : np.ndarray or None
        세그멘테이션 결과 (H, W), 실패시 None
    """
    import os
    import tempfile
    import shutil
    
    try:
        import zarr
        import toml
        from cellulus.cli import ExperimentConfig, infer_experiment
    except ImportError:
        return None
    
    try:
        # 입력 형태 확인 및 변환
        # (H, W) -> (1, 1, H, W)
        images = img_norm[np.newaxis, np.newaxis, ...].astype(np.float32)
        
        # 모델 경로의 부모 폴더를 작업 디렉토리로 사용
        checkpoint_path_abs = os.path.abspath(checkpoint_path)
        model_dir = os.path.dirname(os.path.dirname(checkpoint_path_abs))  # models/ 의 부모
        
        # 임시 디렉토리 생성 (모델 폴더 내)
        temp_dir = tempfile.mkdtemp(prefix="cellulus_infer_", dir=model_dir)
        zarr_path = os.path.join(temp_dir, "data.zarr")
        
        # 모델 체크포인트의 상대 경로 (작업 디렉토리 기준)
        checkpoint_rel = os.path.relpath(checkpoint_path_abs, model_dir)
        
        try:
            # zarr 파일 생성
            root = zarr.open(zarr_path, mode='w')
            root['train/raw'] = images
            root['train/raw'].attrs['resolution'] = (1, 1)
            root['train/raw'].attrs['axis_names'] = ('s', 'c', 'y', 'x')
            
            # zarr 경로도 상대 경로로
            zarr_rel = os.path.relpath(zarr_path, model_dir)
            
            # TOML 설정 생성 (dict)
            config = {
                'normalization_factor': 1.0,
                'model_config': {
                    'num_fmaps': num_fmaps,
                    'fmap_inc_factor': fmap_inc_factor,
                    'checkpoint': checkpoint_rel,
                },
                'inference_config': {
                    'crop_size': [crop_size, crop_size],
                    'device': device,
                    'p_salt_pepper': p_salt_pepper,
                    'num_infer_iterations': num_infer_iterations,
                    'num_bandwidths': num_bandwidths,
                    'reduction_probability': reduction_probability,
                    'clustering': clustering,
                    'post_processing': post_processing,
                    'grow_distance': grow_distance,
                    'shrink_distance': shrink_distance,
                    'dataset_config': {
                        'container_path': zarr_rel,
                        'dataset_name': 'train/raw',
                    },
                    'prediction_dataset_config': {
                        'container_path': zarr_rel,
                        'dataset_name': 'embeddings',
                    },
                    'detection_dataset_config': {
                        'container_path': zarr_rel,
                        'dataset_name': 'detection',
                        'secondary_dataset_name': 'embeddings',
                    },
                    'segmentation_dataset_config': {
                        'container_path': zarr_rel,
                        'dataset_name': 'segmentation',
                        'secondary_dataset_name': 'detection',
                    },
                },
            }
            
            if bandwidth is not None:
                config['inference_config']['bandwidth'] = bandwidth
            if min_size is not None:
                config['inference_config']['min_size'] = min_size
            
            # 작업 디렉토리 변경 후 Cellulus 실행
            original_cwd = os.getcwd()
            os.chdir(model_dir)
            
            try:
                # ExperimentConfig 로드 및 추론 실행
                experiment_config = ExperimentConfig(**config)
                infer_experiment(experiment_config)
            finally:
                os.chdir(original_cwd)
            
            # 결과 로드
            root = zarr.open(zarr_path, mode='r')
            labels = np.array(root['segmentation'][:])  # (1, 1, H, W)
            labels = labels[0, 0, :, :]  # (H, W)
            
            return labels.astype(np.int32)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def _segment_with_simple_cellulus(
    img_norm: np.ndarray,
    checkpoint_path: str,
    device: "torch.device",
) -> Optional[np.ndarray]:
    """
    간소화된 Cellulus 모델로 세그멘테이션
    
    train_model.py로 학습된 간소화 모델을 사용합니다.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy import ndimage as ndi_local
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max as plm
    
    # 간소화된 U-Net 모델 정의
    class SimpleUNet(nn.Module):
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
    
    try:
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        
        # 모델 상태 확인
        if 'model_state_dict' not in checkpoint:
            return None  # 간소화 모델이 아님
        
        # 모델 생성 및 가중치 로드
        model = SimpleUNet().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 입력 텐서 생성
        img_tensor = torch.from_numpy(img_norm).float().unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.to(device)
        
        # 임베딩 추출
        with torch.no_grad():
            embeddings = model(img_tensor)  # (1, C, H, W)
        
        # 임베딩 -> 세그멘테이션 (클러스터링 기반)
        embeddings = embeddings.squeeze(0).cpu().numpy()  # (C, H, W)
        labels = _embeddings_to_labels(embeddings, img_norm)
        
        return labels
        
    except Exception:
        return None


def _embeddings_to_labels(embeddings: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Object-centric embeddings를 인스턴스 라벨로 변환
    
    임베딩 공간에서 유사한 픽셀들을 그룹화합니다.
    """
    from scipy import ndimage as ndi_local
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max as plm
    
    c, h, w = embeddings.shape
    
    # 임베딩 정규화
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=0, keepdims=True) + 1e-10)
    
    # 각 픽셀의 임베딩 변화량 계산 (경계 검출)
    gradient_x = np.abs(np.diff(embeddings_norm, axis=2))
    gradient_y = np.abs(np.diff(embeddings_norm, axis=1))
    
    # 패딩
    gradient_x = np.pad(gradient_x, ((0, 0), (0, 0), (0, 1)), mode='edge')
    gradient_y = np.pad(gradient_y, ((0, 0), (0, 1), (0, 0)), mode='edge')
    
    # 경계 강도
    boundary = np.sqrt(np.sum(gradient_x**2, axis=0) + np.sum(gradient_y**2, axis=0))
    
    # 경계를 기반으로 마커 생성
    smooth = ndi_local.gaussian_filter(1 - boundary, sigma=2)
    
    # 로컬 맥시마 찾기
    coords = plm(smooth, min_distance=5, threshold_abs=0.1)
    
    if len(coords) == 0:
        return np.zeros((h, w), dtype=np.int32)
    
    # 마커 이미지 생성
    markers = np.zeros((h, w), dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    
    # Watershed 세그멘테이션
    labels = watershed(boundary, markers)
    
    return labels.astype(np.int32)

