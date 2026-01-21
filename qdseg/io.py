"""
XQD/XQF IO utilities
--------------------
헤더 해석, RAW 로드, 높이(nm) 변환 등 입출력 전용 기능을 제공합니다.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type hints only, no runtime dependency
    import numpy


# ────────── Fixed offsets / formats (from parser) ──────────
HDR_SIZE: int = 0x0B80  # 2 944 B

O = {
    'sig':   (0x0000, 12,   '12s'),   # File signature
    'xres':  (0x0056, 2,    '<H'),    # X pixel count #1
    'yres':  (0x057C, 2,    '<H'),    # Y pixel count
    'xp_nm': (0x0098, 8,    '<d'),    # 1 pixel X length [nm]
    'yp_nm': (0x00A0, 8,    '<d'),    # 1 pixel Y length [nm]
    'z_nm':  (0x00A8, 8,    '<d'),    # 1 count Z length [nm]
    'z_off': (0x00E0, 8,    '<d'),    # Z offset (header value needs sign inversion)
}


def _read_from_header(buf: bytes, key: str):
    off, _, fmt = O[key]
    return struct.unpack_from(fmt, buf, off)[0]


def _resolve_dimensions(header: bytes, path: Path | str | None = None) -> Tuple[int, int]:
    """헤더와(선택적으로) 파일 크기를 이용해 일관된 (xres, yres)를 산출한다.

    - 기본 xres: 0x0056, yres: 0x057C
    - 대체 xres 후보: 0x057A (일부 파일에서 실제 xres가 여기에 존재)
    - 파일 크기 기반 RAW 픽셀 수 검증: (filesize - HDR_SIZE) / 2
    """
    primary_xres = _read_from_header(header, 'xres')
    yres = _read_from_header(header, 'yres')

    # Alternative xres candidate
    try:
        alt_xres = struct.unpack_from('<H', header, 0x057A)[0]
    except Exception:
        alt_xres = None

    # If file available, validate with raw pixel count
    raw_pixels = None
    if path is not None:
        p = Path(path)
        try:
            size = p.stat().st_size
            raw_bytes = size - HDR_SIZE
            if raw_bytes > 0 and raw_bytes % 2 == 0:
                raw_pixels = raw_bytes // 2
        except FileNotFoundError:
            pass

    xres = primary_xres
    if raw_pixels is not None and yres > 0:
        # If primary doesn't match, try alternative
        if primary_xres * yres != raw_pixels:
            if alt_xres and alt_xres * yres == raw_pixels:
                xres = alt_xres
            else:
                # Fallback: deduce xres from raw_pixels and yres
                if raw_pixels % yres == 0:
                    xres = raw_pixels // yres

    return int(xres), int(yres)


def read_xqd_header(path: str | Path, *, processed: bool = True) -> Dict:
    """
    XQD/XQF 헤더를 읽어 dict로 반환 (nm 기준)
    """
    p = Path(path)
    with p.open('rb') as f:
        header = f.read(HDR_SIZE)

    # raw values
    sig_raw = _read_from_header(header, 'sig')
    xres, yres = _resolve_dimensions(header, p)
    xp_nm = _read_from_header(header, 'xp_nm')
    yp_nm = _read_from_header(header, 'yp_nm')
    z_nm = _read_from_header(header, 'z_nm')
    z_off = _read_from_header(header, 'z_off')

    if not processed:
        return {
            'signature_raw': sig_raw.decode('ascii', 'replace'),
            'xres': xres,
            'yres': yres,
            'xp_nm': xp_nm,
            'yp_nm': yp_nm,
            'z_nm': z_nm,
            'z_off': z_off,
        }

    # processed / convenient values (nm)
    signature = sig_raw.decode('ascii', 'replace')
    zoffset_nm = -z_off  # sign inversion
    scan_size_nm: Tuple[float, float] = (xp_nm * xres, yp_nm * yres)

    return {
        'signature': signature,
        'xres': xres,
        'yres': yres,
        'pixel_nm': (xp_nm, yp_nm),
        'scan_size_nm': scan_size_nm,
        'zscale_nm_per_count': z_nm,
        'zoffset_nm': zoffset_nm,
        # Derived µm values for convenience
        'pixel_um': (xp_nm / 1000.0, yp_nm / 1000.0),
        'scan_size_um': (scan_size_nm[0] / 1000.0, scan_size_nm[1] / 1000.0),
        'zscale_um_per_count': z_nm / 1000.0,
        'zoffset_um': zoffset_nm / 1000.0,
    }


def read_xqd_raw(path: str | Path) -> "numpy.ndarray":
    """
    헤더 이후의 RAW 카운트 이미지를 읽어 반환 (dtype=uint16, shape=(yres, xres))
    """
    import numpy as np  # Lazy import to avoid numpy dependency when reading header only
    p = Path(path)
    with p.open('rb') as f:
        header = f.read(HDR_SIZE)
        xres, yres = _resolve_dimensions(header, p)
        raw_counts = np.fromfile(f, dtype='<u2', count=xres * yres)

    return raw_counts.reshape(yres, xres)


def raw_to_height_nm(raw_counts: "numpy.ndarray", header: Dict) -> "numpy.ndarray":
    """
    RAW 카운트를 nm 단위 높이 배열로 변환한다.
    height_nm = raw * zscale_nm_per_count + zoffset_nm
    """
    import numpy as np
    zscale_nm = float(header.get('zscale_nm_per_count'))
    zoffset_nm = float(header.get('zoffset_nm', 0.0))
    return raw_counts.astype(np.float64) * (zscale_nm) + (zoffset_nm)


def load_height_nm(path: str | Path) -> Tuple["numpy.ndarray", Dict]:
    """
    파일로부터 높이(nm) 배열과 헤더를 함께 로드한다.
    Y축을 뒤집어서 반환한다 (flipud).
    """
    header = read_xqd_header(path, processed=True)
    raw = read_xqd_raw(path)
    height_nm = raw_to_height_nm(raw, header)
    
    # Y축을 뒤집기 (위아래 반전)
    import numpy as np
    height_nm = np.flipud(height_nm)
    
    return height_nm, header

