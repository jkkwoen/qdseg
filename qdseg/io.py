"""
XQD/XQF IO utilities
--------------------
Provides I/O functions including header parsing, RAW loading, and height (nm) conversion.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Dict, List, Tuple
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

_NANOSCOPE_SCALE_DENOMINATOR = 65536.0
_LENGTH_UNITS_TO_NM = {
    "m": 1e9,
    "meter": 1e9,
    "meters": 1e9,
    "um": 1000.0,
    "µm": 1000.0,
    "micron": 1000.0,
    "microns": 1000.0,
    "nm": 1.0,
    "nanometer": 1.0,
    "nanometers": 1.0,
    "a": 0.1,
    "ang": 0.1,
    "angstrom": 0.1,
    "angstroms": 0.1,
}


def _read_from_header(buf: bytes, key: str):
    off, _, fmt = O[key]
    return struct.unpack_from(fmt, buf, off)[0]


def _resolve_dimensions(header: bytes, path: Path | str | None = None) -> Tuple[int, int]:
    """Determine consistent (xres, yres) using the header and (optionally) file size.

    - Primary xres: 0x0056, yres: 0x057C
    - Alternative xres candidate: 0x057A (some files store the actual xres here)
    - File-size-based RAW pixel count validation: (filesize - HDR_SIZE) / 2
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
    Read the XQD/XQF header and return it as a dict (in nm units).
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
    Read the RAW count image after the header and return it (dtype=uint16, shape=(yres, xres)).
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
    Convert RAW counts to a height array in nm units.
    height_nm = raw * zscale_nm_per_count + zoffset_nm
    """
    import numpy as np
    zscale_nm = float(header.get('zscale_nm_per_count'))
    zoffset_nm = float(header.get('zoffset_nm', 0.0))
    return raw_counts.astype(np.float64) * (zscale_nm) + (zoffset_nm)


def _normalize_nanoscope_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", key.lower())


def _parse_nanoscope_value(value: str):
    value = value.strip()
    if not value:
        return ""

    numbers = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        value,
    )
    quoted = re.findall(r'"([^"]+)"', value)
    if quoted:
        return quoted[-1]
    if len(numbers) == 1 and not re.search(r"[A-Za-zµÅ]", value):
        number = numbers[0]
        return float(number) if "." in number or "e" in number.lower() else int(number)
    return value


def _split_nanoscope_param(line: str) -> Tuple[str, str] | None:
    body = line.strip().lstrip("\\").strip()
    if not body:
        return None

    if body.startswith("@"):
        body = body[1:].strip()
        body = re.sub(r"^\d+\s*:\s*", "", body)

    if ":" not in body:
        return None

    key, value = body.split(":", 1)
    return key.strip(), value.strip()


def _parse_nanoscope_sections(file_bytes: bytes) -> List[Dict]:
    text = file_bytes.decode("latin-1", errors="ignore")
    sections: List[Dict] = []
    current: Dict | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip("\x00\r\n ")
        if not line:
            continue

        if line.startswith("\\*"):
            current = {
                "name": line.lstrip("\\*").strip(),
                "items": {},
                "raw_items": {},
            }
            sections.append(current)
            continue

        if current is None or not line.startswith("\\"):
            continue

        parsed = _split_nanoscope_param(line)
        if parsed is None:
            continue

        key, value = parsed
        normalized = _normalize_nanoscope_key(key)
        current["raw_items"][key] = value
        current["items"][normalized] = _parse_nanoscope_value(value)

    return sections


def _get_nanoscope_item(section: Dict, *keys: str):
    items = section.get("items", {})
    for key in keys:
        normalized = _normalize_nanoscope_key(key)
        if normalized in items:
            return items[normalized]
    return None


def _parse_length_nm(value, default_unit: str = "nm") -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) * _LENGTH_UNITS_TO_NM.get(default_unit, 1.0)

    text = str(value).replace("Âµ", "µ").replace("~m", "um")
    matches = re.findall(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-zµÅ]+)?",
        text,
    )
    if not matches:
        return None

    number, unit = matches[-1]
    unit = (unit or default_unit).replace("Å", "angstrom").lower()
    if unit not in _LENGTH_UNITS_TO_NM:
        unit = default_unit
    return float(number) * _LENGTH_UNITS_TO_NM[unit]


def _nanoscope_channel_name(section: Dict) -> str:
    raw_items = section.get("raw_items", {})
    for key, value in raw_items.items():
        if _normalize_nanoscope_key(key) in {"imagedata", "imagedata2"}:
            quoted = re.findall(r'"([^"]+)"', value)
            if quoted:
                return quoted[-1]
            bracketed = re.findall(r"\[([^\]]+)\]", value)
            if bracketed:
                return bracketed[-1]
    return ""


def _nanoscope_image_sections(sections: List[Dict]) -> List[Dict]:
    return [
        section for section in sections
        if "ciao image list" in section.get("name", "").lower()
        and _get_nanoscope_item(section, "Data offset") is not None
    ]


def _select_nanoscope_image_section(sections: List[Dict], channel: str | None) -> Dict:
    image_sections = _nanoscope_image_sections(sections)
    if not image_sections:
        raise ValueError("No NanoScope image channel with a Data offset was found")

    if channel:
        channel_l = channel.lower()
        for section in image_sections:
            if channel_l in _nanoscope_channel_name(section).lower():
                return section
        available = [
            _nanoscope_channel_name(section) or "<unnamed>"
            for section in image_sections
        ]
        raise ValueError(f"NanoScope channel {channel!r} not found. Available: {available}")

    for section in image_sections:
        if "height" in _nanoscope_channel_name(section).lower():
            return section
    return image_sections[0]


def read_nanoscope_header(
    path: str | Path,
    *,
    channel: str | None = None,
) -> Dict:
    """Read Bruker/Veeco NanoScope metadata for an image channel.

    The parser follows the common NanoScope text-header fields also used by
    tools such as Gwyddion: ``Data offset``, ``Data length``, ``Samps/line``,
    ``Number of lines`` and ``Bytes/pixel``.
    """
    p = Path(path)
    file_bytes = p.read_bytes()
    sections = _parse_nanoscope_sections(file_bytes)
    image = _select_nanoscope_image_section(sections, channel)

    xres = int(_get_nanoscope_item(image, "Samps/line"))
    yres = int(_get_nanoscope_item(image, "Number of lines"))
    data_offset = int(_get_nanoscope_item(image, "Data offset"))
    data_length = int(_get_nanoscope_item(image, "Data length") or 0)
    bytes_per_pixel = int(_get_nanoscope_item(image, "Bytes/pixel") or 2)
    channel_name = _nanoscope_channel_name(image)

    scan_size_nm = None
    for section in [image] + sections:
        scan_size_nm = _parse_length_nm(_get_nanoscope_item(section, "Scan Size"))
        if scan_size_nm is not None:
            break

    px_nm = scan_size_nm / xres if scan_size_nm else 1.0
    py_nm = scan_size_nm / yres if scan_size_nm else px_nm

    zscale_nm = None
    for section in [image] + sections:
        zscale_nm = _parse_length_nm(
            _get_nanoscope_item(section, "Z scale"),
            default_unit="um",
        )
        if zscale_nm is not None:
            break

    if zscale_nm is None:
        zscale_nm = _NANOSCOPE_SCALE_DENOMINATOR

    return {
        "signature": "Bruker NanoScope",
        "format": "nanoscope",
        "source_path": str(p),
        "channel": channel_name,
        "xres": xres,
        "yres": yres,
        "pixel_nm": (px_nm, py_nm),
        "scan_size_nm": (px_nm * xres, py_nm * yres),
        "zscale_nm_per_count": zscale_nm / _NANOSCOPE_SCALE_DENOMINATOR,
        "zoffset_nm": 0.0,
        "pixel_um": (px_nm / 1000.0, py_nm / 1000.0),
        "scan_size_um": (px_nm * xres / 1000.0, py_nm * yres / 1000.0),
        "zscale_um_per_count": zscale_nm / _NANOSCOPE_SCALE_DENOMINATOR / 1000.0,
        "zoffset_um": 0.0,
        "data_offset": data_offset,
        "data_length": data_length,
        "bytes_per_pixel": bytes_per_pixel,
    }


def read_nanoscope_raw(
    path: str | Path,
    *,
    channel: str | None = None,
) -> "numpy.ndarray":
    """Read a Bruker/Veeco NanoScope image channel as signed integer counts."""
    import numpy as np

    p = Path(path)
    header = read_nanoscope_header(p, channel=channel)
    dtype_map = {
        1: np.dtype("i1"),
        2: np.dtype("<i2"),
        4: np.dtype("<i4"),
    }
    bytes_per_pixel = int(header["bytes_per_pixel"])
    if bytes_per_pixel not in dtype_map:
        raise ValueError(f"Unsupported NanoScope bytes per pixel: {bytes_per_pixel}")

    count = int(header["xres"]) * int(header["yres"])
    with p.open("rb") as f:
        f.seek(int(header["data_offset"]))
        raw_counts = np.fromfile(f, dtype=dtype_map[bytes_per_pixel], count=count)

    if raw_counts.size != count:
        raise ValueError(
            f"NanoScope image data is truncated: expected {count} pixels, "
            f"got {raw_counts.size}"
        )
    return raw_counts.reshape(int(header["yres"]), int(header["xres"]))


def load_nanoscope_height_nm(
    path: str | Path,
    *,
    channel: str | None = None,
) -> Tuple["numpy.ndarray", Dict]:
    """Load a Bruker/Veeco NanoScope image channel as height in nm."""
    import numpy as np

    header = read_nanoscope_header(path, channel=channel)
    raw = read_nanoscope_raw(path, channel=channel)
    height_nm = raw.astype(np.float64) * float(header["zscale_nm_per_count"])
    height_nm = np.flipud(height_nm)
    return height_nm, header


def load_height_nm(path: str | Path) -> Tuple["numpy.ndarray", Dict]:
    """
    Load the height (nm) array and header together from a supported AFM file.
    Returns the array with the Y-axis flipped (flipud).
    """
    suffix = Path(path).suffix.lower()
    if suffix in {".spm"} or re.fullmatch(r"\.\d{3}", suffix):
        return load_nanoscope_height_nm(path)

    header = read_xqd_header(path, processed=True)
    raw = read_xqd_raw(path)
    height_nm = raw_to_height_nm(raw, header)
    
    # Flip the Y-axis (upside-down inversion)
    import numpy as np
    height_nm = np.flipud(height_nm)
    
    return height_nm, header
