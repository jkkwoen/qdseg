"""
Bruker/Veeco NanoScope file reader.

Supports common NanoScope image blocks used by .spm and numbered
.000/.001/.002-style files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy


NANOSCOPE_SCALE_DENOMINATOR = 65536.0
LENGTH_UNITS_TO_NM = {
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


def is_nanoscope_path(path: str | Path) -> bool:
    """Return True when the extension is a known NanoScope extension."""
    suffix = Path(path).suffix.lower()
    return suffix == ".spm" or re.fullmatch(r"\.\d{3}", suffix) is not None


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", key.lower())


def _parse_value(value: str):
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


def _split_param(line: str) -> Tuple[str, str] | None:
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


def _parse_sections(file_bytes: bytes) -> List[Dict]:
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

        parsed = _split_param(line)
        if parsed is None:
            continue

        key, value = parsed
        current["raw_items"][key] = value
        current["items"][_normalize_key(key)] = _parse_value(value)

    return sections


def _get_item(section: Dict, *keys: str):
    items = section.get("items", {})
    for key in keys:
        normalized = _normalize_key(key)
        if normalized in items:
            return items[normalized]
    return None


def _parse_length_nm(value, default_unit: str = "nm") -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) * LENGTH_UNITS_TO_NM.get(default_unit, 1.0)

    text = str(value).replace("Âµ", "µ").replace("~m", "um")
    matches = re.findall(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-zµÅ]+)?",
        text,
    )
    if not matches:
        return None

    number, unit = matches[-1]
    unit = (unit or default_unit).replace("Å", "angstrom").lower()
    if unit not in LENGTH_UNITS_TO_NM:
        unit = default_unit
    return float(number) * LENGTH_UNITS_TO_NM[unit]


def _channel_name(section: Dict) -> str:
    raw_items = section.get("raw_items", {})
    for key, value in raw_items.items():
        if _normalize_key(key) in {"imagedata", "imagedata2"}:
            quoted = re.findall(r'"([^"]+)"', value)
            if quoted:
                return quoted[-1]
            bracketed = re.findall(r"\[([^\]]+)\]", value)
            if bracketed:
                return bracketed[-1]
    return ""


def _image_sections(sections: List[Dict]) -> List[Dict]:
    return [
        section for section in sections
        if "ciao image list" in section.get("name", "").lower()
        and _get_item(section, "Data offset") is not None
    ]


def _select_image_section(sections: List[Dict], channel: str | None) -> Dict:
    image_sections = _image_sections(sections)
    if not image_sections:
        raise ValueError("No NanoScope image channel with a Data offset was found")

    if channel:
        channel_l = channel.lower()
        for section in image_sections:
            if channel_l in _channel_name(section).lower():
                return section
        available = [_channel_name(section) or "<unnamed>" for section in image_sections]
        raise ValueError(f"NanoScope channel {channel!r} not found. Available: {available}")

    for section in image_sections:
        if "height" in _channel_name(section).lower():
            return section
    return image_sections[0]


def read_nanoscope_header(
    path: str | Path,
    *,
    channel: str | None = None,
) -> Dict:
    """Read Bruker/Veeco NanoScope metadata for an image channel."""
    p = Path(path)
    sections = _parse_sections(p.read_bytes())
    image = _select_image_section(sections, channel)

    xres = int(_get_item(image, "Samps/line"))
    yres = int(_get_item(image, "Number of lines"))
    data_offset = int(_get_item(image, "Data offset"))
    data_length = int(_get_item(image, "Data length") or 0)
    bytes_per_pixel = int(_get_item(image, "Bytes/pixel") or 2)
    channel_name = _channel_name(image)

    scan_size_nm = None
    for section in [image] + sections:
        scan_size_nm = _parse_length_nm(_get_item(section, "Scan Size"))
        if scan_size_nm is not None:
            break

    px_nm = scan_size_nm / xres if scan_size_nm else 1.0
    py_nm = scan_size_nm / yres if scan_size_nm else px_nm

    zscale_nm = None
    for section in [image] + sections:
        zscale_nm = _parse_length_nm(_get_item(section, "Z scale"), default_unit="um")
        if zscale_nm is not None:
            break

    if zscale_nm is None:
        zscale_nm = NANOSCOPE_SCALE_DENOMINATOR

    return {
        "signature": "Bruker NanoScope",
        "format": "nanoscope",
        "source_path": str(p),
        "channel": channel_name,
        "xres": xres,
        "yres": yres,
        "pixel_nm": (px_nm, py_nm),
        "scan_size_nm": (px_nm * xres, py_nm * yres),
        "zscale_nm_per_count": zscale_nm / NANOSCOPE_SCALE_DENOMINATOR,
        "zoffset_nm": 0.0,
        "pixel_um": (px_nm / 1000.0, py_nm / 1000.0),
        "scan_size_um": (px_nm * xres / 1000.0, py_nm * yres / 1000.0),
        "zscale_um_per_count": zscale_nm / NANOSCOPE_SCALE_DENOMINATOR / 1000.0,
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
