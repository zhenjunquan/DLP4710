#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLP4710LC / DLPC3479 dual-controller PATN generator (offline, no GUI).
Target: BOPIXEL OPR407 / TI-style Internal Pattern Streaming PATN files.

VALIDATED against an actual 26,212-byte 8-bit HORIZONTAL PATN sample:
  - PATN file header and 24-byte pattern-order records
  - Multiple Pattern Set offsets/lengths
  - 8-bit horizontal 1D bit-plane encoding (bit-for-bit roundtrip)
UNVERIFIED (requires reference BIN + EVM validation):
  - 1-bit horizontal and 1-bit/8-bit vertical encoding
The tool deliberately rejects unverified layouts unless explicitly enabled.

PATN is NOT the controller firmware. This tool does not flash hardware.
Python >=3.9; Pillow is optional (required for PNG/BMP import).
"""
import argparse
import glob
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

DMD_WIDTH = 1920
DMD_HEIGHT = 1080
MAX_ORDER_ENTRIES = 128
MAX_SET_COUNTS = {
    ("horizontal", 8): 8, ("vertical", 8): 6,
    ("horizontal", 1): 64, ("vertical", 1): 51,
}
LED_MASKS = {"none": 0, "red": 1, "green": 2, "blue": 4,
             "rg": 3, "rb": 5, "gb": 6, "rgb": 7}
DEFAULT_TIMING = {
    "pre_dark_us": 10000, "exposure_us": 4000,
    "post_dark_us": 40000, "led": "blue", "invert_mask": 0,
}
EXAMPLE = {
    "_note": "Example only. Output supports arbitrary 1D patterns, not just 3-frequency 4-step.",
    "device": "OPR407 / DLP4710LC / DLPC3479 dual",
    "auto_split": True,
    "defaults": {
        "led": "blue", "pre_dark_us": 10000,
        "exposure_us": 4000, "post_dark_us": 40000,
    },
    "groups": [
        {
            "name": "mixed_8bit_horizontal",
            "orientation": "horizontal",
            "bit_depth": 8,
            "patterns": [
                {"kind": "phase_shift", "frequencies": [70, 64, 59],
                 "phases_deg": [0, 90, 180, 270],
                 "mean": 128, "amplitude": 127},
                {"kind": "gray_code", "bits": 7, "inverse": True},
                {"kind": "ramp"},
            ],
        }
    ],
    "_order_note": "Omit 'order' for automatic playback of every created Pattern Set; "
                   "or provide an array of {set,start,count,led,exposure_us,pre_dark_us,post_dark_us,invert_mask}.",
}


def fail(message):
    raise ValueError(str(message))


def uint(value, bits, name):
    value = int(value)
    if not 0 <= value < (1 << bits):
        fail(f"{name} must be in [0, {(1 << bits) - 1}], got {value}")
    return value


def parse_led(value):
    if isinstance(value, int):
        return uint(value, 8, "LED mask")
    name = str(value).lower().replace("+", "")
    if name not in LED_MASKS:
        fail(f"Invalid LED '{value}'; valid: {list(LED_MASKS)}")
    return LED_MASKS[name]


def dims(orientation):
    if orientation == "horizontal":
        return DMD_HEIGHT
    if orientation == "vertical":
        return DMD_WIDTH
    fail(f"Invalid orientation '{orientation}'; use horizontal or vertical")


def controller_specs(orientation):
    # 1D horizontal: 1080 rows duplicated to the two controllers.
    # 1D vertical: ASSUMED (not validated): 1920 columns split into 960+960.
    if orientation == "horizontal":
        return (DMD_HEIGHT, DMD_HEIGHT)
    if orientation == "vertical":
        return (DMD_WIDTH // 2, DMD_WIDTH // 2)
    fail(f"Invalid orientation: {orientation}")


def aligned_pixels(size):
    return ((size + 15) // 16) * 16


def encode_line(values, depth):
    if depth not in (1, 8):
        fail("Only 1-bit / 8-bit supported by this OPR407 profile")
    n = len(values)
    pad = aligned_pixels(n) - n
    if depth == 1:
        # Accept both binary 0/1 arrays and 8-bit binary images (0/255).
        binary_array = all(v in (0, 1) for v in values)
        values = [1 if v != 0 else 0 for v in values] if binary_array else [
            1 if v >= 128 else 0 for v in values]
    # LSB-first bit planes; pattern is reversed and left-padded.
    pixels = [0] * pad + list(reversed(values))
    blob = bytearray()
    for bit in range(depth):
        for offset in range(0, len(pixels), 8):
            byte = 0
            for k in range(8):
                byte |= ((pixels[offset + k] >> bit) & 1) << k
            blob.append(byte)
    return bytes(blob)


def decode_line(data, logical_pixels, depth):
    padded = aligned_pixels(logical_pixels)
    stride = padded // 8
    if len(data) != stride * depth:
        fail(f"Bit-plane size mismatch: {len(data)}, expected {stride * depth}")
    pixels = [0] * padded
    for bit in range(depth):
        plane = data[bit * stride: (bit + 1) * stride]
        for i, val in enumerate(plane):
            for k in range(8):
                pixels[i * 8 + k] |= ((val >> k) & 1) << bit
    result = list(reversed(pixels[padded - logical_pixels:]))
    return [255 if x else 0 for x in result] if depth == 1 else result


def normalize_samples(values, axis_len):
    if len(values) != axis_len:
        fail(f"1D pattern has {len(values)} pixels; expected {axis_len}")
    out = []
    for i, value in enumerate(values):
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            fail(f"Invalid pixel at {i}: {value!r}")
        out.append(min(255, max(0, round(float(value)))))
    return out


def read_image_1d(path, orientation):
    try:
        from PIL import Image, ImageChops
    except ImportError:
        fail("Install Pillow to import PNG/BMP: pip install Pillow")
    n = dims(orientation)
    with Image.open(path) as src:
        if src.mode not in ("L", "1"):
            fail(f"{path}: expected grayscale image, got mode {src.mode}")
        img = src.convert("L")
        w, h = img.size
        if orientation == "horizontal" and (w, h) == (1, n):
            return list(img.getdata())
        if orientation == "vertical" and (w, h) == (n, 1):
            return list(img.getdata())
        if (w, h) == (DMD_WIDTH, DMD_HEIGHT):
            # Full-frame 1D images are supported only if every row/column repeats.
            if orientation == "horizontal":
                strip = img.crop((0, 0, 1, h))
                reference = strip.resize((w, h))
                if ImageChops.difference(img, reference).getbbox() is not None:
                    fail(f"{path}: full-frame image is not a horizontal 1D pattern")
                return list(strip.getdata())
            strip = img.crop((0, 0, w, 1))
            reference = strip.resize((w, h))
            if ImageChops.difference(img, reference).getbbox() is not None:
                fail(f"{path}: full-frame image is not a vertical 1D pattern")
            return list(strip.getdata())
    fail(f"{path}: unexpected size {w}x{h}, expected "
         + (f"1x{n}" if orientation == "horizontal" else f"{n}x1")
         + " or a repeated full 1920x1080 frame")


def load_file_values(path, orientation):
    ext = path.suffix.lower()
    if ext in (".png", ".bmp", ".pgm", ".tiff", ".tif"):
        return read_image_1d(path, orientation)
    if ext in (".txt", ".csv"):
        content = path.read_text(encoding="utf-8")
        vals = content.replace(",", " ").replace(";", " ").split()
        return [float(x) for x in vals]
    if ext == ".json":
        content = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(content, list):
            fail(f"{path}: JSON file should contain an array of pixel values")
        return content
    fail(f"Unsupported file extension {ext!r} for {path}")


def pattern_generator(spec, orientation, base_dir):
    n = dims(orientation)
    if isinstance(spec, str):
        spec = {"kind": "file", "path": spec}
    kind = spec.get("kind", "file")
    if kind == "phase_shift":
        fs = spec.get("frequencies", [])
        phases = spec.get("phases_deg", [360 * k / int(spec.get("steps", 4))
                                          for k in range(int(spec.get("steps", 4)))])
        mean = float(spec.get("mean", 128))
        amplitude = float(spec.get("amplitude", 127))
        for freq in fs:
            for phase in phases:
                theta = math.radians(float(phase))
                vals = [mean + amplitude * math.cos(
                    2 * math.pi * float(freq) * x / n + theta) for x in range(n)]
                yield f"sine_f{freq}_p{phase}", normalize_samples(vals, n)
    elif kind == "sine":
        freq, phase = float(spec.get("cycles", 1)), float(spec.get("phase_deg", 0))
        mean, amp = float(spec.get("mean", 128)), float(spec.get("amplitude", 127))
        vals = [mean + amp * math.cos(2 * math.pi * freq * x / n
                                    + math.radians(phase)) for x in range(n)]
        yield f"sine_f{freq}_p{phase}", normalize_samples(vals, n)
    elif kind in ("square", "bars"):
        cycles = float(spec.get("cycles", 16))
        phase = float(spec.get("phase_deg", 0)) / 360.0
        duty = float(spec.get("duty", 0.5))
        if not (0 < duty < 1):
            fail("Square pattern 'duty' must be strictly between 0 and 1")
        high, low = int(spec.get("high", 255)), int(spec.get("low", 0))
        vals = [high if ((x * cycles / n + phase) % 1) < duty else low
                for x in range(n)]
        yield f"square_f{cycles}", normalize_samples(vals, n)
    elif kind == "gray_code":
        bits = int(spec.get("bits", math.ceil(math.log2(n))))
        if not (1 <= bits <= 16):
            fail("gray_code bits must be 1..16")
        inverse = bool(spec.get("inverse", False))
        codes = []
        for x in range(n):
            index = min((1 << bits) - 1, (x * (1 << bits)) // n)
            codes.append(index ^ (index >> 1))
        for bit in reversed(range(bits)):
            vals = [255 if (v >> bit) & 1 else 0 for v in codes]
            yield f"gray_bit{bit}", vals
            if inverse:
                yield f"gray_bit{bit}_inverse", [255 - v for v in vals]
    elif kind == "ramp":
        low, high = float(spec.get("low", 0)), float(spec.get("high", 255))
        vals = [low + (high - low) * x / max(n - 1, 1) for x in range(n)]
        yield "ramp", normalize_samples(vals, n)
    elif kind == "constant":
        val = spec.get("value", 128)
        yield f"constant_{val}", normalize_samples([val] * n, n)
    elif kind == "values":
        yield str(spec.get("name", "values")), normalize_samples(spec["data"], n)
    elif kind == "file":
        path = (base_dir / spec["path"]).resolve()
        yield path.stem, normalize_samples(load_file_values(path, orientation), n)
    elif kind == "files":
        paths = sorted(glob.glob(str(base_dir / spec["glob"])))
        if not paths:
            fail(f"No files matched glob {spec['glob']!r}")
        for name in paths:
            path = Path(name)
            yield path.stem, normalize_samples(load_file_values(path, orientation), n)
    else:
        fail(f"Unsupported pattern kind: {kind}")


def encode_set(patterns, orientation, depth):
    """Return a complete PATN Pattern Set record, including its 8-byte header."""
    if not patterns:
        fail("Pattern Set cannot be empty")
    if (orientation, depth) not in MAX_SET_COUNTS:
        fail("Unsupported direction/bit depth combination")
    if len(patterns) > MAX_SET_COUNTS[orientation, depth]:
        fail(f"{orientation} {depth}-bit set exceeds maximum "
             f"{MAX_SET_COUNTS[orientation, depth]}")
    n = dims(orientation)
    for p in patterns:
        if len(p) != n:
            fail(f"Pattern length should be {n}, got {len(p)}")
    if orientation == "horizontal":
        ctrl0 = [encode_line(p, depth) for p in patterns]
        payload = b"".join(ctrl0) + b"".join(ctrl0)
    else:
        # Assumption only: two 960-pixel halves arranged as controllers 0 and 1.
        payload = (b"".join(encode_line(p[:960], depth) for p in patterns)
                   + b"".join(encode_line(p[960:], depth) for p in patterns))
    return struct.pack("<HHI", len(patterns), depth, len(payload)) + payload


def infer_orientation(pattern_count, depth, payload_size):
    possible = []
    for orient in ("horizontal", "vertical"):
        sizes = controller_specs(orient)
        size = sum(aligned_pixels(k) // 8 * depth * pattern_count for k in sizes)
        if size == payload_size:
            possible.append(orient)
    if not possible:
        fail(f"Unknown PATN payload layout: count={pattern_count} "
             f"depth={depth} bytes={payload_size}")
    if len(possible) > 1:
        fail("PATN orientation ambiguous for this payload")
    return possible[0]


def decode_set(record, force_orientation=None):
    if len(record) < 8:
        fail("Set record is truncated")
    count, depth, declared = struct.unpack_from("<HHI", record, 0)
    if declared != len(record) - 8:
        fail("Set payload length mismatch")
    orient = force_orientation or infer_orientation(count, depth, declared)
    data = record[8:]
    size0, size1 = controller_specs(orient)
    stride0 = aligned_pixels(size0) // 8 * depth
    stride1 = aligned_pixels(size1) // 8 * depth
    ctrl0 = data[: count * stride0]
    ctrl1 = data[count * stride0:]
    if len(ctrl1) != count * stride1:
        fail("Bad controller payload lengths")
    patterns = []
    for index in range(count):
        a = decode_line(ctrl0[index * stride0:(index + 1) * stride0], size0, depth)
        b = decode_line(ctrl1[index * stride1:(index + 1) * stride1], size1, depth)
        if orient == "horizontal":
            if a != b:
                fail(f"Controller mismatch on horizontal pattern {index}")
            patterns.append(a)
        else:
            patterns.append(a + b)
    return orient, depth, patterns


def build_entry(e, sets):
    set_idx = uint(e.get("set", 0), 8, "set index")
    if set_idx >= len(sets):
        fail(f"Pattern Order references absent Pattern Set {set_idx}")
    start = uint(e.get("start", 0), 8, "PatternEntryIndex")
    total = struct.unpack_from("<H", sets[set_idx])[0]
    count = uint(e.get("count", total - start), 8, "Number of patterns")
    if not count or start + count > total:
        fail(f"Entry set={set_idx}, start={start}, count={count}, total={total}")
    mask = uint(e.get("invert_mask", 0), 64, "invert_mask")
    if mask >= 1 << total:
        fail("invert_mask has bits beyond the number of patterns in the set")
    return {
        "set": set_idx, "start": start, "count": count,
        "led": parse_led(e.get("led", "blue")),
        "invert_mask": mask,
        "exposure_us": uint(e.get("exposure_us", 4000), 32, "exposure_us"),
        "pre_dark_us": uint(e.get("pre_dark_us", 10000), 32, "pre_dark_us"),
        "post_dark_us": uint(e.get("post_dark_us", 40000), 32, "post_dark_us"),
    }


def serialize_patn(sets, order):
    if not sets:
        fail("PATN must include at least one set")
    if len(sets) > 255:
        fail("This writer supports at most 255 Pattern Sets")
    if not (1 <= len(order) <= MAX_ORDER_ENTRIES):
        fail(f"Pattern Order Table must contain 1..{MAX_ORDER_ENTRIES} entries")
    table = bytearray(struct.pack("<I", len(order)))
    for e in order:
        table += struct.pack(
            "<BBBBQIII", e["set"], e["count"], e["led"], e["start"],
            e["invert_mask"], e["exposure_us"], e["pre_dark_us"],
            e["post_dark_us"],
        )
    data_start = 20 + len(table)
    set_start = data_start + 4 + 4 * len(sets)
    set_offsets = []
    cursor = set_start
    for record in sets:
        set_offsets.append(cursor)
        cursor += len(record)
    data_section = (struct.pack("<I", len(sets))
                    + b"".join(struct.pack("<I", x) for x in set_offsets)
                    + b"".join(sets))
    header = b"PATN" + struct.pack("<IIII", data_start, len(data_section),
                                    20, len(table))
    return bytes(header) + bytes(table) + data_section


def parse_patn(data):
    if len(data) > 200_000_000:
        fail("Refusing to parse PATN larger than 200 MB")
    if len(data) < 32 or data[:4] != b"PATN":
        fail("File is not a valid PATN header")
    data_start, data_size, table_start, table_size = struct.unpack_from("<IIII", data, 4)
    if table_start != 20 or data_start != table_start + table_size:
        fail("Unsupported PATN table offsets")
    if data_start + data_size != len(data):
        fail("PATN file length does not match header")
    if table_size < 4 or (table_size - 4) % 24:
        fail("Unexpected 24-byte order entry table size")
    entry_count = struct.unpack_from("<I", data, table_start)[0]
    if entry_count > MAX_ORDER_ENTRIES or table_size != 4 + entry_count * 24:
        fail("Bad order count")
    entries = []
    for i in range(entry_count):
        rec = struct.unpack_from("<BBBBQIII", data, table_start + 4 + i * 24)
        s, count, led, start, invert, exp, pre, post = rec
        entries.append({
            "set": s, "start": start, "count": count, "led": led,
            "invert_mask": invert, "exposure_us": exp,
            "pre_dark_us": pre, "post_dark_us": post,
        })
    if data_size < 8:
        fail("PATN data section is too short")
    set_count = struct.unpack_from("<I", data, data_start)[0]
    if not 1 <= set_count <= 255:
        fail("Unsupported Pattern Set count")
    table_end = data_start + 4 + set_count * 4
    if table_end > len(data):
        fail("Truncated Pattern Set index table")
    offsets = [struct.unpack_from("<I", data, data_start + 4 + i * 4)[0]
               for i in range(set_count)]
    if offsets[0] != table_end or offsets != sorted(offsets):
        fail("Unexpected Pattern Set offsets")
    ends = offsets[1:] + [len(data)]
    sets = []
    for s, e in zip(offsets, ends):
        if s + 8 > e:
            fail("Truncated Pattern Set record")
        declared = struct.unpack_from("<I", data, s + 4)[0]
        if e - s != declared + 8:
            fail("Incorrect Pattern Set payload length")
        sets.append(data[s:e])
    for e in entries:
        if e["set"] >= len(sets):
            fail("Pattern Order references missing Pattern Set")
    return sets, entries


def compile_project(config, base_dir, allow_unverified):
    sets, auto_entries, info = [], [], []
    defaults = dict(DEFAULT_TIMING)
    defaults.update(config.get("defaults", {}))
    auto_split = bool(config.get("auto_split", True))
    groups = config.get("groups", [])
    if not groups:
        fail("Configuration must include 'groups'")
    for group_idx, group in enumerate(groups):
        orientation = group.get("orientation", "horizontal").lower()
        bit_depth = int(group.get("bit_depth", 8))
        if (orientation, bit_depth) not in MAX_SET_COUNTS:
            fail(f"Unsupported group configuration: {orientation}, {bit_depth}")
        if (orientation, bit_depth) != ("horizontal", 8) and not allow_unverified:
            fail(f"Unverified layout: {orientation}/{bit_depth}-bit. "
                 "Use --allow-unverified-layouts only for controlled hardware tests.")
        max_patterns = MAX_SET_COUNTS[orientation, bit_depth]
        names, patterns = [], []
        for spec in group.get("patterns", []):
            for name, pixels in pattern_generator(spec, orientation, base_dir):
                names.append(name)
                patterns.append(pixels)
        if not patterns:
            fail(f"Group {group_idx} has no patterns")
        if not auto_split and len(patterns) > max_patterns:
            fail(f"Group {group_idx} has {len(patterns)} patterns; max is "
                 f"{max_patterns}. Enable auto_split.")
        timing = dict(defaults)
        timing.update(group.get("timing", {}))
        chunks = [
            (patterns[i:i + max_patterns], names[i:i + max_patterns])
            for i in range(0, len(patterns), max_patterns)
        ] if auto_split else [(patterns, names)]
        for chunk, subnames in chunks:
            set_index = len(sets)
            sets.append(encode_set(chunk, orientation, bit_depth))
            info.append({
                "set": set_index, "orientation": orientation, "bit_depth": bit_depth,
                "count": len(chunk), "pattern_names": subnames,
            })
            auto_entries.append(build_entry(
                {**timing, "set": set_index, "start": 0, "count": len(chunk)}, sets
            ))
    if "order" in config:
        order = []
        for item in config["order"]:
            d = dict(defaults)
            d.update(item)
            order.append(build_entry(d, sets))
    else:
        order = auto_entries
    return serialize_patn(sets, order), info


def inspect_patn(data):
    sets, entries = parse_patn(data)
    set_info = []
    for i, rec in enumerate(sets):
        count, depth, size = struct.unpack_from("<HHI", rec)
        orient = infer_orientation(count, depth, size)
        set_info.append({
            "index": i, "orientation": orient, "bit_depth": depth,
            "count": count, "payload_bytes": size,
            "encoding_validated_against_sample": (orient, depth) == ("horizontal", 8),
        })
    return {
        "format": "PATN", "file_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "set_count": len(sets), "pattern_sets": set_info,
        "order_count": len(entries), "order": entries,
    }


def write_pgm(path, values, orient):
    width, height = (1, DMD_HEIGHT) if orient == "horizontal" else (DMD_WIDTH, 1)
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + bytes(values))


def cmd_build(args):
    file_path = Path(args.config).resolve()
    config = json.loads(file_path.read_text(encoding="utf-8"))
    data, description = compile_project(
        config, file_path.parent, args.allow_unverified_layouts
    )
    destination = Path(args.output or config.get("output", "patterns.bin"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    # Parse the written binary again to verify table/header consistency.
    summary = inspect_patn(data)
    print(f"Saved: {destination} ({len(data)} bytes)")
    for item in description:
        print(f"  Set {item['set']}: {item['count']} patterns | "
              f"{item['orientation']} | {item['bit_depth']}-bit")
    print(f"Pattern Order entries: {summary['order_count']}")
    print("NOTE: hardware timing limits are not automatically checked.")
    if args.allow_unverified_layouts:
        print("WARNING: unverified direction/bit-depth layouts may be incompatible.")


def cmd_inspect(args):
    summary = inspect_patn(Path(args.file).read_bytes())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def validate_patn(data_or_path):
    """
    Validates a PATN binary file or bytes.
    Returns: (is_valid: bool, issues: List[str], report: dict)
    """
    issues = []
    if isinstance(data_or_path, (str, Path)):
        p = Path(data_or_path)
        if not p.is_file():
            return False, [f"文件不存在: {data_or_path}"], {}
        try:
            data = p.read_bytes()
        except Exception as exc:
            return False, [f"读取文件失败: {exc}"], {}
    elif isinstance(data_or_path, (bytes, bytearray)):
        data = bytes(data_or_path)
    else:
        return False, ["无效的输入类型，期望 bytes 或文件路径"], {}

    try:
        sets, order = parse_patn(data)
    except Exception as exc:
        return False, [f"PATN文件解析异常: {exc}"], {}

    try:
        recreated = serialize_patn(sets, order)
        if recreated != data:
            issues.append("PATN头部、Pattern Order或Pattern Sets序列化与原数据不一致")
    except Exception as exc:
        issues.append(f"序列化回环失败: {exc}")

    total_patterns = 0
    set_details = []
    for i, record in enumerate(sets):
        try:
            orientation, depth, patterns = decode_set(record)
            regenerated = encode_set(patterns, orientation, depth)
            if regenerated != record:
                issues.append(f"Pattern Set {i} 位平面重编码回环校验失败")
            total_patterns += len(patterns)
            set_details.append({
                "set_index": i,
                "orientation": orientation,
                "bit_depth": depth,
                "pattern_count": len(patterns),
                "payload_bytes": len(record),
            })
        except Exception as exc:
            issues.append(f"Pattern Set {i} 解码异常: {exc}")

    inspection = inspect_patn(data)
    report = {
        "is_valid": len(issues) == 0,
        "set_count": len(sets),
        "order_count": len(order),
        "total_patterns": total_patterns,
        "file_bytes": len(data),
        "sha256": inspection.get("sha256", ""),
        "pattern_sets": set_details,
        "order": order,
        "issues": issues,
    }
    return (len(issues) == 0), issues, report


def build_standard_phase_shift_bin(
    frequencies=(70, 64, 59),
    steps=4,
    exposure_us=20000,
    dark_us=5000,
    led="blue",
    orientation="horizontal",
    bit_depth=8,
    allow_unverified=False,
):
    """
    Builds a standard multi-frequency phase-shifting PATN binary.
    Default: 3 frequencies (70, 64, 59), 4 steps per freq (12 patterns), 8-bit horizontal.
    Returns: (bin_bytes: bytes, info: list)
    """
    cfg = {
        "defaults": {
            "exposure_us": int(exposure_us),
            "dark_time_us": int(dark_us),
            "led": str(led),
        },
        "auto_split": True,
        "groups": [
            {
                "orientation": orientation,
                "bit_depth": bit_depth,
                "patterns": [
                    {
                        "kind": "phase_shift",
                        "frequencies": list(frequencies),
                        "steps": int(steps),
                        "mean": 128,
                        "amplitude": 127,
                    }
                ],
            }
        ],
    }
    allow = allow_unverified or (orientation == "horizontal" and bit_depth == 8)
    return compile_project(cfg, base_dir=Path("."), allow_unverified=allow)


def build_gray_and_phase_shift_bin(
    gray_bits=4,
    frequencies=(64,),
    steps=4,
    exposure_us=20000,
    dark_us=5000,
    led="blue",
    orientation="horizontal",
    bit_depth=8,
    allow_unverified=False,
):
    """
    Builds a Gray code + Phase shift hybrid PATN binary.
    Returns: (bin_bytes: bytes, info: list)
    """
    cfg = {
        "defaults": {
            "exposure_us": int(exposure_us),
            "dark_time_us": int(dark_us),
            "led": str(led),
        },
        "auto_split": True,
        "groups": [
            {
                "orientation": orientation,
                "bit_depth": bit_depth,
                "patterns": [
                    {"kind": "gray_code", "bits": int(gray_bits), "inverse": True},
                    {
                        "kind": "phase_shift",
                        "frequencies": list(frequencies),
                        "steps": int(steps),
                        "mean": 128,
                        "amplitude": 127,
                    },
                ],
            }
        ],
    }
    allow = allow_unverified or (orientation == "horizontal" and bit_depth == 8)
    return compile_project(cfg, base_dir=Path("."), allow_unverified=allow)


def cmd_validate(args):
    data = Path(args.file).read_bytes()
    sets, order = parse_patn(data)
    recreated = serialize_patn(sets, order)
    if recreated != data:
        fail("PATN header/order/set re-serialization differs from source")
    count = 0
    for i, record in enumerate(sets):
        orientation, depth, patterns = decode_set(record)
        regenerated = encode_set(patterns, orientation, depth)
        if regenerated != record:
            fail(f"Set {i} bit-plane roundtrip FAILED")
        count += len(patterns)
    print(f"PASS: byte-for-byte reserialization, pattern bit-plane roundtrip; "
          f"{len(sets)} Pattern Sets, {count} stored patterns, "
          f"{len(order)} order entries.")
    print("NOTE: this is a file-format check, not a projector acceptance test.")


def cmd_extract(args):
    data = Path(args.file).read_bytes()
    sets, order = parse_patn(data)
    dest = Path(args.dir)
    dest.mkdir(parents=True, exist_ok=True)
    for index, rec in enumerate(sets):
        orient, depth, patterns = decode_set(rec)
        for num, values in enumerate(patterns):
            write_pgm(dest / f"set{index:03d}_pattern{num:03d}_{orient}.pgm",
                      values, orient)
    (dest / "pattern_order.json").write_text(json.dumps(order, indent=2), encoding="utf-8")
    print(f"Extracted {sum(struct.unpack_from('<H', s)[0] for s in sets)} "
          f"1D patterns (.pgm) and pattern_order.json to {dest}")


def cmd_example(args):
    destination = Path(args.output)
    destination.write_text(json.dumps(EXAMPLE, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Wrote sample configuration: {destination}")


def main():
    parser = argparse.ArgumentParser(
        description="Offline DLP4710LC/DLPC3479 dual-controller PATN generator"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Build PATN BIN from JSON config")
    build.add_argument("config", help="Configuration JSON")
    build.add_argument("-o", "--output", help="Output BIN path")
    build.add_argument("--allow-unverified-layouts", action="store_true",
                       help="Enable experimental 1-bit / vertical layouts")
    build.set_defaults(func=cmd_build)
    inspect = sub.add_parser("inspect", help="Show sets, order, timing in BIN")
    inspect.add_argument("file")
    inspect.set_defaults(func=cmd_inspect)
    valid = sub.add_parser("validate", help="Validate and roundtrip a BIN file")
    valid.add_argument("file")
    valid.set_defaults(func=cmd_validate)
    extract = sub.add_parser("extract", help="Decode patterns from BIN as PGM")
    extract.add_argument("file")
    extract.add_argument("-d", "--dir", default="extracted_patterns")
    extract.set_defaults(func=cmd_extract)
    example = sub.add_parser("example", help="Write example JSON configuration")
    example.add_argument("-o", "--output", default="pattern_config_example.json")
    example.set_defaults(func=cmd_example)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, KeyError, FileNotFoundError, OSError, json.JSONDecodeError) as err:
        parser.exit(2, f"ERROR: {err}\n")


if __name__ == "__main__":
    main()
