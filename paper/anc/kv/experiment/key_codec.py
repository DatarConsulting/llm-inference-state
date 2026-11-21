"""Each 256-word record has a 4-byte index and 384 or 512 bytes of payload."""
import numpy as np

D = 256


def pack(words):
    words = np.asarray(words)
    if words.dtype != np.uint16 or words.shape[-1] != D or not words.size:
        raise ValueError("Expected nonempty uint16 BF16 bit patterns with head dimension 256")
    w = np.ascontiguousarray(words).reshape(-1, D)
    exponent = ((w >> 7) & 255).astype(np.uint8)
    top = exponent.max(1).astype(np.int32)
    bottom = np.where(exponent == 0, 255, exponent).min(1).astype(np.int32)
    raw = top - bottom > 14
    widths = np.where(raw, 4, 3)
    offsets = np.cumsum(widths, dtype=np.int64) - widths
    if offsets[-1] >= 2**23:
        raise ValueError("Start a new packed arena before its record offset reaches 1 GiB")
    header = ((offsets << 9) | (top << 1) | raw).astype(np.uint32)
    data = np.empty(int(widths.sum()) * 128, dtype=np.uint8)
    base = 128 * offsets
    loc = base[:, None] + np.arange(D)
    data[loc] = ((w & 127) | ((w >> 8) & 128)).astype(np.uint8)
    data[loc[raw] + D] = exponent[raw]
    e = exponent[~raw]
    codes = np.where(e == 0, 0, top[~raw,None] + 1 - e).astype(np.uint8)
    data[base[~raw,None] + D + np.arange(D//2)] = codes[:,::2] | (codes[:,1::2] << 4)
    return header.reshape(words.shape[:-1]), data


def unpack(header, data):
    h = np.asarray(header).reshape(-1)
    base = (h >> 9).astype(np.int64) * 128
    top = ((h >> 1) & 255).astype(np.int32)
    raw = (h & 1).astype(bool)
    d = np.arange(D)
    sm = data[base[:, None] + d].astype(np.uint16)
    x = data[base[:, None] + D + np.where(raw[:,None], d, d//2)]
    code = (x >> ((d % 2)*4)) & 15
    e = np.where(raw[:,None], x, np.where(code == 0, 0, top[:,None]+1-code)).astype(np.uint16)
    return ((sm & 127) | ((sm & 128)<<8) | (e<<7)).reshape(*header.shape,D)
