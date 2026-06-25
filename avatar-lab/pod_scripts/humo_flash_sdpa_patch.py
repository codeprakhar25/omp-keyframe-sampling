p = "/workspace/avatar/HuMo/humo/models/wan_modules/attention.py"
s = open(p).read()
marker = "    assert q.size(-1) <= 256\n"
fallback = marker + (
    "    # --- Blackwell sdpa fallback (no flash-attn wheel for cu128/torch2.7) ---\n"
    "    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):\n"
    "        import torch.nn.functional as _F\n"
    "        _hd = (torch.float16, torch.bfloat16)\n"
    "        qh = q if q.dtype in _hd else q.to(dtype)\n"
    "        kh = k if k.dtype in _hd else k.to(dtype)\n"
    "        vh = v if v.dtype in _hd else v.to(dtype)\n"
    "        qh = qh.to(vh.dtype); kh = kh.to(vh.dtype)\n"
    "        if q_scale is not None: qh = qh * q_scale\n"
    "        qh = qh.transpose(1, 2); kh = kh.transpose(1, 2); vh = vh.transpose(1, 2)\n"
    "        gqa = qh.size(1) != kh.size(1)\n"
    "        x = _F.scaled_dot_product_attention(qh, kh, vh, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale, enable_gqa=gqa)\n"
    "        return x.transpose(1, 2).type(q.dtype)\n"
)
assert marker in s, "marker not found"
if "Blackwell sdpa fallback" in s:
    print("already patched")
else:
    s = s.replace(marker, fallback, 1)
    open(p, "w").write(s)
    print("patched sdpa fallback")
