from .unet_baseline import UNetBaseline
from .phase2_unet_freq import Phase2UNetFreq
from .phase3_unet_freq_symscan import Phase3UNetFreqSymScan


def _cfg_get(cfg_node, key, default=None):
    try:
        if cfg_node is None:
            return default
        if isinstance(cfg_node, dict):
            return cfg_node.get(key, default)
        return cfg_node.get(key, default)
    except Exception:
        try:
            return getattr(cfg_node, key)
        except Exception:
            return default


def build_model(cfg):
    name = str(_cfg_get(cfg.model, "name", "unet_base")).lower()

    in_ch = int(_cfg_get(cfg.model, "in_ch", 3))
    out_ch = int(_cfg_get(cfg.model, "out_ch", 3))
    base_ch = int(_cfg_get(cfg.model, "base_ch", 48))
    freq_ch = int(_cfg_get(cfg.model, "freq_ch", 16))
    ssm_mode = str(_cfg_get(cfg.model, "ssm_mode", "convscan"))

    if name == "unet_base":
        return UNetBaseline(
            in_ch=in_ch,
            out_ch=out_ch,
            base_ch=base_ch,
        )

    if name == "unet_freq":
        return Phase2UNetFreq(
            in_ch=in_ch,
            out_ch=out_ch,
            base_ch=base_ch,
            freq_ch=freq_ch,
        )

    if name == "unet_freq_symscan":
        return Phase3UNetFreqSymScan(
            in_ch=in_ch,
            out_ch=out_ch,
            base_ch=base_ch,
            freq_ch=freq_ch,
            ssm_mode=ssm_mode,
        )

    raise ValueError(f"Unknown model.name: {name}")