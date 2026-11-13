"""The selected custom RL preset uses the conditional model without CFG."""


def require_no_cfg(settings, source="RL settings"):
    for key in ("cfg_text_scale", "cfg_img_scale"):
        value = settings.get(key)
        if type(value) not in (int, float) or value != 1.0:
            raise ValueError(f"{source}: {key} must explicitly be 1.0 (CFG disabled); got {value!r}")
