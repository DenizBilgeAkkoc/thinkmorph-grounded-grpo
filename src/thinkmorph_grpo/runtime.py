from pathlib import Path
import sys


def load_native(source, checkpoint):
    """Released notebook loading, with the full policy on one visible RTX GPU."""
    source, checkpoint = Path(source).resolve(), Path(checkpoint).resolve()
    sys.path.insert(0, str(source))
    import torch
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from data.transforms import ImageTransform
    from data.data_utils import add_special_tokens
    from modeling.bagel import (BagelConfig, Bagel, Qwen2Config,
        Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel)
    from modeling.autoencoder import load_ae
    from modeling.qwen2 import Qwen2Tokenizer
    from inferencer import InterleaveInferencer
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Give each worker exactly one visible GPU")
    llm = Qwen2Config.from_json_file(str(checkpoint / "llm_config.json"))
    llm.qk_norm, llm.tie_word_embeddings = True, False
    llm.layer_module = "Qwen2MoTDecoderLayer"
    vit = SiglipVisionConfig.from_json_file(str(checkpoint / "vit_config.json"))
    vit.rope = False
    vit.num_hidden_layers -= 1
    vae, vae_config = load_ae(local_path=str(checkpoint / "ae.safetensors"))
    vae.eval().requires_grad_(False)  # Native notebook: FP32 VAE on CPU.
    cfg = BagelConfig(visual_gen=True, visual_und=True, llm_config=llm,
        vit_config=vit, vae_config=vae_config, vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh", latent_patch_size=2, max_latent_size=64)
    with init_empty_weights():
        model = Bagel(Qwen2ForCausalLM(llm), SiglipVisionModel(vit), cfg)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit, meta=True)
    tokenizer = Qwen2Tokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    tokenizer, token_ids, _ = add_special_tokens(tokenizer)
    device_map = {"": 0}
    for name in ("language_model.model.embed_tokens", "time_embedder", "latent_pos_embed",
                 "vae2llm", "llm2vae", "connector", "vit_pos_embed"):
        device_map[name] = 0
    model = load_checkpoint_and_dispatch(model, checkpoint=str(checkpoint / "model.safetensors"),
        device_map=device_map, dtype=torch.bfloat16, force_hooks=True, offload_buffers=True)
    model.eval().requires_grad_(False)
    from .execution_tuning import configure_vae
    configure_vae(vae)
    return model, vae, tokenizer, ImageTransform(1024, 512, 16), ImageTransform(980, 224, 14), token_ids, InterleaveInferencer
