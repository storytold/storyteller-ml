from hf_utils import load_custom_model_from_hf

print("loading all models")
print(load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None))

print(
    load_custom_model_from_hf(
        "Plachta/Seed-VC",
        "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
        "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
    )
)
print(
    load_custom_model_from_hf(
        "Plachta/Seed-VC",
        "DiT_seed_v2_uvit_facodec_small_wavenet_f0_bigvgan_pruned.pth",
        "config_dit_mel_seed_facodec_small_wavenet_f0.yml",
    )
)
