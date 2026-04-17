import yaml


def test_configs_are_valid_yaml():
    for p in [
        "SwinGAN/config_i2i_baseline.yaml",
        "SwinGAN/config_i2i_attention.yaml",
    ]:
        with open(p, "r") as f:
            cfg = yaml.safe_load(f)
        assert "img_size" in cfg
        assert "patch_size" in cfg
        assert "embed_dim" in cfg

