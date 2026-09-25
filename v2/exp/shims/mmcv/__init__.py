"""Minimal stand-in for the three mmcv symbols StegTransX-V1.py imports.

Only ConvModule, build_activation_layer and build_norm_layer are provided,
with mmcv's defaults for the configurations StegTransX uses (conv -> norm ->
activation, bias='auto', Kaiming fan-out initialisation, norm weight 1 / bias 0).
"""
