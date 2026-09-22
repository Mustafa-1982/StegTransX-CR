"""StegTransX-CR: multi-format compression-resilient deep image steganography.

Implementation of the technical specification (stegno_specification.pdf) for the
paper "StegTransX-CR: Multi-Format Compression-Resilient Deep Steganography via
Unified Differentiable Codec Simulation and Frequency-Adaptive Embedding".

Protocol deviations from the paper, applied deliberately (see PROTOCOL_NOTES):
  1. The reveal network never receives the codec index k (paper Algorithm 4 /
     real deployment), unlike paper Algorithm 1 line 22.
  2. L_H and L_R include the Charbonnier term named in paper section 3.6.1 but
     missing from paper equations (14)-(15).
  3. Experiments run at most 500 epochs each with early stopping (patience 9)
     instead of the paper's 8,000 epochs, to fit a 24 h Colab Pro+ budget.
  4. Evaluation is reported twice: through the differentiable simulator S (the
     paper's protocol) and through real JPEG/WebP/HEIF/AVIF encoders.
"""

__version__ = "1.0.0"

PROTOCOL_NOTES = [
    "Reveal network R_phi takes only the received image (no codec index k).",
    "L_H and L_R = Laplacian pyramid + Charbonnier + range restriction.",
    "Max 500 epochs per experiment, early stopping on val secret PSNR, patience 9.",
    "Test twice: differentiable simulator S and real codec files.",
    "Learned codec-simulator parameters are frozen during steganography training.",
    "Experiments A/B/C each start from a fresh hiding/reveal network.",
    "Restriction loss stays on during the hiding-loss warmup so the residual cannot explode.",
]

__all__ = ["__version__", "PROTOCOL_NOTES"]
