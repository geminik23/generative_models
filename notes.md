## REALNVP

RealNVP (Real-valued Non-Volume Preserving) is a generative model that utilizes normalizing flows for density estimation and sampling. The model is based on coupling layers and permutation operations, making the transformation invertible and easy to compute Jacobian determinants. This implementation is for a 1D version of the RealNVP model.

**Brief Description (README):**

RealNVP is a generative model that uses invertible coupling layers and permutation operations for efficient density estimation and sampling. This implementation is a 1D version of the model, making it suitable for simple toy projects.

**One or two-line version:**

RealNVP: An invertible 1D generative model for efficient density estimation and sampling using coupling layers and permutation operations.

## GLOW

GLOW is an extension of RealNVP and is another generative model that utilizes normalizing flows. GLOW builds on RealNVP by introducing invertible 1x1 convolutions, which improve the expressiveness of the model while maintaining invertibility. This implementation is a 2D version of the GLOW model.

**Brief Description (README):**

GLOW is a generative model that extends RealNVP by incorporating invertible 1x1 convolutions for improved expressiveness while maintaining invertibility. This implementation is a 2D version of the model, suitable for image generation tasks.

**One or two-line version:**

GLOW: A 2D generative model that extends RealNVP with invertible 1x1 convolutions for improved expressiveness in image generation tasks.