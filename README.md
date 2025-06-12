The task was to build a deep learning model that takes as input an image of a mathematical expression and outputs the corresponding LaTeX code using an Encoder-Decoder architecture.

For the competitive part, implemented a model using ResNet18 as the encoder to extract features from expression images, and a single-layer LSTM as the decoder to generate LaTeX sequences autoregressively.

Employed cross-entropy loss, teacher forcing (50%), and special START/END tokens for sequence modeling.

Part-a: Trained only on the synthetic dataset.

Achieved BLEU score of 36.13 and CHRF score of 37.77 on validation of handwritten and both validation/test sets of synthetic data.

Part-b: Fine-tuned the synthetic-trained model on the handwritten dataset.

Achieved BLEU score of 35.20 and CHRF score of 36.65 on the same evaluation sets.
