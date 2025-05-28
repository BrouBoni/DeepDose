import numpy as np
from tensorflow import concat as cat
from tensorflow.keras import layers, Model

from src.blocks import ConvBlock, TransformerEncoder, PosEmbedding

def idota(inshape, steps, enc_feats, num_heads, num_transformers,
          kernel_size, dropout_rate=0.2, causal=True):
    """ Creates the transformer model for dose calculation."""

    # Calculate size of input tokens
    slice_dim = inshape[1:]
    token_dim = (*[int(i/2**steps) for i in slice_dim[:-1]], enc_feats)
    token_size = np.prod(token_dim)
    num_tokens = inshape[0]

    # Input CT and ray tracing values
    ct_vol = layers.Input((num_tokens, *slice_dim))
    ray = layers.Input((num_tokens, *slice_dim))
    x = layers.Concatenate()([ct_vol, ray])
    x_history = [x]

    # Encode inputs
    for _ in range(steps):
        x = ConvBlock(kernel_size=kernel_size, downsample=True)(x)
        x_history.append(x)

    # Tokenize + positional embedding
    tokens = ConvBlock(enc_feats, kernel_size, flatten=True)(x)
    tokens = PosEmbedding(num_tokens, token_size)(tokens)
    
    # Stack transformer encoders
    for i in range(num_transformers):

        # Transformer encoder blocks
        tokens = TransformerEncoder(num_heads, num_tokens, token_size,
            causal=causal, dropout_rate=dropout_rate)(tokens)

    # Reshape to cube
    x = layers.TimeDistributed(layers.Reshape((token_dim)))(tokens)

    # Decode and upsample
    for _ in range(steps):
        x = cat([x, x_history.pop()], axis=-1)
        x = ConvBlock(kernel_size=kernel_size, upsample=True)(x)

    dose = layers.Conv3D(1, kernel_size, padding='same')(x)

    return Model(inputs=[ct_vol, ray], outputs=dose)
