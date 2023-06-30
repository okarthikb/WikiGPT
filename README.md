## WikiGPT

81M parameter GPT model trained on [WikiText-103](https://huggingface.co/datasets/wikitext) using `torch.distributed` for DDP and `torch.cuda.amp` for mixed precision. Used [youtokentome](https://github.com/VKCOM/YouTokenToMe) for tokenizing the corpus.

<table>
  <tr>
    <td>d_model</td>
    <td>768</td>
  </tr>
  <tr>
    <td>n_head</td>
    <td>16</td>
  </tr>
  <tr>
    <td>n_layer</td>
    <td>8</td>
  </tr>
  <tr>
    <td>ctx_len</td>
    <td>512</td>
  </tr>
  <tr>
    <td>vocab_size</td>
    <td>16384</td>
  </tr>
  <tr>
    <td>max_lr</td>
    <td>3e-4</td>
  </tr>
  <tr>
    <td>min_lr</td>
    <td>5e-5</td>
  <tr>
  <tr>
    <td>weight_decay</td>
    <td>1e-2</td>
  </tr>
  <tr>
    <td>train_steps</td>
    <td>15000</td>
  </tr>
  <tr>
    <td>lr_warmup_steps</td>
    <td>1500</td>
  </tr>
  <tr>
    <td>lr_schedule</td>
    <td>inverse square root</td>
  </tr>
  <tr>
    <td>batch_size</td>
    <td>128</td>
  </tr>
  <tr>
    <td>gpu</td>
    <td>RTX 3090</td>
  </tr>
  <tr>
    <td>n_gpu</td>
    <td>4</td>
  </tr>
</table>

Few changes: [RMS norm](https://arxiv.org/abs/1910.07467) instead of layer norm, [RoPE](https://arxiv.org/abs/2104.09864) instead of learned position encoding, [SwiGLU](https://arxiv.org/abs/2002.05202v1) instead of GELU, and [Lion](https://arxiv.org/abs/2302.06675) optimizer instead of Adam(W).

### Lion

The difference between AdamW and Lion is as follows

<img width="866" alt="Lion" src="https://github.com/okarthikb/WikiGPT/assets/86470305/488c267e-a966-49db-bbe5-77cd669e6349">

Lion saves memory by only keeping track of the first moment (the EMA of gradients) whereas Adam and variants use the second moment (the EMA of gradients squared) as well.

<img width="1336" alt="Screenshot 2023-03-21 at 1 16 53 AM" src="https://user-images.githubusercontent.com/86470305/226524451-2930c367-4748-45d6-8f68-56f40f54f51d.png">

### Scaled RoPE (update!)

A few days ago 4chan anon kaiokendev made a two line edit in his RoPE implementation which appeared to [double the model's context length](https://kaiokendev.github.io/til#extending-context-to-8k) at test time. The equivalent edit in the implementation here would be

<img width="538" alt="RoPE-scale" src="https://github.com/okarthikb/WikiGPT/assets/86470305/d4999d5c-75b9-4b67-baa4-4e923bf7f8fc">

There's no fine-tuning going on, all we have to do is scale `torch.arange(l)` by `scale`. What's happening? Here, `x` will be the query `q` or key `k` batch on which we apply RoPE. `x` will be of shape `(batch_size, seq_len, d_model)` - we only need to care about the last two dimensions (assume we're working with a single batch, which is what we do at test time). We first define a vector `theta` of `d` angle values like so

`[θ_1, θ_1, θ_2, θ_2, ..., θ_{d/2}]`

where `d` is assumed to be even. Then we consider `torch.arange(l)` which is simply

`[1, 2, ..., l]`

where `l` is `seq_len`. `torch.einsum('i, j -> ij', torch.arange(l), theta)` is the outer product between the two arguments, so you get `ltheta` which is a matrix that looks like

```
[[1 * θ_1, 1 * θ_1, ..., 1 * θ_{d/2}, 1 * θ_{d/2}],
 [2 * θ_1, 2 * θ_1, ..., 2 * θ_{d/2}, 2 * θ_{d/2}],
 ...
 [l * θ_1, l * θ_1, ..., l * θ_{d/2}, l * θ_{d/2}]]
```

So `ltheta` is same shape as `x` - `(seq_len, d_model)`. `x_rot` is just `x` with alternating columns and even columns made negative. We apply `sin` and `cos` along each row of `ltheta`, and elementwise multiply the outputs with `x_rot` and `x` respectively. To double the context length, we halve the frequencies

```
[[1/2 * θ_1, 1/2 * θ_1, ..., 1/2 * θ_{d/2}, 1/2 * θ_{d/2}],
 [2/2 * θ_1, 2/2 * θ_1, ..., 2/2 * θ_{d/2}, 2/2 * θ_{d/2}],
 ...
 [2l/2 * θ_1, 2l/2 * θ_1, ..., 2l/2 * θ_{d/2}, 2l/2 * θ_{d/2}]]
```

Intuitively, the model has learned to map sections of a context to specific sections of the `sin` and `cos` curves. Imagine dividing an arbitrary context into uniform chunks. Then no matter the context length, we want each chunk to be mapped to the same section of the trig curves. We do this by scaling down the frequencies. A [paper](https://arxiv.org/pdf/2306.15595.pdf) laying out the exact trick came out _just_ recently and it has this helpful visualization of what's going on

<img width="738" alt="RoPE-explain" src="https://github.com/okarthikb/WikiGPT/assets/86470305/00a433b1-b12d-414d-b5cc-22dd2e085b25">

If we have a trained model with context length `l`, we can make it work with longer context length `k * l` by doing

```python
from gpt import *


k = 2

model = GPT(...)

model.load_state_dict(torch.load('trained_model.pt'))

# update RoPE
for layer in model.layers:
  layer.pe = RotaryEmbedding(k * model.l, model.d, scale=k)

torch.save(model.state_dict(), 'extended_model.pt')
```

### Attention visualization

Attention scores visualized for 64 tokens in the middle of a completion:

![attention](https://user-images.githubusercontent.com/86470305/226539557-17c81ed9-b38b-4af0-aaee-29fcae1817e7.png)
