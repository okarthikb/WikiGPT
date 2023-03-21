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

Few changes: [RMS norm](https://arxiv.org/abs/1910.07467) instead of layer norm, [RoPE](https://arxiv.org/abs/2104.09864) instead of learned position encoding, [SwiGLU](https://arxiv.org/abs/2002.05202v1), and [Lion](https://arxiv.org/abs/2302.06675) instead of Adam(W).

<img width="1336" alt="Screenshot 2023-03-21 at 1 16 53 AM" src="https://user-images.githubusercontent.com/86470305/226524451-2930c367-4748-45d6-8f68-56f40f54f51d.png">
