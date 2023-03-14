## WikiGPT

143M parameter GPT model trained on [WikiText-103](https://huggingface.co/datasets/wikitext) using `torch.distributed` for DDP and `torch.cuda.amp` for mixed precision. Used [youtokentome](https://github.com/VKCOM/YouTokenToMe) for tokenizing the corpus.

<table>
  <tr>
    <td>d_model</td>
    <td>1024</td>
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
    <td>lr</td>
    <td>0.0001</td>
  </tr>
  <tr>
    <td>train_steps</td>
    <td>15000</td>
  </tr>
  <tr>
    <td>lr_schedule</td>
    <td> - </td>
  </tr>
  <tr>
    <td>batch_size_per_gpu</td>
    <td>16</td>
  </tr>
  <tr>
    <td>gradient_accumulation</td>
    <td> - </td>
  </tr>
  <tr>
    <td>gpu</td>
    <td>RTX 3090</td>
  </tr>
  <tr>
    <td>n_gpu</td>
    <td>14</td>
  </tr>
</table>

Few changes include using [RMS norm](https://arxiv.org/abs/1910.07467) instead of layer norm, [RoPE](https://arxiv.org/abs/2104.09864) instead of learned position encodings, [SwiGLU](https://arxiv.org/abs/2002.05202v1) activation in FFNs instead of GELU, and [Lion](https://arxiv.org/abs/2302.06675) instead of Adam.

![WikiGPT](https://user-images.githubusercontent.com/86470305/224977751-e4aafd76-58ba-4584-a6de-55cfa10830bc.png)
