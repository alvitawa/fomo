# Discrete Prompt Tuning

Official implementation of our FOMO project paper:

**> Discrete Prompt Tuning for Vision-Language Models**
> Alfonso Taboada Warmerdam, Lucas Meijer, Thomas Komen, Max de Redelijkheid, Yingjun Du
> 
![image](https://github.com/alvitawa/fomo/assets/10909323/8225533f-32ae-49ea-a139-82d4752bffbf)

# Reproduce

## Installation
1. `python3 -m venv venv; source /venv/bin/activate; pip install -r requirements.txt` (Check out CoPrompt instructions (README_CoPrompt.md) if issues arise)
2. Do a search and replace (Ctrl+Shift+R in IntelliJ) for '/home/ataboadawarmer/data' and replace with the output path you want
3. Same with '//scratch-shared/promptlearning/coop_datasets/' and replace with the path to the datasets (as specified in CoPrompt README)

## Training
1. Grid search `bash grid.sh` and then `grid.ipynb` to analyze the results
2. Component ablations `bash ablate_{$dataset}.sh` and then `abla.ipynb` to analyze the results
3. ImageNet train with `bash trainimnet.sh` and then `grid.ipynb` to analyze the results
4. For dataset transferability do `bash fullimnet.sh` first and then `bash crossds.sh` and/or `bash crossds_projfree.sh` and then `crossds.ipynb` to analyze the results

## Discrete Prompt Tuning

If you want to use DPT to learn discrete prompts for the text encoder in your own framework you can just include this code snippet:

```python

import torch
import torch.nn as nn
from torch import autograd
from clip import clip

class GetToggle(autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor):
        out = torch.zeros_like(scores)

        order = torch.argsort(scores, dim=-1, descending=True)

        bos, eos = [clip._tokenizer.encoder['<|startoftext|>'], clip._tokenizer.encoder['<|endoftext|>']]

        # This is algorithm 1 from the paper
        ranks = torch.zeros_like(order[:, 0])
        while True:
            best = order[torch.arange(order.shape[0]), ranks]
            # no eos or bos
            for i, t in enumerate(best):
                if t.item() in [bos, eos]:
                    ranks[i] += 1
                    break
            else:
                # break
                decoded = clip._tokenizer.decode(best.cpu().numpy())
                re_encoded = clip._tokenizer.encode(decoded)
                if not (re_encoded == best.cpu().numpy().tolist()):
                    re_decoded = clip._tokenizer.decode(re_encoded)
                    print('!= {} vs {}'.format(decoded, re_decoded))
                    scores_now = scores.gather(1, ranks.unsqueeze(1)).squeeze(1)
                    next_ranks = ranks + 1
                    next_scores = scores.gather(1, next_ranks.unsqueeze(1)).squeeze(1)
                    # Find the smallest difference in scores to update
                    smallest = torch.argsort(scores_now - next_scores, descending=True)
                    for i in smallest:
                        ranks[i] += 1
                        best = order[torch.arange(order.shape[0]), ranks]
                        decoded = clip._tokenizer.decode(best.cpu().numpy())
                        re_encoded = clip._tokenizer.encode(decoded)
                        if re_encoded == best.cpu().numpy().tolist():
                            break
                        ranks[i] -= 1
                    else:
                        ranks[smallest[0]] += 1
                else:
                    break

        out[torch.arange(scores.shape[0]), best] = 1
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        # (passthrough trick)
        return g, None




class EmbeddingsToggler(nn.Module):
    def __init__(self, embeddings: nn.Embedding, n: int, init_indices=None, init_value=0.001, dtype=torch.float32):
        super().__init__()
        self.embeddings = embeddings
        self.n = n
        self.scores = nn.Parameter(torch.zeros(n, embeddings.num_embeddings, dtype=dtype), requires_grad=True)
        if init_indices is not None:
            self.scores.data[torch.arange(n), init_indices] = torch.tensor(init_value, dtype=dtype)
        self.last_best = None


    def forward(self):
        toggle = GetToggle.apply(self.scores)  # (n, num_embeddings)
        best = torch.argmax(toggle, dim=-1).detach()

        return toggle @ self.embeddings.weight.type(self.scores.dtype), best
```

And then use it like:

```python
# In model init
self.emb_toggler = EmbeddingsToggler(clip_model.token_embedding, n_ctx, dtype=dtype)
mean = 0
std = 0.004
self.emb_toggler.scores.data.normal_(mean, std)

# In forward
# embs has the sequence of embeddings
embs, best = self.emb_toggler()

# Optional to print progress
decoded = clip._tokenizer.decode(best.cpu().numpy())
print(decoded, self.emb_toggler.scores[torch.arange(self.n_ctx), best])
```