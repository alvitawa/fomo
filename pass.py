
class GetToggle(autograd.Function):
    @staticmethod
    def forward(ctx, scores: torch.Tensor, k=1):
        out = torch.zeros_like(scores)

        order = torch.argsort(scores, dim=-1, descending=True)

        topk = order[:, :k]
        topk_values = scores.gather(1, topk)
        topk_totals = topk_values.sum(dim=1, keepdim=True)

        out.scatter_(1, topk, topk_values / topk_totals)
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None




class EmbeddingsToggler(nn.Module):
    def __init__(self, embeddings: nn.Embedding, n: int, k=1, init_indices=None, init_value=0.001, dtype=torch.float32):
        super().__init__()
        self.embeddings = embeddings
        self.n = n
        self.k = k
        self.scores = nn.Parameter(torch.zeros(n, embeddings.num_embeddings, dtype=dtype), requires_grad=True)
        if init_indices is not None:
            self.scores.data[torch.arange(n), init_indices] = torch.tensor(init_value, dtype=dtype)

    def forward(self):
        toggle = GetToggle.apply(self.scores, self.k)  # (n, num_embeddings)
        best = torch.argmax(toggle, dim=-1).detach()


        return toggle @ self.embeddings.weight.type(self.scores.dtype), best  # (n, embedding_dim)
