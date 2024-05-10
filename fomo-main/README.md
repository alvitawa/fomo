# Example

        from pass import *

## In model init
        self.emb_toggler = EmbeddingsToggler(clip_model.token_embedding, n_ctx, k=1, dtype=dtype)
        mean = 0
        std = 0.004
        self.emb_toggler.scores.data.normal_(mean, std)

## In forward
        # embs has the sequence of embeddings
        embs, best = self.emb_toggler()

        # Optional to print progress
        decoded = clip._tokenizer.decode(best.cpu().numpy())
        print(decoded, self.emb_toggler.scores[torch.arange(self.n_ctx), best])
