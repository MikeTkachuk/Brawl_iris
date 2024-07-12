### Overview

1. Last commit is highly experimental, i did not make it work. It's about making multi-head generation sequential instead of fully parallel.
   The motivation was that with 8 gen heads and 16x16 token img size the image rows were kinda smoothed out.
   That was fixed with reducing the gen heads to 2, so I suspected lack of inter-head communication was the reason.
   Without multiple heads the actor training is really slow. I feel like adding sequential generation slowed it a lot.
2. Current tokenizer is using unit vector embeddings. 16x16 tokenization is fine with voc size of 64 and looks it can be reduced even further.
3. While I fixed noise in movement controls, shot control noise seem to be still present, however for Jacky its visually similar to her passive.
4. My experiment with 2 gen heads was quite good at generating ends, still had lots of noise.
5. Last commit also removed first frame from loss and reduced context to 4 (3 + first). I also increased BS and LR x10. That run + sequential heads produced weird 
   behaviour where the central player token would not always be there (happened 1st tme), its weird as the loss was the lowest of all my runs. overfit? wrong extra seq input aka data leak?
   Movements seem fine otherwise. All players just disappear though. 
6. Stable WM has value prediction 


### Todos

1. Try simple large batch converge from scratch on 1 gen head first. Vary context size 4-8. If it doesn't work add eval viz (4.) for intuition.
2. Add evaluation with generation and pre-defined actions.
3. Try reducing vocab size to like 16-32.
4. Last commit is likely to have a bug with sequence alignment. either in labels, logits or in heads.
5. Rolling back experimental is optional. If I make it work it will be nice - 8 heads reduce quality too much.
6. If does not converge (apparently it does so this is optional) add pixel values into wm input to estimate where tokenizer fails to capture useful info.