import time
from dataclasses import dataclass

@dataclass
class TrainState:
    step: int = 0
    accum_step: int = 0
    samples: int = 0
    tokens: int = 0

def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState()
):
    start_time = time.time()
    total_loss = 0
    total_tokens = 0
    log_interval = 10

    for i, batch in enumerate(data_iter):
        out = model(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode.startswith("train"):
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens

            if (i + 1) % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                train_state.accum_step += 1

        total_loss += loss
        total_tokens += batch.ntokens

        if (i + 1) % log_interval == 0 and mode.startswith("train"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start_time

            print(f"Epoch Step: {i+1:6d} | Accumulation Step: {train_state.accum_step:3d} | "
                  f"Loss: {loss / batch.ntokens:.4f} | Tokens / Sec: {train_state.tokens / elapsed:.2f} | "
                  f"Learning Rate: {lr:.4f}")

            start_time = time.time()
            train_state.tokens = 0

    return total_loss / total_tokens, train_state
