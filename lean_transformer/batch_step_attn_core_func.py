
import torch
import torch.nn.functional as F


class BatchStepAttnCoreFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        heads,
        scale,
        loop_batch_step,
        queries,
        keys,
        values
    ):
        num_seqs = keys.size(0)
        seq_len = keys.size(1)
        hidden_dim = keys.size(2)
        head_dim = hidden_dim // heads

        heads_t = torch.tensor([heads])
        scale_t = torch.tensor([scale])
        loop_batch_step_t = torch.tensor([loop_batch_step])
        num_seqs_t = torch.tensor([num_seqs])
        seq_len_t = torch.tensor([seq_len])
        hidden_dim_t = torch.tensor([hidden_dim])

        queries = queries.view(num_seqs, seq_len, heads, head_dim).transpose(1, 2).contiguous().view(num_seqs * heads, seq_len, head_dim)
        keys = keys.view(num_seqs, seq_len, heads, head_dim).transpose(1, 2).contiguous().view(num_seqs * heads, seq_len, head_dim)
        values = values.view(num_seqs, seq_len, heads, head_dim).transpose(1, 2).contiguous().view(num_seqs * heads, seq_len, head_dim)

        matmul2_results = torch.empty(
            (num_seqs * heads, seq_len, head_dim), dtype=keys.dtype, device=keys.device
        )

        iter_step = int(loop_batch_step_t.item())
        iter_count = num_seqs * heads
        for iter_idx in range(0, iter_count, iter_step):
            ibatch_range = [iter_idx, min(iter_idx + iter_step, iter_count)]

            # output:           [batch, seql_q, seql_k]
            matmul1_results = torch.bmm(
                queries[ibatch_range[0]:ibatch_range[1], :, :],
                keys[ibatch_range[0]:ibatch_range[1], :, :].transpose(1, 2)
            ) * scale_t

            # output:           [batch, seql_q, seql_k]
            softmax_results = F.softmax(matmul1_results, dim=-1)

            matmul2_results[ibatch_range[0]:ibatch_range[1], :, :] = torch.bmm(
                    softmax_results,
                    values[ibatch_range[0]:ibatch_range[1], :, :])

        outputs = matmul2_results.reshape(num_seqs, heads, seq_len, head_dim).transpose(1, 2).reshape(num_seqs, seq_len, hidden_dim)

        ctx.save_for_backward(
            heads_t,
            scale_t,
            loop_batch_step_t,
            num_seqs_t,
            seq_len_t,
            hidden_dim_t,
            queries,
            keys,
            values
        )

        return outputs.detach()

    @staticmethod
    def backward(ctx, output_grads):
        (
            heads_t,
            scale_t,
            loop_batch_step_t,
            num_seqs_t,
            seq_len_t,
            hidden_dim_t,
            queries,
            keys,
            values
        ) = ctx.saved_tensors

        heads = heads_t[0].item()
        num_seqs = int(num_seqs_t.item())
        seq_len = int(seq_len_t.item())
        hidden_dim = int(hidden_dim_t.item())
        head_dim = hidden_dim // heads

        # [seqs * heads, seql, emb_dim]
        queries_grads = torch.empty((num_seqs * heads, seq_len, head_dim), dtype=queries.dtype, device=queries.device)
        keys_grads = torch.empty((num_seqs * heads, seq_len, head_dim), dtype=keys.dtype, device=keys.device)
        values_grads = torch.empty((num_seqs * heads, seq_len, head_dim), dtype=values.dtype, device=values.device)

        output_grads = output_grads.view(num_seqs, seq_len, heads, head_dim).transpose(1, 2).contiguous().view(num_seqs * heads, seq_len, head_dim)

        # output_grads [seqs, seql, emb_dim]
        iter_step = int(loop_batch_step_t.item())
        iter_count = num_seqs * heads
        for iter_idx in range(0, iter_count, iter_step):
            ibatch_range = [iter_idx, min(iter_idx + iter_step, iter_count)]
            ibatch_sz = ibatch_range[1] - ibatch_range[0]

            # reconstruct softmax_results
            # output:           [seqs*heads, seql_q, seql_k]
            matmul1_results = torch.bmm(
                queries[ibatch_range[0]:ibatch_range[1], :, :],
                keys[ibatch_range[0]:ibatch_range[1], :, :].transpose(1, 2)
            ) * scale_t

            # output:           [seqs*heads, seql_q, seql_k]
            softmax_results = F.softmax(matmul1_results, dim=-1)

            # output_grads  [ seqs * heads, seql, head_dim ]
            # values [ seqs * heads, seql, head_dim ]
            # output: [ seqs * heads, seql, seql ]
            matmul2_dgrad1 = torch.bmm(output_grads[ibatch_range[0]:ibatch_range[1], :, :],
                                       values[ibatch_range[0]:ibatch_range[1], :, :].transpose(1, 2))

            # softmax_results [ seqs * heads, seql, seql ]
            # output_grads  [ seqs * heads, seql, head_dim ]
            # output: [ seqs * heads, seql, head_dim ]
            values_grads[ibatch_range[0]:ibatch_range[1], :, :] = torch.bmm(
                    softmax_results.transpose(1, 2),
                    output_grads[ibatch_range[0]:ibatch_range[1], :, :])
            # output: [ seqs * heads, seql, seql ]
            softmax_grads = torch._softmax_backward_data(matmul2_dgrad1, softmax_results, -1, softmax_results.dtype)

            softmax_grads = softmax_grads.view(ibatch_sz, seq_len, seq_len)

            queries_grads[ibatch_range[0]:ibatch_range[1], :, :] = torch.baddbmm(
                queries_grads[ibatch_range[0]:ibatch_range[1], :, :],
                softmax_grads,
                keys[ibatch_range[0]:ibatch_range[1], :, :],
                beta=0.0,
                alpha=scale_t[0],
            )

            keys_grads[ibatch_range[0]:ibatch_range[1], :, :] = torch.baddbmm(
                keys_grads[ibatch_range[0]:ibatch_range[1], :, :],
                softmax_grads.transpose(1, 2),
                queries[ibatch_range[0]:ibatch_range[1], :, :],
                beta=0.0,
                alpha=scale_t[0],
            )

        queries_grads = queries_grads.reshape(num_seqs, heads, seq_len, head_dim).transpose(1, 2).reshape(num_seqs, seq_len, hidden_dim)
        keys_grads = keys_grads.reshape(num_seqs, heads, seq_len, head_dim).transpose(1, 2).reshape(num_seqs, seq_len, hidden_dim)
        values_grads = values_grads.reshape(num_seqs, heads, seq_len, head_dim).transpose(1, 2).reshape(num_seqs, seq_len, hidden_dim)

        # [ seqs * heads, seql, head_dim ]
        return (
            None,  # heads
            None,  # scale
            None,  # loop_batch_step
            queries_grads,  # queries
            keys_grads,     # keys
            values_grads,   # values
        )


batch_step_attn_core_func = BatchStepAttnCoreFunc.apply
