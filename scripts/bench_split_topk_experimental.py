"""Scratch experiment: split a long FP32 row across several DeepSelect CTAs.

This intentionally covers only contiguous inputs, K=2048, and the index-only
output mode. It is not part of the public DeepSelect API.
"""

import statistics

import torch

import deep_select


K = 2048
SHAPES = [
    (1, 262144),
    (1, 524288),
    (1, 1048576),
    (8, 262144),
    (8, 1048576),
    (32, 262144),
    (32, 1048576),
    (64, 1048576),
    (128, 1048576),
]


def single_cta(x):
    return deep_select.topk(x, K, indices_type=torch.int32, return_value=False)[1]


def split_row(x, parts):
    batch, length = x.shape
    assert x.is_contiguous() and length % parts == 0
    chunk = length // parts
    assert chunk >= K and chunk % 256 == 0

    local_values, local_indices = deep_select.topk(
        x.reshape(batch * parts, chunk), K,
        indices_type=torch.int32, return_value=True,
    )
    _, selected = deep_select.topk(
        local_values.reshape(batch, parts * K), K,
        indices_type=torch.int32, return_value=False,
    )
    return (
        local_indices.reshape(batch, parts * K).gather(1, selected.long())
        + (selected // K) * chunk
    )


def bench(fn):
    for _ in range(8):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(16):
            fn()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    samples = []
    for _ in range(7):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / 16)
    return statistics.median(samples)


if __name__ == "__main__":
    torch.manual_seed(23)
    for batch, length in SHAPES:
        x = torch.randn((batch, length), device="cuda", dtype=torch.float32)
        expected = torch.topk(x, K, dim=1).indices.to(torch.int32).sort(dim=1).values
        assert torch.equal(single_cta(x).sort(dim=1).values, expected)
        print(batch, length, "single", f"{bench(lambda: single_cta(x)):.2f}")
        for parts in (4, 8, 16):
            if length // parts < K:
                continue
            actual = split_row(x, parts)
            assert torch.equal(actual.sort(dim=1).values, expected)
            print(batch, length, "split", parts, f"{bench(lambda: split_row(x, parts)):.2f}")
