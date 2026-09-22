"""Reproducible FP32 K=2048 benchmark, including optional large query blocks.

Run with CUDA_VISIBLE_DEVICES set to the GPU under test. CUDA Graph capture
amortizes Python launch overhead; warmup and capture are outside timed events.
"""

import argparse
import statistics

import torch

import deep_select


SHAPES = [
    (1, 129280),
    (1, 262144),
    (1, 1048576),
    (8, 1048576),
    (32, 262144),
    (64, 1048576),
    (128, 1048576),
]
LARGE_BATCH_SHAPES = [
    (512, 262144),
    (1024, 262144),
    (1024, 1048576),
]
TOPK = 2048
CAPTURED_CALLS = 32


def benchmark(batch: int, length: int) -> float:
    x = torch.randn((batch, length), device="cuda", dtype=torch.float32)

    def run():
        return deep_select.topk(
            x, TOPK, indices_type=torch.int32, return_value=False
        )[1]

    actual = run()
    expected = torch.topk(x, TOPK, dim=1).indices.to(torch.int32)
    assert torch.equal(actual.sort(dim=1).values, expected.sort(dim=1).values)

    for _ in range(10):
        run()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(CAPTURED_CALLS):
            run()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    samples_us = []
    for _ in range(9):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000 / CAPTURED_CALLS)
    return statistics.median(samples_us)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--large-batch", action="store_true",
        help="also measure query blocks representative of a 1024-row indexer slab",
    )
    args = parser.parse_args()
    torch.manual_seed(7)
    print(f"deep_select: {deep_select.__file__}", flush=True)
    print("batch length median_us", flush=True)
    for b, n in SHAPES + (LARGE_BATCH_SHAPES if args.large_batch else []):
        print(b, n, f"{benchmark(b, n):.2f}", flush=True)
