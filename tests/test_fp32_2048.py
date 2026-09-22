"""Focused correctness checks for the FP32 K=1025..2048 dispatch tier."""

import unittest

import torch

import deep_select


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class FP32TopK2048Test(unittest.TestCase):
    def test_modes_with_short_and_varlen_rows(self):
        torch.manual_seed(19)
        length = 65537
        # The visible width is deliberately unaligned; the row stride meets
        # the input alignment contract.
        x = torch.randn((3, 65792), device="cuda", dtype=torch.float32)[:, :length]
        x[1] = torch.arange(length, device="cuda", dtype=torch.float32)

        modes = [
            (False, False, False),
            (False, False, True),
            (False, True, False),
            (False, True, True),
            (True, False, True),
        ]
        for topk in (1025, 2048):
            ends = torch.tensor([topk - 1, 16384, length], device="cuda", dtype=torch.int32)
            offsets = torch.tensor([7, -5, 11], device="cuda", dtype=torch.int32)
            for indices_type in (torch.int32, torch.int64):
                for sorted_value, sorted_index, return_value in modes:
                    with self.subTest(topk=topk, indices_type=indices_type,
                                      sorted_value=sorted_value,
                                      sorted_index=sorted_index,
                                      return_value=return_value):
                        values, indices = deep_select.topk(
                            x, topk, sorted=sorted_value, end=ends,
                            indices_type=indices_type, sorted_index=sorted_index,
                            output_idx_offset=offsets, idx_oob_fill_value=-123456,
                            return_value=return_value,
                        )
                        torch.cuda.synchronize()
                        self.assertEqual(indices.shape, (3, topk))
                        self.assertEqual(values is not None, return_value)
                        for row, end in enumerate(ends.tolist()):
                            count = min(end, topk)
                            selected = indices[row, :count].to(torch.int64) - offsets[row]
                            expected = torch.topk(x[row, :end], count)
                            self.assertTrue(torch.equal(
                                selected.sort().values, expected.indices.sort().values
                            ))
                            self.assertTrue(torch.all(indices[row, count:] == -123456).item())
                            if sorted_index:
                                self.assertTrue(torch.all(selected[1:] >= selected[:-1]).item())
                            if values is not None:
                                self.assertTrue(torch.equal(values[row, :count], x[row, selected]))
                                if sorted_value:
                                    self.assertTrue(torch.all(
                                        values[row, 1:count] <= values[row, :count - 1]
                                    ).item())

    def test_nan_guard(self):
        x = torch.randn((1, 32768), device="cuda", dtype=torch.float32)
        x[0, 20000] = float("nan")
        _, indices = deep_select.topk(
            x, 2048, indices_type=torch.int32, return_value=False,
            abort_when_nan_found=False,
        )
        self.assertEqual(indices[0, 0].item(), 0x3F3F3F3F)

    def test_tie_heavy_rows(self):
        length = 32768
        x = torch.zeros((2, length), device="cuda", dtype=torch.float32)
        x[1, ::7] = 1.0
        _, indices = deep_select.topk(
            x, 2048, indices_type=torch.int32, return_value=False,
        )
        for row in range(2):
            selected = indices[row].to(torch.int64)
            self.assertTrue(torch.all((selected >= 0) & (selected < length)).item())
            self.assertEqual(torch.unique(selected).numel(), 2048)
            threshold = torch.topk(x[row], 2048).values[-1]
            self.assertTrue(torch.all(x[row, selected] >= threshold).item())


if __name__ == "__main__":
    unittest.main()
