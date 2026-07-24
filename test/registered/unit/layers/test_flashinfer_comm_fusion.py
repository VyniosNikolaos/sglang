import functools
import types
import unittest
from enum import Enum
from unittest.mock import patch

import torch

from sglang.srt.layers import flashinfer_comm_fusion as fusion
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-c", runner_config="4-gpu-h100")
register_cuda_ci(est_time=30, stage="base-c", runner_config="4-gpu-b200")
register_cuda_ci(est_time=30, stage="base-c", runner_config="4-gpu-gb300")


class _FakeWorkspace:
    def __init__(self, backend, world_size):
        self.backend = backend
        self.world_size = world_size

    def is_buffer_size_sufficient(self, **_kwargs):
        return True


class _FakeMNNVLStrategy(Enum):
    ONESHOT = 0
    TWOSHOT = 1
    AUTO = 99


class _FakeMNNVLWorkspace:
    def __init__(self):
        self.strategy = None

    @functools.cache
    def is_buffer_size_sufficient(
        self,
        tp_size,
        num_tokens,
        hidden_dim,
        dtype,
        strategy=_FakeMNNVLStrategy.AUTO,
    ):
        self.strategy = strategy
        return True


class _FakeTRTLLMWorkspace:
    def __init__(self):
        self.use_oneshot = None

    def is_buffer_size_sufficient(
        self, tp_size, num_tokens, hidden_dim, dtype, use_oneshot=None
    ):
        self.use_oneshot = use_oneshot
        return True


class _FakeFlashInferComm:
    class AllReduceFusionPattern:
        kARResidualRMSNorm = object()

    def __init__(self):
        self.calls = []

    def create_allreduce_fusion_workspace(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeWorkspace(kwargs["backend"], kwargs["world_size"])

    def allreduce_fusion(
        self,
        *,
        input,
        workspace,
        residual_out,
        norm_out,
        residual_in,
        rms_gamma,
        rms_eps,
        **_kwargs,
    ):
        allreduced = input * workspace.world_size
        expected_residual = allreduced + residual_in
        variance = expected_residual.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        expected_norm = (
            expected_residual.to(torch.float32)
            * torch.rsqrt(variance + rms_eps)
            * rms_gamma.to(torch.float32)
        ).to(input.dtype)
        residual_out.copy_(expected_residual)
        norm_out.copy_(expected_norm)


def _torch_allreduce_residual_rmsnorm_baseline(
    input_tensor, residual, weight, world_size, eps
):
    allreduced = input_tensor * world_size
    residual_out = allreduced + residual
    variance = residual_out.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    norm_out = (
        residual_out.to(torch.float32)
        * torch.rsqrt(variance + eps)
        * weight.to(torch.float32)
    ).to(input_tensor.dtype)
    return norm_out, residual_out


class TestFlashInferCommFusion(unittest.TestCase):
    def test_workspace_initialization_configures_size_check(self):
        manager = fusion.FlashInferWorkspaceManager()
        workspace = _FakeMNNVLWorkspace()

        with (
            patch.object(fusion, "_flashinfer_comm", object()),
            patch.object(
                fusion,
                "_create_allreduce_fusion_workspace",
                return_value=workspace,
            ),
            patch.object(
                fusion, "_preflight_check_workspace_memory", return_value=True
            ),
        ):
            manager.initialize(
                world_size=4,
                rank=0,
                max_token_num=8,
                hidden_dim=16,
                backend="mnnvl",
                dtype=torch.bfloat16,
                use_oneshot=True,
            )

        self.assertTrue(manager.initialized)
        self.assertEqual(manager._workspace_size_check_kwarg, "strategy")
        self.assertIs(manager._workspace_size_check_strategy_type, _FakeMNNVLStrategy)

    def test_workspace_size_check_adapts_backend_strategy(self):
        manager = fusion.FlashInferWorkspaceManager()
        manager.initialized = True
        manager.world_size = 4

        manager.workspace = _FakeMNNVLWorkspace()
        manager._configure_workspace_size_check()
        with patch.object(
            fusion.inspect,
            "signature",
            side_effect=AssertionError("signature must be cached"),
        ):
            self.assertTrue(
                manager.is_buffer_size_sufficient(
                    token_num=8,
                    hidden_dim=16,
                    dtype=torch.bfloat16,
                    use_oneshot=True,
                )
            )
            self.assertEqual(manager.workspace.strategy, _FakeMNNVLStrategy.ONESHOT)
            self.assertTrue(
                manager.is_buffer_size_sufficient(
                    token_num=8,
                    hidden_dim=16,
                    dtype=torch.bfloat16,
                    use_oneshot=False,
                )
            )
            self.assertEqual(manager.workspace.strategy, _FakeMNNVLStrategy.TWOSHOT)
            self.assertTrue(
                manager.is_buffer_size_sufficient(
                    token_num=8,
                    hidden_dim=16,
                    dtype=torch.bfloat16,
                    use_oneshot=None,
                )
            )
            self.assertEqual(manager.workspace.strategy, _FakeMNNVLStrategy.AUTO)

        manager.workspace = _FakeTRTLLMWorkspace()
        manager._configure_workspace_size_check()
        with patch.object(
            fusion.inspect,
            "signature",
            side_effect=AssertionError("signature must be cached"),
        ):
            self.assertTrue(
                manager.is_buffer_size_sufficient(
                    token_num=8,
                    hidden_dim=16,
                    dtype=torch.bfloat16,
                    use_oneshot=True,
                )
            )
        self.assertTrue(manager.workspace.use_oneshot)

    def test_auto_backend_resolves_by_arch(self):
        single_node = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="auto", nnodes=1
        )
        multi_node = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="auto", nnodes=2
        )

        # Blackwell: mnnvl on both single-node and multi-node.
        with patch.object(fusion, "is_sm100_supported", return_value=True):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(single_node),
                "mnnvl",
            )
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node), "mnnvl"
            )

        # SM90: auto uses trtllm on single-node, multi-node is unsupported.
        with (
            patch.object(fusion, "is_sm100_supported", return_value=False),
            patch.object(fusion, "is_sm90_supported", return_value=True),
        ):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(single_node),
                "trtllm",
            )
            with self.assertRaises(ValueError):
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node)

        # Architectures outside SM90/SM10X are unsupported. Both pre-SM90
        # and post-SM10X devices (e.g. SM120) must fail closed.
        for arch in ("pre_sm90", "post_sm10x"):
            with (
                self.subTest(arch=arch),
                patch.object(fusion, "is_sm100_supported", return_value=False),
                patch.object(fusion, "is_sm90_supported", return_value=False),
            ):
                with self.assertRaises(ValueError):
                    fusion.resolve_flashinfer_allreduce_fusion_backend(single_node)
                with self.assertRaises(ValueError):
                    fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node)

    def test_explicit_backend_validation(self):
        single_node_mnnvl = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="mnnvl", nnodes=1
        )
        multi_node_mnnvl = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="mnnvl", nnodes=2
        )
        single_node_trtllm = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="trtllm", nnodes=1
        )
        multi_node_trtllm = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="trtllm", nnodes=2
        )

        with (
            patch.object(fusion, "is_sm100_supported", return_value=False),
            patch.object(fusion, "is_sm90_supported", return_value=True),
        ):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(single_node_mnnvl),
                "mnnvl",
            )
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(single_node_trtllm),
                "trtllm",
            )
            with self.assertRaises(ValueError):
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node_mnnvl)
            with self.assertRaises(ValueError):
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node_trtllm)

        with patch.object(fusion, "is_sm100_supported", return_value=True):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node_mnnvl),
                "mnnvl",
            )
            with self.assertRaises(ValueError):
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node_trtllm)

        for arch in ("pre_sm90", "post_sm10x"):
            with (
                self.subTest(arch=arch),
                patch.object(fusion, "is_sm100_supported", return_value=False),
                patch.object(fusion, "is_sm90_supported", return_value=False),
            ):
                for args in (
                    single_node_mnnvl,
                    multi_node_mnnvl,
                    single_node_trtllm,
                    multi_node_trtllm,
                ):
                    with self.subTest(backend=args.flashinfer_allreduce_fusion_backend):
                        with self.assertRaises(ValueError):
                            fusion.resolve_flashinfer_allreduce_fusion_backend(args)

    def test_allreduce_fusion_backends_match_torch_baseline(self):
        fake_comm = _FakeFlashInferComm()
        original_comm = fusion._flashinfer_comm
        original_create = fusion._create_allreduce_fusion_workspace
        original_unavailable = fusion._flashinfer_allreduce_unavailable
        from sglang.srt.runtime_context import get_resources

        buffers = get_resources().buffers
        manager_key = "flashinfer_fusion_attn_tp_workspace"
        original_manager = buffers.get(manager_key)
        try:
            fusion._flashinfer_comm = fake_comm
            fusion._create_allreduce_fusion_workspace = (
                fake_comm.create_allreduce_fusion_workspace
            )
            fusion._flashinfer_allreduce_unavailable = False

            for backend in ("trtllm", "mnnvl"):
                with self.subTest(backend=backend):
                    world_size = 4
                    manager = fusion.FlashInferWorkspaceManager()
                    manager.workspace = _FakeWorkspace(backend, world_size)
                    manager.initialized = True
                    buffers[manager_key] = manager
                    if not torch.cuda.is_available():
                        self.skipTest("FlashInfer allreduce custom op is CUDA-only")
                    device = torch.device("cuda")
                    torch.manual_seed(0)
                    input_tensor = torch.randn(4, 8, dtype=torch.float32, device=device)
                    residual = torch.randn(4, 8, dtype=torch.float32, device=device)
                    weight = torch.randn(8, dtype=torch.float32, device=device)
                    eps = 1e-6

                    expected_norm, expected_residual = (
                        _torch_allreduce_residual_rmsnorm_baseline(
                            input_tensor, residual, weight, world_size, eps
                        )
                    )

                    with (
                        patch.object(
                            fusion, "is_flashinfer_available", return_value=True
                        ),
                        get_parallel().override(attn_tp_size=world_size),
                        patch.object(
                            fusion, "ensure_workspace_initialized", return_value=True
                        ),
                    ):
                        norm_out, residual_out = (
                            fusion.flashinfer_allreduce_residual_rmsnorm(
                                input_tensor=input_tensor,
                                residual=residual,
                                weight=weight,
                                eps=eps,
                                max_token_num=8,
                            )
                        )

                    torch.testing.assert_close(norm_out, expected_norm)
                    torch.testing.assert_close(residual_out, expected_residual)
        finally:
            fusion._flashinfer_comm = original_comm
            fusion._create_allreduce_fusion_workspace = original_create
            if original_manager is None:
                buffers.pop(manager_key, None)
            else:
                buffers[manager_key] = original_manager
            fusion._flashinfer_allreduce_unavailable = original_unavailable


if __name__ == "__main__":
    unittest.main()
