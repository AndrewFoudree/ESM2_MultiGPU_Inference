"""
Tests for GPU Distributor functionality.

Tests the batch distribution logic across different GPU configurations.
"""


class TestGPUDistributor:
    """Test suite for GPUDistributor class."""

    def test_detect_4_gpus(self, mock_torch_cuda):
        """Test detection of 4 GPUs."""
        from app.model_manager import GPUDistributor

        distributor = GPUDistributor()

        assert distributor.gpu_count == 4
        assert distributor.gpu_ids == [0, 1, 2, 3]
        assert distributor.device_type == "cuda"

    def test_detect_8_gpus(self, mock_torch_cuda_8gpu):
        """Test detection of 8 GPUs (g5.48xlarge/p4d.24xlarge)."""
        from app.model_manager import GPUDistributor

        distributor = GPUDistributor()

        assert distributor.gpu_count == 8
        assert distributor.gpu_ids == [0, 1, 2, 3, 4, 5, 6, 7]
        assert distributor.device_type == "cuda"

    def test_detect_no_gpus(self, mock_torch_no_cuda):
        """Test fallback to CPU when no GPUs available."""
        from app.model_manager import GPUDistributor

        distributor = GPUDistributor()

        assert distributor.gpu_count == 0
        assert distributor.gpu_ids == []
        assert distributor.device_type == "cpu"

    def test_distribute_batch_even_split_4_gpus(self, mock_torch_cuda):
        """Test even distribution: 8 sequences across 4 GPUs."""
        from app.model_manager import GPUDistributor

        distributor = GPUDistributor()
        sequences = [f"SEQ{i}" for i in range(8)]

        distribution = distributor.distribute_batch(sequences)

        # Should have 4 GPU assignments
        assert len(distribution) == 4

        # Each GPU should get 2 sequences
        for gpu_id, indices, seqs in distribution:
            assert len(indices) == 2
            assert len(seqs) == 2

        # Verify correct assignment
        assert distribution[0] == (0, [0, 1], ["SEQ0", "SEQ1"])
        assert distribution[1] == (1, [2, 3], ["SEQ2", "SEQ3"])
        assert distribution[2] == (2, [4, 5], ["SEQ4", "SEQ5"])
        assert distribution[3] == (3, [6, 7], ["SEQ6", "SEQ7"])

    def test_distribute_batch_uneven_split_4_gpus(self, mock_torch_cuda):
        """Test uneven distribution: 10 sequences across 4 GPUs."""
        from app.model_manager import GPUDistributor

        distributor = GPUDistributor()
        sequences = [f"SEQ{i}" for i in range(10)]

        distribution = distributor.distribute_batch(sequences)

        # First 2 GPUs get 3 sequences, last 2 get 2
        # 10 // 4 = 2 base, 10 % 4 = 2 remainder
        assert distribution[0] == (0, [0, 1, 2], ["SEQ0", "SEQ1", "SEQ2"])
        assert distribution[1] == (1, [3, 4, 5], ["SEQ3", "SEQ4", "SEQ5"])
        assert distribution[2] == (2, [6, 7], ["SEQ6", "SEQ7"])
        assert distribution[3] == (3, [8, 9], ["SEQ8", "SEQ9"])

    def test_distribute_batch_8_gpus_64_sequences(self, mock_torch_cuda_8gpu):
        """Test max batch: 64 sequences across 8 GPUs."""
        from app.model_manager import GPUDistributor

        distributor = GPUDistributor()
        sequences = [f"SEQ{i}" for i in range(64)]

        distribution = distributor.distribute_batch(sequences)

        # 64 / 8 = 8 sequences per GPU
        assert len(distribution) == 8

        for gpu_id, indices, seqs in distribution:
            assert len(indices) == 8
            assert len(seqs) == 8

    def test_distribute_batch_fewer_sequences_than_gpus(self, mock_torch_cuda_8gpu):
        """Test when batch size < GPU count."""
        from app.model_manager import GPUDistributor

        distributor = GPUDistributor()
        sequences = ["SEQ0", "SEQ1", "SEQ2"]  # Only 3 sequences, 8 GPUs

        distribution = distributor.distribute_batch(sequences)

        # Only first 3 GPUs should be used
        assert len(distribution) == 3

        for i, (gpu_id, indices, seqs) in enumerate(distribution):
            assert gpu_id == i
            assert len(indices) == 1
            assert len(seqs) == 1

    def test_distribute_single_sequence_8_gpus(self, mock_torch_cuda_8gpu):
        """Test single sequence uses only first GPU."""
        from app.model_manager import GPUDistributor

        distributor = GPUDistributor()
        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQL"]

        distribution = distributor.distribute_batch(sequences)

        assert len(distribution) == 1
        assert distribution[0] == (0, [0], sequences)

    def test_distribute_batch_no_gpus(self, mock_torch_no_cuda):
        """Test distribution falls back to CPU."""
        from app.model_manager import GPUDistributor

        distributor = GPUDistributor()
        sequences = [f"SEQ{i}" for i in range(10)]

        distribution = distributor.distribute_batch(sequences)

        # All sequences go to "CPU" (represented as gpu_id 0)
        assert len(distribution) == 1
        gpu_id, indices, seqs = distribution[0]
        assert gpu_id == 0
        assert indices == list(range(10))
        assert seqs == sequences

    def test_distribution_info(self, mock_torch_cuda):
        """Test distribution info generation."""
        from app.model_manager import GPUDistributor

        distributor = GPUDistributor()
        sequences = [f"SEQ{i}" for i in range(10)]

        info = distributor.get_distribution_info(sequences)

        assert info.total_sequences == 10
        assert info.num_gpus_used == 4
        assert info.sequences_per_gpu["gpu_0"] == 3
        assert info.sequences_per_gpu["gpu_1"] == 3
        assert info.sequences_per_gpu["gpu_2"] == 2
        assert info.sequences_per_gpu["gpu_3"] == 2

    def test_get_device_cuda(self, mock_torch_cuda):
        """Test getting CUDA device."""
        import torch

        from app.model_manager import GPUDistributor

        distributor = GPUDistributor()

        device = distributor.get_device(0)
        assert device == torch.device("cuda:0")

        device = distributor.get_device(3)
        assert device == torch.device("cuda:3")

    def test_get_device_cpu_fallback(self, mock_torch_no_cuda):
        """Test CPU fallback for device."""
        import torch

        from app.model_manager import GPUDistributor

        distributor = GPUDistributor()

        device = distributor.get_device(0)
        assert device == torch.device("cpu")


class TestGPUDistributionAlgorithm:
    """Tests for the distribution algorithm correctness."""

    def test_all_sequences_assigned(self, mock_torch_cuda_8gpu):
        """Verify all sequences are assigned exactly once."""
        from app.model_manager import GPUDistributor

        distributor = GPUDistributor()

        for batch_size in [1, 7, 8, 15, 16, 32, 63, 64]:
            sequences = [f"SEQ{i}" for i in range(batch_size)]
            distribution = distributor.distribute_batch(sequences)

            # Collect all assigned indices
            all_indices = []
            for _, indices, _ in distribution:
                all_indices.extend(indices)

            # Every index should appear exactly once
            assert sorted(all_indices) == list(
                range(batch_size)
            ), f"Failed for batch_size={batch_size}"

    def test_sequences_match_indices(self, mock_torch_cuda_8gpu):
        """Verify sequences match their indices."""
        from app.model_manager import GPUDistributor

        distributor = GPUDistributor()
        sequences = [f"SEQ{i}" for i in range(64)]

        distribution = distributor.distribute_batch(sequences)

        for gpu_id, indices, seqs in distribution:
            for idx, seq in zip(indices, seqs):
                assert seq == f"SEQ{idx}"

    def test_distribution_is_balanced(self, mock_torch_cuda_8gpu):
        """Verify load is balanced (max diff of 1 between GPUs)."""
        from app.model_manager import GPUDistributor

        distributor = GPUDistributor()

        for batch_size in range(1, 65):
            sequences = [f"SEQ{i}" for i in range(batch_size)]
            distribution = distributor.distribute_batch(sequences)

            counts = [len(indices) for _, indices, _ in distribution]

            # Max difference between any two GPUs should be at most 1
            if counts:
                assert (
                    max(counts) - min(counts) <= 1
                ), f"Unbalanced distribution for batch_size={batch_size}"
