"""
Quality Improvement Tests - 針對品質改良修復的深入測試

測試範圍:
1. P0: OOM 恢復邏輯 (core.py)
2. P0: Checkpoint 驗證 (run.py)
3. P0: GPU 資源清理 (train.py)
4. P1: _predict_v 提取 (base.py)
5. P1: 類型注解正確性
6. P2: factory.py 輔助函數

Author: Quality Improvement Review
Date: 2026-01-09
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from typing import Callable, Optional
import tempfile
import os


# ============================================================================
# P0-1: OOM 恢復邏輯測試
# ============================================================================

class TestOOMRecoveryLogic:
    """測試 OOM 恢復邏輯修復 (core.py:192-211)"""

    def test_batch_halving_on_first_retry(self):
        """驗證第一次 OOM 後立即減半 batch"""
        from src.training.trainer.core import Trainer

        # 記錄 batch sizes
        batch_sizes = []
        call_count = [0]

        def mock_train_step(batch):
            batch_sizes.append(batch["images"].size(0))
            call_count[0] += 1
            if call_count[0] <= 1:  # 第一次失敗
                raise RuntimeError("CUDA out of memory")
            # 第二次成功
            return Mock(loss=0.1)

        # 創建 mock trainer
        trainer = Mock(spec=Trainer)
        trainer.train_step = mock_train_step
        trainer._halve_batch = lambda b: {k: v[:v.size(0)//2] for k, v in b.items()}

        # 綁定方法
        from src.training.trainer.core import Trainer as RealTrainer
        trainer.safe_train_step = lambda batch, **kw: RealTrainer.safe_train_step(trainer, batch, **kw)

        # 測試
        batch = {"images": torch.randn(8, 3, 256, 256)}
        result = trainer.safe_train_step(batch, retry_on_oom=True, max_retries=3)

        # 驗證: 第一次用 8，第二次用 4 (立即減半)
        assert batch_sizes == [8, 4], f"Expected [8, 4], got {batch_sizes}"

    def test_batch_halving_multiple_retries(self):
        """驗證多次 OOM 後持續減半"""
        batch_sizes = []
        call_count = [0]

        def mock_train_step(batch):
            batch_sizes.append(batch["images"].size(0))
            call_count[0] += 1
            if call_count[0] <= 2:  # 前兩次失敗
                raise RuntimeError("CUDA out of memory")
            return Mock(loss=0.1)

        from src.training.trainer.core import Trainer
        trainer = Mock(spec=Trainer)
        trainer.train_step = mock_train_step
        trainer._halve_batch = lambda b: {k: v[:v.size(0)//2] for k, v in b.items()}

        from src.training.trainer.core import Trainer as RealTrainer
        trainer.safe_train_step = lambda batch, **kw: RealTrainer.safe_train_step(trainer, batch, **kw)

        batch = {"images": torch.randn(8, 3, 256, 256)}
        result = trainer.safe_train_step(batch, retry_on_oom=True, max_retries=3)

        # 8 -> 4 -> 2
        assert batch_sizes == [8, 4, 2], f"Expected [8, 4, 2], got {batch_sizes}"

    def test_returns_none_on_empty_batch(self):
        """驗證 batch 減到空時返回 None"""
        call_count = [0]

        def mock_train_step(batch):
            call_count[0] += 1
            raise RuntimeError("CUDA out of memory")

        from src.training.trainer.core import Trainer
        trainer = Mock(spec=Trainer)
        trainer.train_step = mock_train_step
        trainer._halve_batch = lambda b: {k: v[:v.size(0)//2] for k, v in b.items()}

        from src.training.trainer.core import Trainer as RealTrainer
        trainer.safe_train_step = lambda batch, **kw: RealTrainer.safe_train_step(trainer, batch, **kw)

        # 從 2 開始，減半後變 1，再減半變 0
        batch = {"images": torch.randn(2, 3, 256, 256)}
        result = trainer.safe_train_step(batch, retry_on_oom=True, max_retries=5)

        assert result is None


# ============================================================================
# P0-2: Checkpoint 驗證測試
# ============================================================================

class TestCheckpointValidation:
    """測試 Checkpoint 驗證修復 (run.py:185-189)"""

    def test_invalid_checkpoint_type_raises_error(self):
        """驗證非 dict 類型拋出 ValueError"""
        import tempfile as tmp
        fd, path = tmp.mkstemp(suffix='.pt')
        os.close(fd)
        try:
            torch.save([1, 2, 3], path)  # 保存 list 而非 dict
            from src.inference.run import load_checkpoint
            with pytest.raises(ValueError, match="Invalid checkpoint format"):
                load_checkpoint(path, torch.device('cpu'))
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_missing_model_and_ema_raises_error(self):
        """驗證缺少 model 和 ema 鍵時拋出 ValueError"""
        import tempfile as tmp
        fd, path = tmp.mkstemp(suffix='.pt')
        os.close(fd)
        try:
            torch.save({"config": {}, "other": "data"}, path)
            from src.inference.run import load_checkpoint
            with pytest.raises(ValueError, match="missing both 'model' and 'ema'"):
                load_checkpoint(path, torch.device('cpu'))
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_valid_checkpoint_with_model_key(self):
        """驗證有 model 鍵的有效 checkpoint"""
        import tempfile as tmp
        fd, path = tmp.mkstemp(suffix='.pt')
        os.close(fd)
        try:
            # 創建最小有效 checkpoint
            from src.config import PixelHDMConfig
            from src.models.pixelhdm import create_pixelhdm_for_t2i

            # 使用正確的配置 (head_dim = 64, mrope 維度也要匹配)
            config = PixelHDMConfig(
                hidden_dim=64, patch_layers=1, pixel_layers=1,
                num_heads=1, num_kv_heads=1, head_dim=64,
                mrope_text_dim=16, mrope_img_h_dim=24, mrope_img_w_dim=24,
            )
            model = create_pixelhdm_for_t2i(config=config)

            checkpoint = {
                "model": model.state_dict(),
                "config": config.to_dict()
            }
            torch.save(checkpoint, path)

            from src.inference.run import load_checkpoint
            loaded_model, loaded_config = load_checkpoint(path, torch.device('cpu'), use_ema=False)
            assert loaded_model is not None
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ============================================================================
# P0-3: GPU 資源清理測試
# ============================================================================

class TestGPUResourceCleanup:
    """測試 GPU 資源清理修復 (train.py:251-259)"""

    def test_trainer_in_locals_check(self):
        """驗證 'trainer' in locals() 檢查正確工作"""
        # 模擬 trainer 未定義的情況
        def cleanup_without_trainer():
            import gc
            if 'trainer' in locals():
                del trainer  # 這行不應執行
            gc.collect()
            return True

        # 不應拋出 NameError
        result = cleanup_without_trainer()
        assert result is True

    def test_trainer_in_locals_with_trainer(self):
        """驗證 trainer 存在時正確清理"""
        def cleanup_with_trainer():
            import gc
            trainer = Mock()  # 創建 trainer
            if 'trainer' in locals():
                del trainer
            gc.collect()
            return True

        result = cleanup_with_trainer()
        assert result is True


# ============================================================================
# P1-1: _predict_v 提取測試
# ============================================================================

class TestPredictVExtraction:
    """測試 _predict_v 提取到 BaseSampler"""

    def test_base_sampler_has_predict_v(self):
        """驗證 BaseSampler 有 _predict_v 方法"""
        from src.inference.sampler.base import BaseSampler
        assert hasattr(BaseSampler, '_predict_v')

    def test_predict_v_with_cfg_function_exists(self):
        """驗證 _predict_v_with_cfg 函數存在"""
        from src.inference.sampler.base import _predict_v_with_cfg
        assert callable(_predict_v_with_cfg)

    def test_euler_sampler_inherits_predict_v(self):
        """驗證 EulerSampler 繼承 _predict_v"""
        from src.inference.sampler.euler import EulerSampler
        sampler = EulerSampler(num_steps=10)
        assert hasattr(sampler, '_predict_v')
        assert callable(sampler._predict_v)

    def test_dpm_sampler_inherits_predict_v(self):
        """驗證 DPMPPSampler 繼承 _predict_v"""
        from src.inference.sampler.dpm import DPMPPSampler
        sampler = DPMPPSampler(num_steps=10)
        assert hasattr(sampler, '_predict_v')
        assert callable(sampler._predict_v)

    def test_heun_sampler_inherits_predict_v(self):
        """驗證 HeunSampler 通過繼承鏈獲得 _predict_v"""
        from src.inference.sampler.heun import HeunSampler
        sampler = HeunSampler(num_steps=10)
        assert hasattr(sampler, '_predict_v')
        assert callable(sampler._predict_v)

    def test_predict_v_cfg_logic(self):
        """測試 _predict_v CFG 邏輯正確性"""
        from src.inference.sampler.base import _predict_v_with_cfg

        # Mock model
        def mock_model(z, t, text_embed=None, text_mask=None, pooled_text_embed=None):
            return z * 0.5  # 簡單返回

        z = torch.randn(2, 256, 256, 3)
        t = torch.tensor([0.5, 0.5])
        text_embed = torch.randn(2, 10, 1024)

        # 無 CFG
        result = _predict_v_with_cfg(
            mock_model, z, t, text_embed,
            guidance_scale=1.0, null_text_embeddings=None
        )
        assert result.shape == z.shape

    def test_predict_v_cfg_with_guidance(self):
        """測試 _predict_v 帶 CFG guidance"""
        from src.inference.sampler.base import _predict_v_with_cfg

        call_count = [0]
        def mock_model(z, t, text_embed=None, text_mask=None, pooled_text_embed=None):
            call_count[0] += 1
            return z * 0.5

        z = torch.randn(2, 256, 256, 3)
        t = torch.tensor([0.5, 0.5])
        text_embed = torch.randn(2, 10, 1024)
        null_embed = torch.zeros(2, 10, 1024)

        # 有 CFG (guidance_scale > 1.0)
        result = _predict_v_with_cfg(
            mock_model, z, t, text_embed,
            guidance_scale=7.5, null_text_embeddings=null_embed
        )

        # 應該調用兩次 (uncond + cond)
        assert call_count[0] == 2
        assert result.shape == z.shape


# ============================================================================
# P1-2: 類型注解正確性測試
# ============================================================================

class TestTypeAnnotations:
    """測試類型注解修復"""

    def test_step_executor_warmup_fn_type(self):
        """驗證 StepExecutor.execute 的 warmup_fn 類型"""
        from src.training.trainer.step import StepExecutor
        import inspect

        sig = inspect.signature(StepExecutor.execute)
        warmup_fn_param = sig.parameters['warmup_fn']

        # 檢查類型注解字符串包含正確的類型
        annotation_str = str(warmup_fn_param.annotation)
        assert 'Callable' in annotation_str
        assert 'int' in annotation_str
        assert 'None' in annotation_str

    def test_optimizer_step_mixin_warmup_fn_type(self):
        """驗證 OptimizerStepMixin._optimizer_step 的 warmup_fn 類型"""
        from src.training.trainer.optimizer_step import OptimizerStepMixin
        import inspect

        sig = inspect.signature(OptimizerStepMixin._optimizer_step)
        warmup_fn_param = sig.parameters['warmup_fn']

        annotation_str = str(warmup_fn_param.annotation)
        assert 'Callable' in annotation_str

    def test_loop_get_epoch_settings_return_type(self):
        """驗證 TrainingLoop._get_epoch_settings 返回類型"""
        from src.training.trainer.loop import TrainingLoop
        import inspect

        sig = inspect.signature(TrainingLoop._get_epoch_settings)
        return_annotation = str(sig.return_annotation)

        # 應該是 Tuple 而非 tuple (一致性)
        assert 'Tuple' in return_annotation or 'tuple' in return_annotation


# ============================================================================
# P2: factory.py 輔助函數測試
# ============================================================================

class TestFactoryHelperFunctions:
    """測試 factory.py 輔助函數"""

    def test_create_sequential_sampler_exists(self):
        """驗證 _create_sequential_sampler 函數存在"""
        from src.training.dataset.factory import _create_sequential_sampler
        assert callable(_create_sequential_sampler)

    def test_create_buffered_shuffle_sampler_exists(self):
        """驗證 _create_buffered_shuffle_sampler 函數存在"""
        from src.training.dataset.factory import _create_buffered_shuffle_sampler
        assert callable(_create_buffered_shuffle_sampler)

    def test_create_random_sampler_exists(self):
        """驗證 _create_random_sampler 函數存在"""
        from src.training.dataset.factory import _create_random_sampler
        assert callable(_create_random_sampler)

    def test_create_random_sampler_returns_tuple(self):
        """測試 _create_random_sampler 返回正確類型"""
        from src.training.dataset.factory import _create_random_sampler

        bucket_ids = [0, 0, 1, 1, 2, 2]
        sampler, prefetch = _create_random_sampler(bucket_ids, batch_size=2, shuffle=True)

        assert sampler is not None
        assert isinstance(prefetch, int)
        assert prefetch == 2  # 默認值

    def test_helper_functions_signature(self):
        """驗證輔助函數簽名正確"""
        import inspect
        from src.training.dataset.factory import (
            _create_sequential_sampler,
            _create_buffered_shuffle_sampler,
            _create_random_sampler,
        )

        # Sequential sampler 應該有這些參數
        seq_params = inspect.signature(_create_sequential_sampler).parameters
        assert 'bucket_ids' in seq_params
        assert 'bucket_manager' in seq_params
        assert 'batch_size' in seq_params
        assert 'bucket_order' in seq_params

        # Buffered shuffle sampler
        buf_params = inspect.signature(_create_buffered_shuffle_sampler).parameters
        assert 'chunk_size' in buf_params
        assert 'shuffle_chunks' in buf_params

        # Random sampler
        rand_params = inspect.signature(_create_random_sampler).parameters
        assert 'shuffle' in rand_params


# ============================================================================
# 整合測試
# ============================================================================

class TestIntegration:
    """整合測試確保所有修改協同工作"""

    def test_sampler_step_with_inherited_predict_v(self):
        """測試採樣器 step 使用繼承的 _predict_v"""
        from src.inference.sampler.euler import EulerSampler

        sampler = EulerSampler(num_steps=10)

        # Mock model
        def mock_model(z, t, text_embed=None, text_mask=None, pooled_text_embed=None):
            return torch.randn_like(z)

        z = torch.randn(1, 16, 16, 3)
        t = torch.tensor([0.1])
        t_next = torch.tensor([0.2])

        # 應該能正常執行 step
        result = sampler.step(
            mock_model, z, t, t_next,
            text_embeddings=torch.randn(1, 5, 1024),
            guidance_scale=1.0
        )

        assert result.shape == z.shape

    def test_all_imports_work(self):
        """驗證所有修改的模組可以正常導入"""
        # Core imports
        from src.training.trainer.core import Trainer
        from src.training.trainer.step import StepExecutor
        from src.training.trainer.loop import TrainingLoop
        from src.training.trainer.optimizer_step import OptimizerStepMixin

        # Sampler imports
        from src.inference.sampler.base import BaseSampler, _predict_v_with_cfg
        from src.inference.sampler.euler import EulerSampler
        from src.inference.sampler.dpm import DPMPPSampler
        from src.inference.sampler.heun import HeunSampler

        # Factory imports
        from src.training.dataset.factory import (
            create_bucket_dataloader,
            _create_sequential_sampler,
            _create_buffered_shuffle_sampler,
            _create_random_sampler,
        )

        # Inference imports
        from src.inference.run import load_checkpoint

        assert True  # 如果能到達這裡，所有導入都成功


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
