# CLI 参考

## 优化训练（ewp_pinn_optimized_train.py）
- 关键参数：`--mode {train,test,infer}`、`--config`、`--resume/--checkpoint`、`--mixed-precision`、`--efficient-architecture`、`--model-compression`、`--model-path`、`--export-onnx`、`--num-samples`、`--data-augmentation`、`--output-dir`、`--device`
- 入口位置：`ewp_pinn_optimized_train.py:2413`
- 参数定义大致位置：`ewp_pinn_optimized_train.py:2261`

## 增强训练（run_enhanced_training.py）
- 关键参数：`--config`、`--output_dir`、`--device`、`--num_samples`、`--resume/--checkpoint`、`--mixed_precision`、`--generate_data_only`、`--validate_only`、`--model_path`、`--visualize`、`--quick_run`、`--clean/--clean_all`、`--seed/--deterministic`、`--debug_capture_nan`、`--clip_grad`、`--override_lr`
- 入口位置：`run_enhanced_training.py:2420`

## 长期训练（long_term_training.py）
- 关键参数：`--output_dir`、`--config`、`--epochs`、`--lr`、`--min_lr`、`--warmup_epochs`、`--batch_size`、`--physics_weight`、`--dynamic_weight`、`--weight_strategy`、`--checkpoint_interval`、`--validation_interval`、`--resume`、`--seed`、`--device`、`--use_3d_mapping`
- 入口位置：`long_term_training.py:615`
