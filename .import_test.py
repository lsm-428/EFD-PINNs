modules = [
    'ewp_pinn_model',
    'ewp_pinn_input_layer',
    'ewp_pinn_output_layer',
    'ewp_pinn_config',
    'ewp_pinn_performance_monitor',
    'ewp_pinn_adaptive_hyperoptimizer',
    'ewp_pinn_optimizer',
    'ewp_pinn_training_tracker',
    'ewp_pinn_physics',
    'ewp_pinn_regularization',
    'ewp_pinn_optimized_train',
    'ewp_pinn_optimized_architecture',
    'ewp_pinn_ensemble',
    'ewp_pinn_integrated',
]
print('开始模块导入测试')
for m in modules:
    try:
        __import__(m)
        print('OK: {}'.format(m))
    except Exception as e:
        print('ERR: {}: {}'.format(m, e))
        import traceback
        traceback.print_exc()
print('导入测试完成')
