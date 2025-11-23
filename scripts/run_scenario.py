#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
import subprocess

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def copy_config(src, dst):
    ensure_dir(os.path.dirname(dst))
    shutil.copyfile(src, dst)

def run_cmd(cmd):
    r = subprocess.run(cmd, shell=True)
    return r.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', type=str, required=True, choices=['dc_step','ac_sweep','hysteresis','thermal'])
    parser.add_argument('--output_dir', type=str, default='results_enhanced')
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_dst = os.path.join(root, 'config', 'model_config.json')
    if args.scenario == 'dc_step':
        cfg_src = os.path.join(root, 'docs', 'config_samples', 'dc_step_config.json')
        copy_config(cfg_src, cfg_dst)
        ensure_dir(args.output_dir)
        code = run_cmd(f"python ewp_pinn_optimized_train.py --mode train --config {cfg_dst} --output-dir {args.output_dir}")
        if code == 0:
            run_cmd(f"python scripts/generate_constraint_report.py --model_path {args.output_dir}/checkpoints/best_model.pth --config_path {cfg_dst} --dataset_path {args.output_dir}/dataset.npz --output_dir {args.output_dir}/consistency_data --applied_voltage 20.0")
            run_cmd(f"python scripts/visualize_constraint_report.py --report_path {args.output_dir}/consistency_data/constraint_diagnostics.json --output_dir {args.output_dir}/consistency_data")
        sys.exit(code)
    if args.scenario == 'ac_sweep':
        cfg_src = os.path.join(root, 'docs', 'config_samples', 'ac_sweep_config.json')
        copy_config(cfg_src, cfg_dst)
        out = 'results_long_run'
        ensure_dir(out)
        code = run_cmd(f"python long_term_training.py --output_dir {out} --epochs 30000 --dynamic_weight --weight_strategy adaptive")
        if code == 0:
            run_cmd(f"python scripts/generate_constraint_report.py --model_path {out}/final_model.pth --config_path {cfg_dst} --dataset_path {out}/dataset.npz --output_dir {out}/consistency_data --applied_voltage 5.0 --time 1.0")
            run_cmd(f"python scripts/visualize_constraint_report.py --report_path {out}/consistency_data/constraint_diagnostics.json --output_dir {out}/consistency_data")
        sys.exit(code)
    if args.scenario == 'hysteresis':
        cfg_src = os.path.join(root, 'docs', 'config_samples', 'contact_line_hysteresis_config.json')
        copy_config(cfg_src, cfg_dst)
        ensure_dir(args.output_dir)
        code = run_cmd(f"python ewp_pinn_optimized_train.py --mode train --config {cfg_dst} --output-dir {args.output_dir}")
        if code == 0:
            run_cmd(f"python scripts/generate_constraint_report.py --model_path {args.output_dir}/checkpoints/best_model.pth --config_path {cfg_dst} --dataset_path {args.output_dir}/dataset.npz --output_dir {args.output_dir}/consistency_data --contact_line_velocity 0.02")
            run_cmd(f"python scripts/visualize_constraint_report.py --report_path {args.output_dir}/consistency_data/constraint_diagnostics.json --output_dir {args.output_dir}/consistency_data")
        sys.exit(code)
    if args.scenario == 'thermal':
        cfg_src = os.path.join(root, 'docs', 'config_samples', 'thermal_rise_config.json')
        copy_config(cfg_src, cfg_dst)
        out = 'results_long_run'
        ensure_dir(out)
        code = run_cmd(f"python long_term_training.py --output_dir {out} --epochs 30000 --dynamic_weight --weight_strategy adaptive")
        if code == 0:
            run_cmd(f"python scripts/generate_constraint_report.py --model_path {out}/final_model.pth --config_path {cfg_dst} --dataset_path {out}/dataset.npz --output_dir {out}/consistency_data --temperature 330.0")
            run_cmd(f"python scripts/visualize_constraint_report.py --report_path {out}/consistency_data/constraint_diagnostics.json --output_dir {out}/consistency_data")
        sys.exit(code)

if __name__ == '__main__':
    main()

