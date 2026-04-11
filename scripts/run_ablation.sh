#!/bin/bash
# =============================================================================
# EFD3D 消融实验批量运行脚本
# =============================================================================
# 用法:
#   ./run_ablation.sh [实验名称]
#   ./run_ablation.sh all          # 运行所有实验
#   ./run_ablation.sh no_continuity # 运行单个实验
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="/home/scnu/Gitee/EFD3D"
CONFIG_DIR="$PROJECT_ROOT/config/ablation"
OUTPUT_DIR="$PROJECT_ROOT/outputs/ablation"

# 所有消融实验配置
ABLATIONS=(
    "no_continuity"
    "no_vof"
    "no_interface"
    "single_stage"
    "smaller_network"
)

# 打印帮助信息
print_help() {
    echo "EFD3D 消融实验批量运行脚本"
    echo ""
    echo "用法:"
    echo "  $0 [实验名称|all|list|help]"
    echo ""
    echo "可用实验:"
    echo "  - no_continuity   : 移除连续性约束 (∇·u = 0)"
    echo "  - no_vof          : 移除VOF输运方程"
    echo "  - no_interface    : 移除界面损失"
    echo "  - single_stage    : 单阶段训练 (无课程学习)"
    echo "  - smaller_network : 更小的网络架构"
    echo ""
    echo "示例:"
    echo "  $0 all            # 运行所有消融实验"
    echo "  $0 no_continuity  # 运行单个实验"
    echo "  $0 list           # 列出所有可用实验"
    echo ""
    echo "预计时间: 每个实验约 21.6 小时 (60K epochs)"
    echo "总计: 约 108 小时 (5个实验)"
}

# 检查配置文件是否存在
check_config() {
    local config_name="$1"
    local config_path="$CONFIG_DIR/${config_name}.json"
    
    if [[ ! -f "$config_path" ]]; then
        echo -e "${RED}错误: 配置文件不存在: $config_path${NC}"
        return 1
    fi
    return 0
}

# 运行单个实验
run_experiment() {
    local config_name="$1"
    local config_path="$CONFIG_DIR/${config_name}.json"
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local exp_output_dir="$OUTPUT_DIR/${config_name}_${timestamp}"
    
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}实验: $config_name${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo -e "配置: $config_path"
    echo -e "输出: $exp_output_dir"
    echo -e "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # 创建输出目录
    mkdir -p "$exp_output_dir"
    
    # 复制配置文件到输出目录
    cp "$config_path" "$exp_output_dir/config.json"
    
    # 切换到项目目录
    cd "$PROJECT_ROOT"
    
    # 运行训练
    echo -e "${YELLOW}开始训练...${NC}"
    
    # 检查是否有GPU
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        echo -e "${GREEN}检测到GPU，使用CUDA...${NC}"
        python train_two_phase.py --config "$config_path" 2>&1 | tee "$exp_output_dir/training.log"
    else
        echo -e "${YELLOW}未检测到GPU，使用CPU (速度较慢)...${NC}"
        python train_two_phase.py --config "$config_path" 2>&1 | tee "$exp_output_dir/training.log"
    fi
    
    local exit_code=${PIPESTATUS[0]}
    
    if [[ $exit_code -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}✓ 实验完成: $config_name${NC}"
        echo -e "结果保存在: $exp_output_dir"
        echo -e "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo ""
        echo -e "${RED}✗ 实验失败: $config_name (exit code: $exit_code)${NC}"
    fi
    
    return $exit_code
}

# 生成对比报告
generate_report() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}生成消融实验对比报告${NC}"
    echo -e "${BLUE}============================================${NC}"
    
    local report_path="$OUTPUT_DIR/ablation_report.txt"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "EFD3D 消融实验对比报告" > "$report_path"
    echo "生成时间: $timestamp" >> "$report_path"
    echo "=======================================" >> "$report_path"
    echo "" >> "$report_path"
    
    for exp in "${ABLATIONS[@]}"; do
        # 查找最新的实验目录
        local exp_dir=$(ls -dt "$OUTPUT_DIR/${exp}_"* 2>/dev/null | head -1)
        if [[ -d "$exp_dir" ]]; then
            local final_loss=$(grep -oP "Final Loss: \K[0-9.]+" "$exp_dir/training.log" 2>/dev/null | tail -1 || echo "N/A")
            local volume_error=$(grep -oP "Volume Error: \K[0-9.]+%" "$exp_dir/training.log" 2>/dev/null | tail -1 || echo "N/A")
            
            echo "实验: $exp" >> "$report_path"
            echo "  输出目录: $exp_dir" >> "$report_path"
            echo "  最终损失: $final_loss" >> "$report_path"
            echo "  体积误差: $volume_error" >> "$report_path"
            echo "" >> "$report_path"
        else
            echo "实验: $exp" >> "$report_path"
            echo "  状态: 未运行" >> "$report_path"
            echo "" >> "$report_path"
        fi
    done
    
    echo -e "${GREEN}报告已生成: $report_path${NC}"
    cat "$report_path"
}

# 主程序
main() {
    local action="${1:-help}"
    
    case "$action" in
        "all")
            echo -e "${GREEN}运行所有消融实验...${NC}"
            echo -e "${YELLOW}预计总时间: 约 108 小时 (5个实验 × 21.6小时)${NC}"
            echo ""
            read -p "确认开始? (y/n): " confirm
            if [[ "$confirm" != "y" ]]; then
                echo "已取消"
                exit 0
            fi
            
            for exp in "${ABLATIONS[@]}"; do
                if check_config "$exp"; then
                    run_experiment "$exp" || true
                fi
            done
            generate_report
            ;;
        "list")
            echo "可用的消融实验配置:"
            echo ""
            for exp in "${ABLATIONS[@]}"; do
                local desc=$(grep '"description"' "$CONFIG_DIR/${exp}.json" 2>/dev/null | cut -d'"' -f4 || echo "")
                printf "  %-20s %s\n" "$exp" "$desc"
            done
            echo ""
            echo "运行命令示例:"
            echo "  ./run_ablation.sh no_continuity"
            echo "  ./run_ablation.sh all"
            ;;
        "report")
            generate_report
            ;;
        "help"|"-h"|"--help")
            print_help
            ;;
        *)
            # 运行指定实验
            if check_config "$action"; then
                run_experiment "$action"
            else
                echo -e "${RED}未知实验: $action${NC}"
                print_help
                exit 1
            fi
            ;;
    esac
}

main "$@"
