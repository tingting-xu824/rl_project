#!/bin/bash

# 目录重构迁移脚本
set -e

echo "=========================================="
echo "开始目录重构迁移..."
echo "=========================================="

# 1. 创建新目录结构
echo "步骤 1: 创建新目录..."
mkdir -p configs
mkdir -p src
mkdir -p scripts
mkdir -p outputs/figures/screenshots
echo "✓ 新目录创建完成"

# 2. 移动核心源代码到 src/
echo ""
echo "步骤 2: 移动核心源代码到 src/..."
if [ -f "common.py" ]; then
    mv common.py src/
    echo "  ✓ common.py -> src/"
fi

if [ -f "cost_wrapper.py" ]; then
    mv cost_wrapper.py src/
    echo "  ✓ cost_wrapper.py -> src/"
fi

touch src/__init__.py
echo "  ✓ 创建 src/__init__.py"

# 3. 移动脚本到 scripts/
echo ""
echo "步骤 3: 移动脚本到 scripts/..."
for script in train.py eval.py eval_and_plot.py make_gui_shots.py plot_bars.py plot_reward_resets_vs_steps.py; do
    if [ -f "$script" ]; then
        mv "$script" scripts/
        echo "  ✓ $script -> scripts/"
    fi
done

# 4. 移动图片到 outputs/figures/
echo ""
echo "步骤 4: 移动图片到 outputs/figures/..."
for img in exp3_*.png shots*.png; do
    if [ -f "$img" ]; then
        mv "$img" outputs/figures/
        echo "  ✓ $img -> outputs/figures/"
    fi
done

# 5. 移动截图
echo ""
echo "步骤 5: 移动截图到 outputs/figures/screenshots/..."
for screenshot in Screenshot*.png; do
    if [ -f "$screenshot" ]; then
        mv "$screenshot" outputs/figures/screenshots/
        echo "  ✓ $screenshot -> outputs/figures/screenshots/"
    fi
done

# 6. 创建配置文件模板
echo ""
echo "步骤 6: 创建配置文件模板..."
touch configs/exp1_free.yaml
touch configs/exp2_punish.yaml
touch configs/exp3_recovery.yaml
touch configs/exp4_goal.yaml
echo "  ✓ 配置文件模板已创建"

# 7. 更新 .gitignore
echo ""
echo "步骤 7: 更新 .gitignore..."
cat >> .gitignore << 'EOF'

# Python
__pycache__/
*.pyc
*.pyo
.Python

# 环境
.venv/
venv/

# 项目输出
outputs/logs/
outputs/models/*.zip
.DS_Store

# IDE
.vscode/
.idea/
EOF
echo "  ✓ .gitignore 已更新"

echo ""
echo "=========================================="
echo "迁移完成！"
echo "==========================================" 
MIGRATE_EOF 
