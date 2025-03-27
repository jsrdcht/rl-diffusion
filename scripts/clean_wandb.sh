#!/bin/bash

# 确保已安装wandb
if ! command -v wandb &> /dev/null; then
    echo "Installing wandb..."
    pip install wandb
fi

# 获取当前项目的wandb项目名
PROJECT_NAME="ddpo-pytorch"
ENTITY_NAME="jsrdcht"  # 使用您的wandb用户名

# 检查wandb登录状态
echo "Checking wandb login status..."
wandb status

# 遍历所有wandb运行目录
for run_dir in wandb/*/; do
    if [ -f "${run_dir}files/config.yaml" ]; then
        # 获取epoch信息
        epoch=$(grep "num_epochs:" "${run_dir}files/config.yaml" | awk '{print $2}')
        
        # 如果epoch小于50，删除该目录
        if [ ! -z "$epoch" ] && [ "$epoch" -lt 30 ]; then
            # 从目录名中提取run ID
            run_id=$(basename "$run_dir" | cut -d'-' -f2)
            
            echo "Found run with epoch=$epoch: $run_dir"
            echo "Extracted run_id: $run_id"
            
            # 删除本地目录
            rm -rf "$run_dir"
            
            # 删除云端记录
            if [ ! -z "$run_id" ]; then
                echo "Attempting to delete cloud run: $ENTITY_NAME/$PROJECT_NAME/$run_id"
                # 使用curl直接调用wandb API删除记录
                API_KEY=$(grep "api.wandb.ai" ~/.netrc | awk '{print $6}')
                if [ ! -z "$API_KEY" ]; then
                    curl -X DELETE "https://api.wandb.ai/api/v1/runs/$ENTITY_NAME/$PROJECT_NAME/$run_id" \
                         -H "Authorization: Bearer $API_KEY" \
                         -H "Content-Type: application/json"
                    echo "Delete API call completed"
                else
                    echo "Error: Could not find API key in ~/.netrc"
                fi
            else
                echo "Warning: Could not extract run_id from directory: $run_dir"
            fi
        fi
    fi
done 