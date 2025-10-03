#!/bin/bash

# ==============================================================================
#  一键转换 h5ad 到 vcc 的脚本
#  使用方法:
#  1. 将此脚本保存为 convert_to_vcc.sh
#  2. 在终端中给予它执行权限: chmod +x convert_to_vcc.sh
#  3. 运行脚本并传入一个或多个 h5ad 文件作为参数:
#     ./convert_to_vcc.sh /path/to/your/file1.h5ad /path/to/another/file2.h5ad
# ==============================================================================

# --- 配置区 ---
# 请将这里的路径确认为您存放 gene_names.csv 的准确位置
GENE_LIST="../vcc_data/gene_names.csv"
# --- 配置结束 ---


# 检查是否提供了至少一个文件作为参数
if [ "$#" -eq 0 ]; then
    echo "错误: 请提供至少一个 .h5ad 文件路径作为参数。"
    echo "用法: $0 <文件1.h5ad> [文件2.h5ad] [...]"
    exit 1
fi

# 检查基因列表文件是否存在
if [ ! -f "$GENE_LIST" ]; then
    echo "错误: 基因列表文件未找到于 '$GENE_LIST'"
    echo "请检查脚本中的 GENE_LIST 变量路径是否正确。"
    exit 1
fi

# 循环处理所有传入的参数（文件路径）
for H5AD_FILE in "$@"
do
    echo "----------------------------------------"
    # 检查文件是否存在且是 .h5ad 文件
    if [[ -f "$H5AD_FILE" && "$H5AD_FILE" == *.h5ad ]]; then
        
        # 通过移除 .h5ad 后缀并添加 .vcc 来创建输出文件名
        OUTPUT_VCC_FILE="${H5AD_FILE%.h5ad}.vcc"
        
        echo "正在处理: $H5AD_FILE"
        echo "输出文件: $OUTPUT_VCC_FILE"
        
        # 执行转换命令
        cell-eval prep \
            -i "$H5AD_FILE" \
            -g "$GENE_LIST" \
            -o "$OUTPUT_VCC_FILE"
        
        # 检查命令是否成功执行
        if [ $? -eq 0 ]; then
            echo "成功转换为 $OUTPUT_VCC_FILE"
        else
            echo "处理 $H5AD_FILE 时发生错误。"
        fi
    else
        echo "跳过: '$H5AD_FILE' 不是一个有效的 .h5ad 文件或文件不存在。"
    fi
done

echo "----------------------------------------"
echo "所有转换任务已完成。"