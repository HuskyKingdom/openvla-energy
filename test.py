from huggingface_hub import snapshot_download

# 下载整个 repo 到本地 ./models 目录下
local_dir = snapshot_download(
    repo_id="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
    local_dir="./ckpts/pre-trained",
    local_dir_use_symlinks=False  # 避免符号链接，方便直接使用
)

print("模型已下载到:", local_dir)