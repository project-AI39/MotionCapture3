import os
import glob

# ベースディレクトリ
base_dir = "./bord"

# ベースディレクトリ内のサブディレクトリを取得
subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# 画像パスを格納するリスト
image_paths_list = []

# サブディレクトリごとに処理
for subdir in subdirs:
    # サブディレクトリ内の .png ファイルのパスを取得
    image_paths = glob.glob(os.path.join(base_dir, subdir, "*.png"))

    # サブディレクトリごとの画像パスリストを追加
    image_paths_list.append(image_paths)

print("画像パスリスト")
print(image_paths_list)
