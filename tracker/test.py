import yaml

# YAMLファイルを読み込む関数
def load_yaml(file_path):
  with open(file_path, 'r') as file:
    data = yaml.safe_load(file)
  return data

# YAMLファイルのパス
file_path = 'models/coco.yaml'

# YAMLファイルを読み込み、データを取得
data = load_yaml(file_path)

# 読み込んだデータを表示
print(data['names'])