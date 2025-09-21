import yaml

# YAMLファイルを読み込む関数
def load_yaml(file_path):
  with open(file_path, 'r') as file:
    data = yaml.safe_load(file)
  return data