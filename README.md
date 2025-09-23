
## 🛠️ 開発環境構築手順

本プロジェクトは、YOLOv8によるオブジェクトトラッキングとFastAPIによるリアルタイム配信を組み合わせたものです。以下の手順に従って、開発環境を構築してください。

### **1. 前提条件**

  * Python 3.8以上
  * pip (Pythonのパッケージマネージャー)
  * Git

### **2. リポジトリのクローン**

まず、本プロジェクトのリポジトリをローカルにクローンします。

```sh
git clone https://github.com/yukibefree/project_yolo.git
cd project_yolo
```

### **3. 仮想環境の作成と必要なライブラリのインストール**

本プロジェクトの依存関係を管理するため、uvを使用します。

```sh
# pythonをインストール
# https://www.python.org/

# バージョン確認
python --version

# uvをインストール
pip install uv

# 仮想環境を作成し、必要なライブラリをインストール
uv venv

# 依存関係の同期
uv sync

# ロックファイルの更新
uv lock
```

### **4. 実行**

uvを使用してアプリケーションを起動します。

```sh
# 仮想環境を有効化した状態で実行
uvicorn main:app --reload

# または、仮想環境を有効化せずにuv runで実行
uv run uvicorn main:app --reload

# streamlitで実行
uv run streamlit run streamlit/app.py
```

  * `main:app` は、`main.py` ファイル内の `FastAPI` インスタンス名 `app` を指します。
  * `--reload` オプションは、コードの変更を検知してサーバーを自動的に再起動します。

サーバーが起動したら、ブラウザで **`http://127.0.0.1:8000`** にアクセスし、フロントエンドのインターフェースを確認してください。
