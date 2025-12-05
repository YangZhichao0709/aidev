# RunPodが用意しているPyTorch入りのベースイメージを使う（CUDA 11.8 / PyTorch 2.0）
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

# 作業ディレクトリを設定
WORKDIR /app

# 必要なファイルをコピー
COPY requirements.txt .

# ライブラリのインストール
# (--no-cache-dir はイメージサイズを小さくするためのおまじない)
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードを全部コピー
COPY . .

# コンテナ起動時に実行するコマンド
CMD [ "python", "-u", "handler.py" ]