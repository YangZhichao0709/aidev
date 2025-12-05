# ベースイメージを 2.0.1 から 2.1.0 に変更（これが決定打です！）
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel

# 作業ディレクトリ
WORKDIR /app

# キャッシュ無効化のための日付（念のため更新）
ENV REBUILD_DATE=20251205-v2

# 必要なファイルをコピー
COPY requirements.txt .

# pipアップグレードとライブラリインストール
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ソースコードをコピー
COPY . .

# 起動コマンド
CMD [ "python", "-u", "handler.py" ]