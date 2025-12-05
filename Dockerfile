# ベースイメージ
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

# 作業ディレクトリ
WORKDIR /app

# --- ここが変更点 ---
# キャッシュを無効にするための環境変数をセット（日付を変えれば毎回強制ビルドできます）
ENV REBUILD_DATE=20251205

# 必要なファイルをコピー
COPY requirements.txt .

# pip自体をアップグレードしてから、ライブラリをインストール
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ソースコードをコピー
COPY . .

# 起動コマンド
CMD [ "python", "-u", "handler.py" ]