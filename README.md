# strands-agents-human-tools

Strands Agentsを使用した人間監視エージェント。Webカメラや画面キャプチャで人間の状況を監視し、音声で通知します。

## 事前設定

### 1. 依存関係のインストール

```bash
uv sync
```

### 2. LM Studioの起動

LM Studioを起動し、ローカルサーバーを `http://localhost:1234` で実行してください。
対応モデル（例: `qwen/qwen3-vl-8b`）をロードしておく必要があります。

## 使い方

### 基本的な実行

```bash
uv run python strandsagents-humantool.py
```

対話形式で指示を入力し、AIがキャプチャモードを自動判断します。

### コマンドライン引数

| 引数 | 説明 |
|------|------|
| `-m`, `--mode` | キャプチャモード (`webcam_capture`, `screen_capture`, `alternate`) |
| `-i`, `--instruction` | エージェントへの指示文 |

### 実行例

```bash
# モードと指示の両方を指定
uv run python strandsagents-humantool.py -m screen_capture -i "人間が作業をサボっていたら警告してください"

# 指示のみ指定（モードはAIが判断）
uv run python strandsagents-humantool.py -i "人間の表情を監視してください"

# モードのみ指定（指示は対話入力）
uv run python strandsagents-humantool.py -m webcam_capture

# ヘルプを表示
uv run python strandsagents-humantool.py --help
```

### キャプチャモード

- `webcam_capture`: Webカメラで人間の顔を監視
- `screen_capture`: 画面をキャプチャしてPC操作を監視
- `alternate`: カメラと画面を交互にキャプチャ
