# 開発環境・言語パッケージ管理ガイドライン (FedProto 再現プロジェクト用)

このプロジェクトは、以下の論文の再現実験および開発を目的としています。
**論文:** [FedProto: Federated Prototype Learning across Heterogeneous Clients (arXiv:2105.00243)](https://arxiv.org/abs/2105.00243)

[cite_start]Geminiがコード生成や提案を行う際は、連合学習 (Federated Learning) における「勾配」ではなく「プロトタイプ (抽象化されたクラス表現)」を通信・集約するという本論文の核心 [cite: 8, 34] を理解し、以下のルールを厳守してください。

## 1. ツールチェーン管理 (mise)
ランタイム（Python等）および開発ツールのバージョン管理には `mise` を使用します。

- **設定ファイル:** `mise.toml`
- **ランタイム管理:** Python インタープリタ等の切り替えは `mise` を介して提案してください。
- **タスク実行:** `mise run <task_name>` (例: `mise run train`) の形式を検討してください。

## 2. Python パッケージ管理 (uv)
パッケージ追加、仮想環境運用には `uv` を使用します。`pip` や `venv` を直接呼び出すことは厳禁です。

- **パッケージ追加:** `uv add <package>` (例: `uv add torch torchvision`)
- **スクリプト実行:** 全ての Python 実行は `uv run <script_name>` で行ってください。
- **管理ファイル:** `pyproject.toml` および `uv.lock` を正しく維持してください。

## 3. FedProto 再現実験に関する固有ルール
- [cite_start]**プロトタイプ集約ロジック:** サーバーとクライアント間でモデルの重み（Gradients/Weights）ではなく、クラスごとの平均表現である「プロトタイプ」をやり取りするロジック [cite: 33, 34, 188] を優先して実装してください。
- [cite_start]**損失関数の構成:** ローカルの分類誤差 $\mathcal{L}_S$ と、グローバルプロトタイプとの距離を測る正則化項 $\mathcal{L}_R$ の両方を考慮した損失関数（論文の式 7  参照）を提案してください。
- [cite_start]**異種環境の許容:** クライアントごとにモデルアーキテクチャが異なる状況（Model Heterogeneity） [cite: 36, 127] を想定した設計を行ってください。
- **Antigravity:** プロジェクト固有のフレームワーク（Antigravity）に FedProto の各コンポーネントを適合させてください。

## 4. 禁止事項
- **`pip install` の提案:** 代わりに `uv add` を使用。
- **`venv` の手動作成:** `uv` の自動管理を利用。
- [cite_start]**重み平均 (FedAvg) の強制:** FedProto の目的は重み平均を行わずに学習することであるため、安易に `FedAvg` を提案しない [cite: 125, 148]。

## 5. 出力ルール
- 回答は常に **日本語** で行ってください。
- [cite_start]理論的な裏付けが必要な場合は、論文内の Assumption 1-4 [cite: 212, 218, 222, 230] [cite_start]や Convergence Analysis [cite: 209, 523] に基づいた説明を添えてください。
