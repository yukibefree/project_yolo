import cv2
import numpy as np
import json
import os
import sys

# --- ファイルパス設定 (ユーザー指定) ---
# プロジェクトルートの再計算
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.append(os.path.join(PROJECT_ROOT))

# ソースファイルパス
SOURCE = os.path.join(PROJECT_ROOT, "solutions", "parking", "source")
# アノテーション対象の画像/動画ファイルを指定
VIDEO_PATH = os.path.join(SOURCE, "no_parking.png") 
# 出力するJSONファイルのパス
JSON_PATH = os.path.join(SOURCE, "bounding_boxes.json")

# --- グローバル変数 ---
# 駐車エリアのリスト。フォーマットに合わせてポイントのリストを保持する
parking_areas = [] 
current_coords = []
window_name = "Parking Area Annotator (Press 'Q' to Save & Exit)"

# --- ファイル操作ヘルパー関数 ---
def load_existing_annotations(path):
    """既存のJSONファイルからアノテーションを読み込み、グローバル変数に設定する"""
    global parking_areas
    
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                # ルートがリストであることを期待して読み込む
                data = json.load(f)
                
                # データがリストであり、"points"キーを持つオブジェクトのリストであることを確認
                if isinstance(data, list) and all("points" in item for item in data):
                    # 新しいフォーマットに合わせて "points" の値のみを保持
                    parking_areas = data
                    print(f"INFO: {len(parking_areas)}個の既存エリアを読み込みました。")
                    return
                else:
                    print(f"WARNING: {path} のJSONフォーマットが予期された形式と異なります。新しいファイルを作成します。")

        except json.JSONDecodeError:
            print(f"WARNING: {path} の読み込みに失敗しました。新しいファイルを作成します。")
        except Exception as e:
            print(f"ERROR: 既存ファイルをロード中に予期せぬエラー: {e}")
            
    # ロードに失敗した場合、空のリストで開始
    parking_areas = []
    print("INFO: 新しいアノテーションを開始します。")

# --- マウスイベントコールバック関数 ---
def mouse_callback(event, x, y, flags, param):
    """マウスイベントを処理し、座標を記録する"""
    global current_coords, parking_areas

    if event == cv2.EVENT_LBUTTONDOWN:
        # 左クリックで座標を記録
        current_coords.append([x, y])
        print(f"座標 {len(current_coords)}: ({x}, {y})")

        # 4点記録したら、エリアとして確定
        if len(current_coords) == 4:
            # 新しいフォーマットに合わせて "points" オブジェクトとして追加
            parking_areas.append({
                "points": current_coords
            })
            area_count = len(parking_areas)
            print(f"--- 駐車エリア {area_count} 記録完了。次のエリアへ (クリックを続けてください) ---")
            current_coords = []
            
# --- メインロジック ---
def run_annotator():
    """アノテーションプロセスを実行する"""
    global VIDEO_PATH
    
    # 既存のデータをロード
    load_existing_annotations(JSON_PATH)

    # -----------------------------------------------
    # 動画/静止画の読み込みロジックを統合
    # -----------------------------------------------
    
    if not os.path.exists(VIDEO_PATH):
        print(f"ERROR: ファイル '{VIDEO_PATH}' が見つかりません。パスを確認してください。")
        sys.exit(1)

    # ファイル拡張子に基づいて動画か静止画かを判定
    file_ext = os.path.splitext(VIDEO_PATH)[1].lower()

    if file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
        # 動画ファイルの場合: 最初のフレームを取得
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"ERROR: 動画ファイル '{VIDEO_PATH}' を読み込めません。OpenCVのコーデックを確認してください。")
            sys.exit(1)
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("ERROR: 動画からフレームを読み込めませんでした。")
            sys.exit(1)
            
    elif file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
        # 静止画ファイルの場合: 画像を直接読み込む
        frame = cv2.imread(VIDEO_PATH)
        
        if frame is None:
            print(f"ERROR: 静止画ファイル '{VIDEO_PATH}' を読み込めませんでした。")
            sys.exit(1)
            
    else:
        print(f"ERROR: サポートされていないファイル形式 '{file_ext}' です。")
        sys.exit(1)

    # -----------------------------------------------
    # アノテーションループ
    # -----------------------------------------------

    # ウィンドウを作成し、コールバックを設定
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        temp_frame = frame.copy()
        
        # 記録済みのエリアを描画 (青色)
        for i, area in enumerate(parking_areas):
            # 新しいフォーマットでは "points" キーから座標を取得
            coords = area['points']
            pts = np.array(coords, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(temp_frame, [pts], True, (255, 0, 0), 3) # 青色の太線
            
            # IDテキスト (リストのインデックス + 1 を ID として表示)
            area_id_display = i + 1
            cv2.putText(temp_frame, f"ID: {area_id_display}", 
                        (coords[0][0], coords[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # 現在記録中の点を描画 (緑色)
        if current_coords:
            for i in range(len(current_coords)):
                (x, y) = current_coords[i]
                # 点
                cv2.circle(temp_frame, (x, y), 8, (0, 255, 0), -1) # 緑色の塗りつぶし円
                
                # 線 (前の点と現在の点を結ぶ)
                if i > 0:
                    cv2.line(temp_frame, tuple(current_coords[i-1]), (x, y), (0, 255, 0), 2)
                
                # 4点目をクリックする前に、最初の点と現在の点を結ぶ仮の線
                if len(current_coords) == 3 and i == 2:
                    cv2.line(temp_frame, (x, y), tuple(current_coords[0]), (0, 255, 0), 2)


        cv2.imshow(window_name, temp_frame)

        key = cv2.waitKey(20) & 0xFF
        
        # 'q' キーでループを終了
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # 最終結果をJSONに保存
    if parking_areas:
        # ルートを配列とし、各要素を "points" を持つオブジェクトとして保存
        with open(JSON_PATH, 'w') as f:
            json.dump(parking_areas, f, indent=4) # parking_areas は既に希望のリスト構造
        print(f"\n--- 完了 ---")
        print(f"{len(parking_areas)} 個の駐車エリアを {JSON_PATH} に保存しました。")
    else:
        print("\nアノテーションは保存されませんでした。")

if __name__ == '__main__':
    run_annotator()
