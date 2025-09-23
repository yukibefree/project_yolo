import cv2
import os

os.environ['OPENCV_LOG_LEVEL'] = 'OFF'

class SelectCamera:
    def __init__(self):
        self.camera_index = None
        self.camera_list = []
        self.camera_info = {}

    def _get_all_cameras(self):
        """
        Finds all connected cameras and collects their information.
        """
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    info = {
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "fps": cap.get(cv2.CAP_PROP_FPS),
                        "backend": cap.get(cv2.CAP_PROP_BACKEND)
                    }
                    self.camera_list.append(i)
                    self.camera_info[i] = info
                    cap.release()
            except Exception as e:
                # Log the exception for debugging, but don't stop the loop.
                print(f"Error checking camera {i}: {e}")
        
        if not self.camera_list:
            print("⚠️ カメラが見つかりませんでした。")
            raise RuntimeError("Camera not found.")
        return True

    def _display_camera_info(self):
        """
        Displays information for all found cameras to the user.
        """
        print("--- 接続されているカメラ ---")
        for index in self.camera_list:
            info = self.camera_info[index]
            print(f"カメラ {index}:")
            print(f"  解像度: {info['width']} x {info['height']}")
            print(f"  フレームレート (FPS): {info['fps']}")
            print(f"  バックエンドAPI: {info['backend']}")
        print("---------------------------")

    def _get_user_selection(self):
        """
        Prompts the user to select a camera and validates the input.
        """
        while True:
            try:
                selected_index = int(input(f"使用するカメラの番号を選択してください ({', '.join(map(str, self.camera_list))}): "))
                if selected_index in self.camera_list:
                    return selected_index
                else:
                    print("❌ 無効な番号です。もう一度入力してください。")
            except ValueError:
                print("❌ 数字を入力してください。")

    def get_camera_index(self):
        """
        Main public method to orchestrate camera selection.
        """
        if self._get_all_cameras():
            self._display_camera_info()
            self.camera_index = self._get_user_selection()
            return self.camera_index
        else:
            return None

# Example usage
if __name__ == "__main__":
    camera_selector = SelectCamera()
    selected_camera = camera_selector.get_camera_index()
    if selected_camera is not None:
        print(f"✅ 選択されたカメラ: {selected_camera}")
    else:
        print("アプリケーションを終了します。")