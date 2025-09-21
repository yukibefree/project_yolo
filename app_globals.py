from tracker.tracker import Tracker

class AppGlobals:
    """アプリケーション全体で共有されるグローバルインスタンスを保持するクラス"""
    def __init__(self):
        self.tracker = None
        self.ws_tracker = None

globals = AppGlobals()