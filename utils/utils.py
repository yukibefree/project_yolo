import os
import sys
import subprocess
from ultralytics.utils import LOGGER

# Ultralyticsの内部ヘルパー関数と定数
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm', 'pgm', 'pbm', 'ppm', 'sr', 'ras', 'jpe', 'jp2', 'j2k', 'rle', 'dib', 'hdr', 'exr', 'pxm', 'pnm', 'sgi', 'raw', 'avif', 'heif', 'heic'
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'wmv', 'webm', 'ts', 'ogg', '3gp', '3g2', 'flv', 'fli', 'f4v', 'gvi', 'divx', 'iso', 'vob', 'dat', 'xvid', 'qt', 'h264', 'h265', 'vp9', 'av1'
FORMATS_HELP_MSG = f"Supported formats are:\n- images: {', '.join(IMG_FORMATS)}\n- videos: {', '.join(VID_FORMATS)}"

IS_COLAB = False 
IS_KAGGLE = False 

def clean_str(s):
    """ファイル名を安全な名前に変換するユーティリティ関数。"""
    return str(s).replace(os.sep, "_").replace("/", "_").replace(":", "_").replace("?", "_").replace("=", "_").replace("&", "_")

def check_requirements(requirements, install=True):
    """必要なPythonパッケージがインストールされているか確認し、なければインストールする。"""
    if isinstance(requirements, str):
        requirements = [requirements]
    for r in requirements:
        try:
            subprocess.check_call(['uv', 'pip', 'install', r])
            
            # インストール後の再インポート確認
            #__import__(r.split(">")[0].split("=")[0].split("<")[0].split("!")[0].split("[")[0].split("~")[0])
            LOGGER.info(f"'{r}' のインストールに成功しました。")
            break
        except Exception as e:
            LOGGER.error(f"'{r}' のインストールに失敗しました: {e}")
            raise RuntimeError(f"必要なパッケージ '{r}' が見つからず、インストールできませんでした。")
    else:
        raise RuntimeError(f"必要なパッケージ '{r}' が見つかりませんでした。")