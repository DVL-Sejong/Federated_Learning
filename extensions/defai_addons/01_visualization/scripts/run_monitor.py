"""연합 학습 환경 평가 시각화 도구 실행.

  python scripts/run_monitor.py --port 5000
  → 브라우저에서 http://127.0.0.1:5000 접속
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "shared")))

from defai_addons.monitor.server import main

if __name__ == "__main__":
    main()
