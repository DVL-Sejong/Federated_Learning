"""한 컴퓨터에서 Flower gRPC 서버 1개 + 클라이언트 N개를 별도 프로세스로 띄워 실제 통신으로 연합 학습을 실행.

  python scripts/flower_launch.py --n-clients 5 --rounds 10 --algorithm fedavg --dashboard http://127.0.0.1:5000
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clients", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--algorithm", default="fedavg")
    ap.add_argument("--address", default="127.0.0.1:8080")
    ap.add_argument("--dashboard", default=None)
    a = ap.parse_args()
    py = sys.executable
    srv = [py, os.path.join(HERE, "flower_server.py"), "--address", a.address, "--rounds", str(a.rounds),
           "--n-clients", str(a.n_clients), "--algorithm", a.algorithm]
    if a.dashboard:
        srv += ["--dashboard", a.dashboard]
    server = subprocess.Popen(srv)
    time.sleep(5)
    clients = [subprocess.Popen([py, os.path.join(HERE, "flower_client.py"), "--cid", str(k),
                                 "--n-clients", str(a.n_clients), "--server", a.address])
               for k in range(a.n_clients)]
    rc = server.wait()
    for c in clients:
        try:
            c.wait(timeout=30)
        except subprocess.TimeoutExpired:
            c.kill()
    sys.exit(rc)


if __name__ == "__main__":
    main()
