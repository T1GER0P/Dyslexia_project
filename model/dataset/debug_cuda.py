import os, torch, sys
print("Python:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
try:
    import subprocess
    print("nvidia-smi output:")
    subprocess.run(["nvidia-smi"], check=True)
except Exception as e:
    print("nvidia-smi error:", e)
