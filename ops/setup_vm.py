# Unpack the uploaded repo on the Colab VM and report the environment.
import subprocess

print(subprocess.run(
    "mkdir -p /content/dlp && tar -xzf /content/dlp.tgz -C /content/dlp && ls /content/dlp && "
    "pip install -q gdown nltk onnx onnxruntime onnxscript faiss-cpu sentence-transformers rank-bm25 2>&1 | tail -1; "
    "python -c 'import keras, tensorflow as tf, torch, transformers; "
    "print(keras.__version__, keras.backend.backend(), tf.__version__, torch.__version__, transformers.__version__); "
    "print(tf.config.list_physical_devices(\"GPU\"))'; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
    shell=True, capture_output=True, text=True).stdout)
