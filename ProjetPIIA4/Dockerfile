FROM python:3.10-slim

# On ajoute cmake et g++ (obligatoires pour dlib)
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 git wget build-essential python3-dev \
    python3-tk cmake g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# On installe les bibliothèques lourdes
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir tensorflow-cpu opencv-python onnxruntime onnx

# On installe dlib séparément (ça va prendre du temps)
RUN pip install --no-cache-dir dlib

# Le reste des dépendances
RUN pip install --no-cache-dir tqdm insightface==0.7.3 psutil pydantic opencv-contrib-python
RUN pip install --no-cache-dir basicsr==1.4.2 gfpgan batch-face configparser customtkinter tkinterdnd2

COPY app/ /app/
CMD ["python", "pipeline.py"]