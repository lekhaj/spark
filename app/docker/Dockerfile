FROM nvidia/cuda:12.1-devel-ubuntu22.04

WORKDIR /app

# Install Python and essentials
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    build-essential \       
    libgl1 \               
    libglib2.0-0  \         
    && ln -s /usr/bin/python3 /usr/bin/python

# Install PyTorch
#RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Hunyuan3D
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git Hunyuan3d && \
    cd Hunyuan3D && \
    pip install -r requirements.txt && \
    pip install -e . && \
    cd hy3dgen/texgen/custom_rasterizer && python setup.py install && \
    cd ../differentiable_renderer && python setup.py install

# Install your dependencies
RUN pip install redis boto3 requests rembg pillow trimesh

# Copy your script
COPY worker.py .

ENV PYTHONPATH=/app/Hunyuan3D:$PYTHONPATH

CMD ["python", "worker.py"]