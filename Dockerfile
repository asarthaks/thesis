FROM python:3.11-slim

WORKDIR /code

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git wget libgl1 libglib2.0-0 libgomp1 \
                       libsm6 libxext6 libgles2 libegl1 build-essential

COPY requirements.txt ./

# Install torch FIRST with exact version from cu118
# This must happen before requirements.txt so pip sees
# torch==2.2.0 as already satisfied and does not upgrade it
RUN pip install --upgrade pip && \
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
        --index-url https://download.pytorch.org/whl/cu118

# Now install everything else — pip will not touch torch
RUN pip install --no-cache-dir -r requirements.txt

COPY ./ /code/

CMD ["jupyter-notebook", "--ip=0.0.0.0", "--port=6969", \
     "--no-browser", "--allow-root", \
     "--NotebookApp.token=''", "--NotebookApp.password=''"]