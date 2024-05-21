
# use NVIDIA's CUDA runtime image with Python 3.11 
FROM nvidia/cuda:12.5.0-runtime-ubuntu22.04

# Set the working directory
WORKDIR /train-llms-from-scratch

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        curl \
        python3.11 \
        python3.11-distutils \
        python3.11-dev && \
    rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.11

# Set Python 3.11 as the default python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# Copy the requirements files
COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt

# install core dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip install --no-cache-dir -r requirements-dev.txt

# copy the entire project into the container
COPY . .

# Expose ports for Jupyter Notebooks if needed
EXPOSE 8888 5000

CMD ["bash"]
# RUN tests
#CMD ["pytest", "tests", "--maxfail=5", "--disable-warnings"]
