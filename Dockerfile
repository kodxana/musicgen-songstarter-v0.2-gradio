# Use the RunPod PyTorch image as the base
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Set the HF_HUB_ENABLE_HF_TRANSFER environment variable
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Set the working directory in the container
WORKDIR /app

# Update the package list and install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install audiocraft from the GitHub repository
RUN pip install -U git+https://github.com/facebookresearch/audiocraft#egg=audiocraft

# Copy the application code to the working directory
COPY . .

# Copy the start.sh script to the container
COPY start.sh /start.sh

# Make the start.sh script executable
RUN chmod +x /start.sh

# Run the cache_model.py script to cache the model
RUN python cache_model.py

# Run the start.sh script when the container starts
CMD ["/start.sh"]
