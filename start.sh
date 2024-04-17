#!/bin/bash

set -e # Exit the script if any statement returns a non-true return value

# ---------------------------------------------------------------------------- #
# Function Definitions #
# ---------------------------------------------------------------------------- #

# Start nginx service
start_nginx() {
  echo "Starting Nginx service..."
  service nginx start
}

# Execute script if exists
execute_script() {
  local script_path=$1
  local script_msg=$2
  if [[ -f ${script_path} ]]; then
    echo "${script_msg}"
    bash ${script_path}
  fi
}

# Setup ssh
setup_ssh() {
  if [[ $PUBLIC_KEY ]]; then
    echo "Setting up SSH..."
    mkdir -p ~/.ssh
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 700 -R ~/.ssh
    if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then
      ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key -q -N ''
      echo "RSA key fingerprint:"
      ssh-keygen -lf /etc/ssh/ssh_host_rsa_key.pub
    fi
    if [ ! -f /etc/ssh/ssh_host_dsa_key ]; then
      ssh-keygen -t dsa -f /etc/ssh/ssh_host_dsa_key -q -N ''
      echo "DSA key fingerprint:"
      ssh-keygen -lf /etc/ssh/ssh_host_dsa_key.pub
    fi
    if [ ! -f /etc/ssh/ssh_host_ecdsa_key ]; then
      ssh-keygen -t ecdsa -f /etc/ssh/ssh_host_ecdsa_key -q -N ''
      echo "ECDSA key fingerprint:"
      ssh-keygen -lf /etc/ssh/ssh_host_ecdsa_key.pub
    fi
    if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then
      ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -q -N ''
      echo "ED25519 key fingerprint:"
      ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
    fi
    service ssh start
    echo "SSH host keys:"
    for key in /etc/ssh/*.pub; do
      echo "Key: $key"
      ssh-keygen -lf $key
    done
  fi
}

# Export env vars
export_env_vars() {
  echo "Exporting environment variables..."
  printenv | grep -E '^RUNPOD_|^PATH=|^_=' | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /etc/rp_environment
  echo 'source /etc/rp_environment' >> ~/.bashrc
}

# Start your Python app
start_app() {
  echo "Starting Python app..."
  cd /app
  python app.py
}

# ---------------------------------------------------------------------------- #
# Main Program #
# ---------------------------------------------------------------------------- #

start_nginx

execute_script "/pre_start.sh" "Running pre-start script..."

echo "Pod Started"

setup_ssh

export_env_vars

execute_script "/post_start.sh" "Running post-start script..."

start_app

echo "Start script(s) finished, pod is ready to use."

sleep infinity