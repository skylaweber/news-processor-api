#!/bin/bash
# Install Rust in the build environment
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
pip3 install -r requirements.txt