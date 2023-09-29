#!/usr/bin/env bash

# Function to handle errors and display error messages
handle_error() {
    echo "Error: $1" >&2
    exit 1
}

# Accepts output directory name as input
output_dir=$1

# Check if output directory is provided
if [ -z "$output_dir" ]; then
    handle_error "Output directory name is missing."
fi

# Check if output directory already exists
if [ -d "$output_dir" ]; then
    handle_error "Output directory already exists."
fi

# Print progress message
echo "Creating output directory..."

# Create the output directory
mkdir -p "$output_dir" || handle_error "Failed to create output directory."

# Function to download files
download_file() {
    url=$1
    filename=$2

    # Print progress message
    echo "Downloading $filename..."

    wget -q "$url" -O "$output_dir/$filename" || handle_error "Failed to download $filename."

    # Print download success message
    echo "Downloaded $filename."
}

process_eips_repo() {
    local repo_url="https://github.com/ethereum/EIPs"
    local repo_dir="$output_dir/EIPs"

    echo "Cloning EIPs repository..."
    git clone -q "$repo_url" "$repo_dir" || handle_error "Failed to clone EIPs repository."

    cd "$repo_dir" || handle_error "Failed to enter EIPs repository directory."

    echo "Deleting unnecessary files..."
    find . -type f ! -name "eip-[0-9]*.md" -exec rm -rf {} +
    find . -type d ! -name "EIPS" ! -name "." ! -name ".." -exec rm -rf {} +
}

# Download book.pdf from https://eth2book.info/latest/book.pdf
download_file "https://eth2book.info/latest/book.pdf" "book.pdf"

# Download paper.pdf from https://ethereum.github.io/yellowpaper/paper.pdf
download_file "https://ethereum.github.io/yellowpaper/paper.pdf" "paper.pdf"

# Process EIPs
process_eips_repo

# Print success message
echo "Script executed successfully."