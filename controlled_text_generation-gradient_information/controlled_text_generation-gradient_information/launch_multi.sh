#!/bin/bash

# --- Configuration ---
THRESHOLD=500                      # VRAM threshold in MB (below this = free)
TOPIC="gpu_alert_sarthak" # Your ntfy.sh unique topic

echo "Waiting for any GPUs to become free..."

while true; do
    # Initialize an empty array to hold the IDs of currently free GPUs
    FREE_GPUS=()

    # Read the index and memory of all GPUs on the server
    while IFS=', ' read -r GPU_INDEX MEM_USED; do
        if [ "$MEM_USED" -lt "$THRESHOLD" ]; then
            # If free, add this GPU's index to our array
            FREE_GPUS+=("$GPU_INDEX")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

    # Check if we found AT LEAST ONE free GPU
    if [ "${#FREE_GPUS[@]}" -ge 1 ]; then
        
        # Join all elements in the array with a space (e.g., "0", or "0 1 2 3")
        GPU_ARGS="${FREE_GPUS[*]}"
        
        echo "Success! Grabbing free GPUs: $GPU_ARGS"
        
        # 1. Send the push notification to your phone
        curl -d "Started job on GPUs: $GPU_ARGS" ntfy.sh/$TOPIC
        
        # 2. Launch your specific command, injecting all free GPUs into the --gpus flag
        ./run_queue.sh \
            --manifest manifest_llama.tsv \
            --gpus "$GPU_ARGS" \
            --vram 48 \
            --out_dir results_llama \
            --status status_llama \
            --env gfn
            
        # 3. Exit the monitoring script so it doesn't loop again after your job finishes
        exit 0
    fi

    # If we get here, absolutely zero GPUs were free. Wait 60 seconds and check again.
    sleep 60
done
