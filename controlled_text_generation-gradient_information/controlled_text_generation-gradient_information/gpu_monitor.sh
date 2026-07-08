#!/bin/bash

# --- Configuration ---
# Set this to the unique topic name you created in the app
TOPIC="gpu_alert_sarthak"

# Memory threshold in MB (if usage is below this, we consider it "free")
THRESHOLD=500 

# Which GPU to check? (0 is the first GPU. Remove the -i 0 flag to check all)
GPU_ID=0 

echo "Monitoring GPU $GPU_ID. Will notify ntfy.sh/$TOPIC when free..."

while true; do
    # Get the current memory usage of the specified GPU
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    
    # Check if memory is below the threshold
    if [ "$MEM_USED" -lt "$THRESHOLD" ]; then
        # Send the push notification
        curl -d "GPU $GPU_ID is finally free! (Memory used: ${MEM_USED}MB)" ntfy.sh/$TOPIC
        
        echo "Alert sent! Exiting monitor."
        break # Exit the loop so it doesn't spam your phone every 60 seconds
    fi
    
    # Wait 60 seconds before checking again
    sleep 60
done
