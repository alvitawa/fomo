# Pseudocode for handling Ctrl+C during ray job submit and asking user to cancel the job

# Generate a unique submission id using current timestamp and random number
submission_id=$(date +%s%N)$RANDOM

# Echo the generated submission id
echo "Generated submission id: $submission_id"

# Setup trap to catch Ctrl+C (SIGINT) and define cleanup function
trap 'cleanup' SIGINT

cleanup() {
#    # Ask user if they want to cancel the job
#    read -p "Do you want to cancel the job? (y/n) " -n 1 -r
#    echo
#    if [[ $REPLY =~ ^[Yy]$ ]]; then
#        # Cancel the job using ray job stop
#        ray job stop "$submission_id"
#        echo "Job with submission id $submission_id has been cancelled."
#    fi
    ray job stop "$submission_id"
    echo "Job with submission id $submission_id has been cancelled."
    exit 130 # Exit with code 130 to indicate script was terminated by Ctrl+C
}

# Execute the ssh command to setup port forwarding
ssh -o ExitOnForwardFailure=yes -f -L 8265:localhost:8265 black sleep 3600

# Submit the job to Ray with the generated submission id
# https://docs.ray.io/en/latest/cluster/running-applications/job-submission/cli.html
ray job submit --working-dir . --entrypoint-memory 1048576000 --runtime-env runtime_env.yaml --entrypoint-num-gpus 1 --submission-id "$submission_id" -- "$@"

# Remove the trap for SIGINT after the command completes successfully
trap - SIGINT

# Echo the submission id again after the job submit command ends
echo "----------------------------------------"
echo "Submission id for the job: $submission_id"
echo "----------------------------------------"
