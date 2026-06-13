#!/bin/bash
# script by Matthew
# Loic added ability to have an input filename when calling it


#SBATCH -J andrea_doublepsf_first_batch # Job name
#SBATCH -o /work/10875/andreavu/ls6/injection_and_recovery_F1000W/job_running_output/o%j # Stdout output file
#SBATCH -e /work/10875/andreavu/ls6/injection_and_recovery_F1000W/job_running_output/e%j # Stdout output file/e%j # Stderr error file
#SBATCH -p normal # Queue (partition) name
#SBATCH -N 1 # Total # of nodes
#SBATCH -n 128 # Total # of MPI tasks
#SBATCH -t 20:00:00 # Run time (hh:mm:ss)
#SBATCH --mail-type=all # Send email at begin and end of job
#SBATCH --mail-user=a24vu@uwaterloo.ca
#SBATCH -A AST24030

module load python3/3.9.7

# Timeout in minutes that no output has been added and afterwhich a job is deemed hung
TIMEOUT=5

# Function to monitor the output file
watchdog() {
    local file="$1"
    local pid="$2"

    while ps -p "$pid" > /dev/null
    do
        if [[ -f "$file" ]]; then
            # Check if file hasn't been modified in $TIMEOUT minutes
            if [[ $(find "$file" -type f -mmin +$(echo "$TIMEOUT" | bc -l) -print) ]]; then
                echo "No output detected for $TIMEOUT minutes in $file. Terminating iteration with process number $pid."
                kill "$pid"
                wait "$pid" 2>/dev/null # Ensure the process is fully cleaned up
                return
            fi
        fi
        sleep 10 # replace with 10 when testing is done
    done
}

# Part added by Loic here - Check if the user provided an argument
if [ -z "$2" ]; then
  echo "Usage: $0 <starting_index> <ending_index>"
  exit 1
fi
# Assign the first parameter to a variable
starting_index="$1"
ending_index="$2"
echo "starting source index is $starting_index"
echo "ending source index is $ending_index"
echo "output file whose time tag is monitored for sign of hanging: $OUTPUT_FILE"


i=$starting_index
while [ $i -le $ending_index ]
do
    echo "Doing source "$i

    # Cleanup any leftover Python processes before starting a new iteration
    echo "First cleaning up - Killing remaining python jobs and pausing 3 seconds..."
    pkill -9 python  # Forcefully clean up lingering Python processes
    sleep 3           # Short pause to ensure cleanup completes

    # Run Python script in the background
    echo "Launching the python script..."
    echo "ibrun python doublepsf_fitting_injected_binary.py $i &"
    ibrun python doublepsf_fitting_injected_binary.py "$i" &
    PID=$!
    # the output_file is named o + PID so forge it here
    OUTPUT_FILE="/work/10875/andreavu/ls6/injection_and_recovery_F1000W/job_running_output/o${SLURM_JOB_ID}"
    echo "output file whose time tag is monitored for sign of hanging: $OUTPUT_FILE"

    # Start watchdog to monitor the output file
    echo "Launching the watchdog monitoring file "$OUTPUT_FILE" with pid "$PID
    watchdog "$OUTPUT_FILE" "$PID" &

    # Wait for the process to complete
    wait $PID || true

    ((i++))
done
