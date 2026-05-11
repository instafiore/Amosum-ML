#!/usr/bin/env bash

uncompress () {
    filename=$1
    output=$2
    [[ ! -e $1 ]] && echo "Input file $1 missing." && exit 2
    type=$(file -b --mime-type $filename)
    echo "Compressed file recognized as: " $type

    if [ $type == "application/x-lzma" ] ; then
         prep_cmd="lzcat $filename"
    elif [ $type == "application/x-bzip2" ] ; then
         prep_cmd="bzcat $filename"
    elif [ $type == "application/x-xz" ] ; then
         prep_cmd="xzcat $filename"
    elif [ $type == "application/octet-stream" ] ; then
         prep_cmd="lzcat $filename"
    else
         prep_cmd="zcat -f $filename"
    fi
    echo "Preparing instance in $output"
    echo "$prep_cmd > $output"
    $prep_cmd > $output
}

_cleanup() {
    # cleanup symlinks
    find . -type l -delete
    # copy output into run dir
    cp * ~/AMOSUM/benchmarks/AMOSUM_GraphColouring/config2/instance142/run1
    # cleanup shm files
    rm -rf /dev/shm/376141a6-2e81-11f1-ac62-000c298f4621/
}

_term() {
  kill -TERM "$child" 2>/dev/null
  _cleanup
}

trap _term SIGTERM
trap _cleanup EXIT

if [ -f ~/AMOSUM/benchmarks/AMOSUM_GraphColouring/config2/instance142/run1/00_finished.log ] ; then
    echo ">>>Solver already finished before. File '00_finished.log' exists."
    echo ">>>...Run the following command manually, if you want to proceed find $(realpath .) -name 00_finished.log -exec rm {} \;"
    echo ">>>...stopping here..."
    exit 4
fi

# change into job directory
mkdir /dev/shm/376141a6-2e81-11f1-ac62-000c298f4621
cd /dev/shm/376141a6-2e81-11f1-ac62-000c298f4621
mkdir input
mkdir output
cd output
# create log files (so that symlinks cannot interfere)
touch runsolver.log stdout.log stderr.log varfile.log perf.log node_info.log
# create symlinks for working directory
ln -s ~//home/guests/mmacri/AMOSUM/benchmarks/* .
# move inputs into shared mem
cp /home/guests/mmacri/AMOSUM/benchmarks/GraphColouring/test-instances-gc/0029-graph_colouring-135-0_2592.asp /dev/shm/376141a6-2e81-11f1-ac62-000c298f4621/input/0029-graph_colouring-135-0_2592.asp
cp /opt/runsolver /dev/shm/376141a6-2e81-11f1-ac62-000c298f4621/input/runsolver
# uncompress input files
# store node info
echo Date: $(date) > node_info.log
echo Node: $(hostname) >> node_info.log
echo Input: "/home/guests/mmacri/AMOSUM/benchmarks/GraphColouring/test-instances-gc/0029-graph_colouring-135-0_2592.asp" >> node_info.log
echo GCC: $(gcc --version | head -n1) >> node_info.log
echo Kernel: $(uname -r) >> node_info.log
echo $(cat /proc/meminfo  | grep MemTotal) >> node_info.log
echo $(cat /proc/cpuinfo  | egrep "^model name|^cache size" | head -2) >> node_info.log
cat /proc/self/status | grep Cpus_allowed: >> node_info.log
echo resctrl cache mask: $(cat /sys/fs/resctrl/$SLURM_JOB_ID/schemata | tail -1) >> node_info.log
echo Slurm Job ID: $SLURM_ARRAY_JOB_ID"_"$SLURM_ARRAY_TASK_ID >> node_info.log
myenv=""
# clear cache if tool is configured and present
if [ -f /opt/clearcache ] ; then
    /opt/clearcache
fi

# execute run
env $myenv /dev/shm/376141a6-2e81-11f1-ac62-000c298f4621/input/runsolver -w /dev/shm/376141a6-2e81-11f1-ac62-000c298f4621/output/runsolver.log -v /dev/shm/376141a6-2e81-11f1-ac62-000c298f4621/output/varfile.log -W 1205 --rss-swap-limit 16384 -d 5 --sigint /usr/bin/perf stat -B -e task-clock,cache-references,cache-misses,cycles,instructions,branches,faults,migrations,context-switches -o /dev/shm/376141a6-2e81-11f1-ac62-000c298f4621/output/perf.log clingo /home/guests/mmacri/AMOSUM/benchmarks/GraphColouring/encoding-plain-amo.asp $instance /dev/shm/376141a6-2e81-11f1-ac62-000c298f4621/input/0029-graph_colouring-135-0_2592.asp 2> /dev/shm/376141a6-2e81-11f1-ac62-000c298f4621/output/stderr.log 1> /dev/shm/376141a6-2e81-11f1-ac62-000c298f4621/output/stdout.log &
child=$!
wait "$child"
touch 00_finished.log