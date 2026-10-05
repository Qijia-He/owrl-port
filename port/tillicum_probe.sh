#!/usr/bin/env bash
# One-shot environment probe for porting OpenWebRL to Tillicum.
#   on the login node:  bash tillicum_probe.sh > probe_login.txt 2>&1
#   on a GPU node:      srun -A <acct> -p <gpu-partition> --gpus=1 --time=10 bash tillicum_probe.sh > probe_gpu.txt 2>&1
# Prints only system facts; no secrets.
sec(){ echo; echo "===== $* ====="; }
sec host;        hostname; date; cat /etc/os-release | head -3; ldd --version | head -1; nproc; free -g | head -2
sec slurm;       sacctmgr show assoc user=$USER format=cluster,account,partition,qos -P 2>/dev/null | sort -u
                 sinfo -o "%P %a %l %D %c %m %G" 2>/dev/null | head -30
                 scontrol show partition 2>/dev/null | grep -E "PartitionName|MaxTime|TRES|DefMem|MaxNodes" | head -40
                 sshare -U -o Account,User,RawShares,FairShare 2>/dev/null | head
sec gpus;        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null; nvidia-smi -L 2>/dev/null | wc -l
sec storage;     echo HOME=$HOME; df -h $HOME 2>/dev/null; for d in /gscratch /scratch /gpfs /mmfs1 /data /projects; do [ -d $d ] && (echo "-- $d"; ls $d 2>/dev/null | head -20; df -h $d 2>/dev/null | tail -1); done
                 (command -v hyakstorage && hyakstorage 2>/dev/null) || (command -v quota && quota -s 2>/dev/null) || true
                 echo "-- node-local"; df -h /tmp /local* /scratch* 2>/dev/null; echo TMPDIR=$TMPDIR
sec modules;     module avail 2>&1 | grep -iE "gcc|cuda|apptainer|singularity|conda|miniconda|python" | head -30; which gcc; gcc --version | head -1; which nvcc; nvcc --version 2>/dev/null | tail -1
sec network;     env | grep -i proxy; for u in https://www.google.com https://www.amazon.com https://api.openai.com/v1/models https://api.wandb.ai https://pypi.org/simple/pip/; do printf "%-45s " $u; curl -s -o /dev/null -m 15 -w "%{http_code} %{time_total}s\n" $u || echo FAIL; done
                 echo "-- with proxy vars cleared:"; for u in https://www.google.com https://api.openai.com/v1/models; do printf "%-45s " $u; env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY curl -s -o /dev/null -m 15 -w "%{http_code} %{time_total}s\n" $u || echo FAIL; done
sec chromium-libs; for l in libnss3.so libnspr4.so libnssutil3.so libatk-1.0.so.0 libatk-bridge-2.0.so.0 libcups.so.2 libxkbcommon.so.0 libXcomposite.so.1 libXdamage.so.1 libXfixes.so.3 libXrandr.so.2 libgbm.so.1 libcairo.so.2 libpango-1.0.so.0 libasound.so.2 libdrm.so.2; do printf "%-26s %s\n" $l "$(ldconfig -p 2>/dev/null | grep -m1 -c $l | sed 's/1/ok/;s/0/MISSING/')"; done
sec limits;      ulimit -a | grep -E "open files|processes|memory"; cat /proc/sys/kernel/pid_max; sysctl -n fs.file-max 2>/dev/null
sec containers;  command -v apptainer singularity docker podman 2>/dev/null
