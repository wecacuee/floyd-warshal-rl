#PBS -N fwrl-qlnet-pong          # Any name to identify your job
#PBS -j oe                   # Join error and output files for convinience
#PBS -l walltime=24:00:00     # Keep walltime big enough to finish the job
#PBS -l nodes=1:ppn=1:gpus=1 # nodes requested: Processor per node: gpus requested
#PBS -S /bin/bash            # Shell to use
#PBS -m a                  # Mail to <user>@umich.edu on abort, begin and end
#PBS -M dhiman@umich.edu     # Email id to alert
#PBS -o /z/home/dhiman/mid/floyd_warshall_rl/pbs/$PBS_JOBID.out
#
# #PBS -q fluxg              # Not required for blindspot but for flux
# #PBS -A jjcorso_fluxg      # Not required for blindspot but for flux

echo "starting pbs script"
VERSION=0.1.0
MID_DIR=/z/home/dhiman/mid/floyd_warshall_rl/
for d in src-$VERSION pbs build; do mkdir -p $MID_DIR/$d; done
cd $MID_DIR/src-$VERSION/
git clone $MID_DIR/git/ floyd_warshall_rl
cd floyd_warshall_rl
git pull
. setup.sh
python <<EOF
from fwrl.conf.qlnet_cartpole import play_qlnet_pong
play_qlnet_pong()
EOF
echo "end of pbs script"
