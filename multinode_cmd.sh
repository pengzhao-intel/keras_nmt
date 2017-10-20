salloc --nodes=2 --partition=bdw-inf
unset I_MPI_PMI_LIBRARY
MLSL_NUM_SERVERS=0 MLSL_HOSTNAME_TYPE=1 I_MPI_FABRICS=tmi mpirun -n 2 -ppn 1 -hosts pcs-bdw05,pcs-bdw06 python train.py --state config.py
