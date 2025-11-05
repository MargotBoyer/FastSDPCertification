#!/bin/bash
NETWORK_NAME=$1
if [ -z "$NETWORK_NAME" ]; then
  echo "Erreur: NETWORK_NAME non fourni."
  exit 1
fi

NOM_RUN=$2

if [ -z "$NOM_RUN" ]; then
  echo "Erreur: NOM_RUN non fourni."
  exit 1
fi

EPSILON=$3
if [ -z "$EPSILON" ]; then
  if [ "$NETWORK_NAME" == "6x100" ]; then
    EPSILON=0.026
  elif [ "$NETWORK_NAME" == "6x200" ]; then
    EPSILON=0.015
  elif [ "$NETWORK_NAME" == "9x100" ]; then
    EPSILON=0.026
  elif [ "$NETWORK_NAME" == "9x200" ]; then
    EPSILON=0.015
  elif [ "$NETWORK_NAME" == "MLP-ADV" ]; then
    EPSILON=0.1
  else 
    echo "Erreur: EPSILON non fourni."
    exit 1
  fi
else 
  echo "Utilisation de EPSILON=$EPSILON pour le réseau $NETWORK_NAME."
fi


if [[ "$NETWORK_NAME" == "6x100" || "$NETWORK_NAME" == "6x200" || "$NETWORK_NAME" == "9x100" || "$NETWORK_NAME" == "9x200" || "$NETWORK_NAME" == "MLP-ADV" ]]
then
    NETWORK_NAME="mnist-$NETWORK_NAME"
fi

python src/certification_problem.py $NETWORK_NAME $NOM_RUN 






################""
sbatch --export=NETWORK_NAME=$NETWORK_NAME,NAME_RUN=$NAME_RUN scripts/run_fastsdp_job.slurm

