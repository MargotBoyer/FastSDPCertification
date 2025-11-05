#!/bin/bash
# Script pour lancer un run sur jean-zay
# Configuration - À modifier selon tes paramètres
JEAN_ZAY_USER="uvq13au"
JEAN_ZAY_HOST="jean-zay.idris.fr"
LOCAL_PROJECT_DIR="/share/homes/boyerma/FastSDPCertification"
REMOTE_WORK_DIR="/lustre/fswork/projects/rech/llc/uvq13au"
REMOTE_PROJECT_DIR="$REMOTE_WORK_DIR/FastSDPCertification"  # Nom de ton projet/dossier sur Jean-Zay

# Dossiers à synchroniser
FOLDERS_TO_SYNC=("src" "data" "notebooks" "config")

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages colorés
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Vérifications préliminaires
if [ ! -d "$LOCAL_PROJECT_DIR" ]; then
    log_error "Le dossier local $LOCAL_PROJECT_DIR n'existe pas !"
    exit 1
fi

# Vérification que les dossiers à synchroniser existent
for folder in "${FOLDERS_TO_SYNC[@]}"; do
    if [ ! -d "$LOCAL_PROJECT_DIR/$folder" ]; then
        log_warning "Le dossier $LOCAL_PROJECT_DIR/$folder n'existe pas, il sera ignoré"
    fi
done

# Test de connexion SSH
log_info "Test de connexion à Jean-Zay..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$JEAN_ZAY_USER@$JEAN_ZAY_HOST" exit 2>/dev/null; then
    log_error "Impossible de se connecter à Jean-Zay. Vérifie tes clés SSH et ta connexion."
    exit 1
fi
log_success "Connexion SSH OK"

NETWORK_NAME=$1
NAME_RUN=$2
EPSILON=$3

if [ -z "$NETWORK_NAME" ] || [ -z "$NAME_RUN" ]; then
  echo "Erreur: paramètres manquants."
  echo "Usage: bash run_fastsdp.sh <NETWORK_NAME> <NAME_RUN> [EPSILON]"
  exit 1
fi

# Détection automatique de EPSILON si non fourni
if [ -z "$EPSILON" ]; then
  case $NETWORK_NAME in
    6x100|9x100) EPSILON=0.026 ;;
    6x200|9x200) EPSILON=0.015 ;;
    MLP-ADV)     EPSILON=0.1   ;;
    *) echo "Erreur: EPSILON non fourni pour réseau inconnu."; exit 1 ;;
  esac
fi

# Préfixe MNIST si besoin
if [[ "$NETWORK_NAME" =~ ^(6x100|6x200|9x100|9x200|MLP-ADV)$ ]]; then
  NETWORK_NAME="mnist-$NETWORK_NAME"
fi

echo "Projet bien reçu avec les paramètres :"

# Envoie le script SLURM et le projet à Jean Zay (si ce n'est pas déjà fait)
log_info "Synchronisation du projet vers Jean-Zay..."
for folder in "${FOLDERS_TO_SYNC[@]}"; do
    if [ -d "$LOCAL_PROJECT_DIR/$folder" ]; then
        rsync -az "$LOCAL_PROJECT_DIR/$folder" "$JEAN_ZAY_USER@$JEAN_ZAY_HOST:$REMOTE_PROJECT_DIR"
    fi
done

echo "Entré dans le dir"

DATE=$(date +%Y_%m_%d_%Hh%M_%Ss)
LOG_DIR="results/benchmark/${NETWORK_NAME}-${EPSILON}/${DATE}_${NAME_RUN}"
mkdir -p "$LOG_DIR"

# Redirige stdout et stderr
exec > >(tee -a "$LOG_DIR/slurm-${SLURM_JOB_ID}.out")
exec 2> >(tee -a "$LOG_DIR/slurm-${SLURM_JOB_ID}.err" >&2)

echo "Network: $NETWORK_NAME"
echo "Run name: $NAME_RUN"
echo "Job ID: $SLURM_JOB_ID"# Crée le répertoire de logs si besoin
DATE=$(date +%Y_%m_%d_%Hh%M_%Ss)
LOG_DIR="results/benchmark/${NETWORK_NAME}-${EPSILON}/${DATE}_${NAME_RUN}"
mkdir -p "$LOG_DIR"

# Redirige stdout et stderr
# exec > >(tee -a "$LOG_DIR/slurm-${SLURM_JOB_ID}.out")
# exec 2> >(tee -a "$LOG_DIR/slurm-${SLURM_JOB_ID}.err" >&2)

# echo "Network: $NETWORK_NAME"
# echo "Run name: $NAME_RUN"
# echo "Job ID: $SLURM_JOB_ID"
# done
rsync -az "$LOCAL_PROJECT_DIR/scripts" "$JEAN_ZAY_USER@$JEAN_ZAY_HOST:$REMOTE_PROJECT_DIR"

log_success "Synchronisation terminée."

# Création d'une commande distante pour lancer le job sur Jean-Zay via sbatch
REMOTE_COMMAND="cd $REMOTE_PROJECT_DIR && sbatch --export=NETWORK_NAME=$NETWORK_NAME,NAME_RUN=$NAME_RUN scripts/run_fastsdp_job.slurm"

log_info "Soumission du job SLURM à Jean-Zay..."
ssh "$JEAN_ZAY_USER@$JEAN_ZAY_HOST" "$REMOTE_COMMAND"
log_success "Job soumis à Jean-Zay."