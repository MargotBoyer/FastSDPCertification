#!/bin/bash
# Script pour récupérer les résultats depuis le serveur Jean-Zay via rsync
# Configuration - À modifier selon tes paramètres
JEAN_ZAY_USER="uvq13au"
JEAN_ZAY_HOST="jean-zay.idris.fr"
LOCAL_PROJECT_DIR="/share/homes/boyerma/FastSDPCertification"
REMOTE_PROJECT_DIR="FastSDPCertification"
LOCAL_RESULTS_DIR="$LOCAL_PROJECT_DIR/results/benchmark/"
REMOTE_RESULTS_DIR="results/benchmark/"

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

# Test de connexion SSH
log_info "Test de connexion à Jean-Zay..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$JEAN_ZAY_USER@$JEAN_ZAY_HOST" exit 2>/dev/null; then
    log_error "Impossible de se connecter à Jean-Zay. Vérifie tes clés SSH et ta connexion."
    exit 1
fi
log_success "Connexion SSH OK"

# Récupération du chemin $WORK depuis Jean-Zay
log_info "Récupération du chemin \$WORK depuis Jean-Zay..."
REMOTE_WORK_PATH=/lustre/fswork/projects/rech/llc/uvq13au          #$(ssh "$JEAN_ZAY_USER@$JEAN_ZAY_HOST" 'echo $WORK')

# if [ -z "$REMOTE_WORK_PATH" ]; then
#     log_error "Impossible de récupérer le chemin \$WORK depuis Jean-Zay"
#     exit 1
# fi

# log_info "Chemin \$WORK détecté: $REMOTE_WORK_PATH"

# Vérification que le dossier results existe sur Jean-Zay
log_info "Vérification de l'existence du dossier results sur Jean-Zay..."
if ! ssh "$JEAN_ZAY_USER@$JEAN_ZAY_HOST" "test -d '$REMOTE_WORK_PATH/$REMOTE_PROJECT_DIR/$REMOTE_RESULTS_DIR'"; then
    log_error "Le dossier results n'existe pas sur Jean-Zay dans $REMOTE_WORK_PATH/$REMOTE_PROJECT_DIR/$REMOTE_RESULTS_DIR"
    exit 1
fi

# Création du dossier results local s'il n'existe pas
if [ ! -d "$LOCAL_RESULTS_DIR" ]; then
    log_info "Création du dossier results local..."
    mkdir -p "$LOCAL_RESULTS_DIR"
fi

# Récupération de la liste des sous-dossiers distants
log_info "Récupération de la liste des sous-dossiers sur Jean-Zay..."
REMOTE_SUBDIRS=$(ssh "$JEAN_ZAY_USER@$JEAN_ZAY_HOST" "find '$REMOTE_WORK_PATH/$REMOTE_PROJECT_DIR/$REMOTE_RESULTS_DIR' -maxdepth 1 -type d -not -path '*/$REMOTE_RESULTS_DIR' -printf '%f\n' 2>/dev/null")

if [ -z "$REMOTE_SUBDIRS" ]; then
    log_warning "Aucun sous-dossier trouvé dans le dossier results sur Jean-Zay"
    exit 0
fi

log_info "Sous-dossiers trouvés sur Jean-Zay:"
echo "$REMOTE_SUBDIRS" | while IFS= read -r subdir; do
    echo "  - $subdir"
done

# Options rsync pour la récupération
RSYNC_OPTIONS=(
    -avz                    # archive, verbose, compress
    --progress             # affiche le progrès
    --human-readable       # tailles lisibles
    --exclude='*.tmp'      # exclut les fichiers temporaires
    --exclude='*.lock'     # exclut les fichiers de verrou
)

# Synchronisation des nouveaux sous-dossiers
log_info "Synchronisation des résultats depuis Jean-Zay..."
sync_count=0
skip_count=0

echo "$REMOTE_SUBDIRS" | while IFS= read -r subdir; do
    if [ -n "$subdir" ]; then
        if [ -d "$LOCAL_RESULTS_DIR/$subdir" ]; then
            log_info "Mise à jour du dossier existant '$subdir'..."
            ((skip_count++))
        else
            log_info "Nouveau dossier détecté: '$subdir' - téléchargement..."
            ((sync_count++))
        fi
        
        # Synchronisation (mise à jour ou nouveau téléchargement)
        if rsync "${RSYNC_OPTIONS[@]}" \
            "$JEAN_ZAY_USER@$JEAN_ZAY_HOST:$REMOTE_WORK_PATH/$REMOTE_PROJECT_DIR/$REMOTE_RESULTS_DIR/$subdir/" \
            "$LOCAL_RESULTS_DIR/$subdir/"; then
            log_success "Dossier '$subdir' synchronisé avec succès"
        else
            log_error "Erreur lors de la synchronisation du dossier '$subdir'"
        fi
    fi
done

# Résumé final
log_info "Récupération terminée !"
log_info "Contenu du dossier results local:"
if [ -d "$LOCAL_RESULTS_DIR" ]; then
    ls -la "$LOCAL_RESULTS_DIR/"
else
    log_warning "Le dossier results local n'existe toujours pas"
fi

# Affichage de l'espace utilisé
log_info "Espace utilisé par le dossier results:"
du -sh "$LOCAL_RESULTS_DIR" 2>/dev/null || log_warning "Impossible de calculer la taille"

# Option pour ouvrir le dossier results
echo
read -p "Veux-tu ouvrir le dossier results local ? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v nautilus >/dev/null 2>&1; then
        nautilus "$LOCAL_RESULTS_DIR" 2>/dev/null &
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$LOCAL_RESULTS_DIR" 2>/dev/null &
    else
        log_info "Contenu du dossier results:"
        ls -la "$LOCAL_RESULTS_DIR/"
    fi
fi