# Sethos Qwen Image Edit 2511 — RunPod Serverless

Worker autonome pour **Administration > Outils créatifs > Créateur photo**.

- Modèle : `Qwen/Qwen-Image-Edit-2511`
- Contrat : `sethos.qwen.image-edit-2511.v1`
- Entrées : photo source signée, seconde référence facultative, prompt et réglages
- Sortie : WebP encodé en base64, copié ensuite dans le stockage privé du serveur Sethos
- Filtrage de contenu : aucun filtre fournisseur n’est intégré au worker ; les droits et le consentement restent contrôlés par l’interface

## Endpoint RunPod

Utiliser une carte 80 Gio (A100 ou H100), `workersMin=0`, `workersMax=1`, délai d’inactivité court et cache de modèle :

`Qwen/Qwen-Image-Edit-2511`

Le worker résout automatiquement le snapshot sous `/runpod-volume/huggingface-cache/hub/` et refuse de télécharger les poids pendant une tâche facturée.

## Image Vast.ai

`Dockerfile.vast` publie le worker HTTPS direct dans `sethosaicreation/sethos-ai-daily`.
Cette variante conserve le contrat Qwen existant, télécharge le snapshot verrouillé dans le disque
temporaire Vast et exige `VAST_WORKER_TOKEN`, `VAST_WORKER_TLS_CERT_B64` et
`VAST_WORKER_TLS_KEY_B64`. Le worker RunPod historique reste inchangé et hors ligne pendant
les tâches facturées.
