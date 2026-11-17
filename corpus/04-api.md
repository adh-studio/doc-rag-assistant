# API et integrations

Nimbus Backup expose une API RESTful qui permet de piloter les sauvegardes
depuis vos propres outils. L'API est disponible a partir de la formule Business.

L'authentification se fait par jeton personnel transmis dans l'en-tete
Authorization. Chaque jeton peut etre revoque a tout moment depuis la console
d'administration.

Les points d'entree principaux permettent de declencher une sauvegarde, de
lister les restaurations disponibles et de telecharger un fichier restaure. La
limite de debit est fixee a 600 requetes par minute et par jeton. Des connecteurs
prets a l'emploi existent pour Slack, Microsoft Teams et Zapier.
