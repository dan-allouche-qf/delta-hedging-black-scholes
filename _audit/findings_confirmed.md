# Findings confirmés — delta-hedging-black-scholes

## P1

### F1 — README mentionne "Gamma" alors qu'aucun Gamma n'est implémenté
- Localisation: `README.md:13` "implementation of Black-Scholes pricing and Greeks (Delta and Gamma)"
- Preuve: notebook contient `black_scholes_delta`, `black_scholes_digital_delta`, mais aucune fonction pour calculer Gamma = partial²V / partial S². Aucune trace de "gamma" comme fonction ou variable Python dans le code. Le mot apparaît uniquement comme suggestion textuelle ("gamma-hedging" comme alternative non implémentée) dans les sections 8.8 et 9.2 du notebook.
- Fix : retirer "and Gamma" du bullet "Theoretical Rigor".
- Double-vérif : `grep -i gamma Delta_Hedging_Analysis.ipynb` retourne 2 hits, tous deux textuels, aucun code.

## P2

### F2 — Import inutilisé `from scipy import stats`
- Localisation: cellule code 2 du notebook (ligne 4 du bloc)
- Preuve: pyflakes RC=1 `'scipy.stats' imported but unused`. Grep `stats.` dans tout le code : 0 hit. (`norm` est importé séparément via `from scipy.stats import norm`, et c'est lui qui est utilisé.)
- Fix : supprimer la ligne `from scipy import stats`.

## P3

### F3 — README "Theoretical Rigor", "robust", "comprehensive"
- Localisation: README.md:3, :13, :14
- Fix : reformulation sobre.

## Phase 7 — Identité (Cas A)
### F4 — Aucune section Auteur dans le README
- Localisation: README.md
- Fix : ajouter à la fin une section `## Auteur` avec "Dan Allouche".
