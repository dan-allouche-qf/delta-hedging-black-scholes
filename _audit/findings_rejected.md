# Findings rejetés — delta-hedging-black-scholes

## R1 — "Seeds peut-être absentes" (hypothèse Phase 1)
- Hypothèse : Monte Carlo sans seed → résultats stochastiques → P0
- Réalité : `np.random.seed(42)` est posée cellule 2 ligne 9, AVANT toute simulation MC.
- Vérification : 4 runs jupyter nbconvert dans env propre (.venv-audit) → 140 nombres extraits par regex, identiques 4/4.
- Décision : REJETÉ. Reproductibilité = OK.

## R2 — "Chiffres README ↔ code peuvent diverger"
- Hypothèse : claim "sigma proportionnel 1/sqrt(n)", paramètres r=10% sigma=20% S0=100 K=90 T=0.25, 1000 simulations à confronter au notebook.
- Réalité : tous les chiffres concordent exactement.
  - Cell 4 output : r=10.0%, sigma=20.0%, S(0)=100.00, K=90.00, T=0.250, sims=1000 → match README.
  - Cell 10 std : n=1 → 1.6775, n=84 → 0.2125 → ratio 7.89 (notebook section 6.6 dit "by 7.9" → match).
  - Slope log-log std vs n = -0.47 (théorique -0.5) → confirme la loi 1/sqrt(n).
  - Tous les chiffres cités dans les markdowns d'interprétation (6.6, 7.4, 8.8, 9.1) correspondent aux sorties.
- Décision : REJETÉ. Cohérence rapport↔code = OK.

## R3 — Emojis / AI-slop verbeux
- grep emojis (catégories So + ranges 1F300–1FAFF + 2600–27BF) → 0 hit dans README et notebook.
- grep patterns AI (claude, voici, let me, in summary, …) → 0 hit.
- Décision : REJETÉ.

## R4 — Notebook variable `pi` recalculée à des endroits "inutiles"
- Hypothèse possible : `pi = delta_current * S + cash` recalculé après chaque rebal sans être lu ensuite avant le suivant.
- Réalité : `pi` est bien utilisé après la boucle (`error = pi - payoff`), et les recalculs intermédiaires servent au tracking conceptuel (lisibilité du suivi du portefeuille step-by-step). Ce n'est PAS du code mort sémantique.
- Décision : REJETÉ.
