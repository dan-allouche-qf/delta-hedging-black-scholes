# Coherence rapport <-> code — delta-hedging-black-scholes

## Vérifications effectuées

### README <-> notebook
- Modèle GBM dS = S mu dt + S sigma dW : README L28 == notebook cell 0 / cell 8. **OK**
- Payoff call max(S(T)-K, 0) : README L34 == notebook cell 0 / cell 8. **OK**
- Paramètres r/sigma/S0/K/T/n_sims : README L40-45 == notebook cell 4 output. **OK**
- Régimes de fréquence "No / Single / Monthly / Weekly / Daily" : README L17-22 == notebook cell 4 (`n_values = [0, 1, 3, 12, 84]`) + libellés cell 10. **OK** (échelles cohérentes : T=3 mois, 1 mois=4 sem, 1 sem=7j → 3*4*7=84 jours).
- Loi "sigma proportionnel 1/sqrt(n)" : README L62. Recalculé empiriquement = slope log-log -0.47, ratio n=1/n=84 = 7.89 vs sqrt(84) = 9.17, std*sqrt(n) ~ 1.85-1.95 (quasi-constant). **OK**.
- "Mean error remains close to zero (under risk-neutral assumptions)" : confirmé par cell 10 (-0.0018 pour n=84) et cell 17 (0.0013 pour mu=r, n=84). **OK**.

### Notebook <-> notebook (texte d'interprétation <-> outputs numériques)
- Tous les chiffres cités dans les markdowns 6.6, 7.4, 8.8, 9.1 ont été extraits et comparés aux outputs. **22 claims, 22 matchs**.
- Voir `claims.csv` pour le détail.

### Méthodologie code <-> description mathématique
- Euler discretisation : README L136-140 (notebook cell 0) -> implémentation cell 8 ligne `S = S + S * mu * delta_t + S * sigma * np.sqrt(delta_t) * Z`. **Correspond.**
- Self-financing : description cell 0 section 2.2 -> implémentation cell 8 boucle (cash propagée à r continu, ajustement delta avec `cash = cash - delta_S_change`). **Correspond.**
- BS pricing : formule fermée cell 6 `S*N(d+) - K*exp(-rT)*N(d-)` -> formule cell 0 section 6.1. **Correspond.**

### Incohérence détectée et fixée
- README L13 mentionnait "Greeks (Delta and Gamma)" mais le notebook n'implémente pas Gamma. **CORRIGÉ** (F1) en remplaçant par "Delta for both European call and digital call options".

## Conclusion
Cohérence rapport <-> code = OK après fix F1. Aucune autre incohérence détectée.
