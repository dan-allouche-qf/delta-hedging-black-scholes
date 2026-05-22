# Findings candidates — delta-hedging-black-scholes

## P1 candidates

### C1 — README claims Gamma computed but only Delta exists
- Fichier: `README.md:13` "implementation of Black-Scholes pricing and Greeks (Delta and Gamma)"
- Preuve attendue: aucune fonction Gamma dans le notebook
- Vérification: scan code cells → seules `black_scholes_call_price`, `black_scholes_delta`, `black_scholes_digital_price`, `black_scholes_digital_delta` définies. Aucun calcul de Gamma (partial² V / partial S²). Le mot "gamma" n'apparaît dans le notebook que dans le texte de discussion (8.8, 9.2) comme suggestion d'alternative future, pas comme implémentation.
- Décision: CONFIRMÉ → P1

## P2 candidates

### C2 — Unused import `from scipy import stats`
- Fichier: notebook cell 2 ligne 4 `from scipy import stats`
- Preuve attendue: pyflakes signale + aucune occurrence `stats.X` dans le code
- Vérification: pyflakes RC=1 `'scipy.stats' imported but unused`. Grep confirme : `stats.` apparaît 0 fois (seul `from scipy.stats import norm` est utilisé en réalité via `norm.cdf` / `norm.pdf`).
- Décision: CONFIRMÉ → P2

## P3 candidates (style/IA)

### C3 — README adjectifs marketing-y
- Fichier: `README.md:3` "comprehensive numerical analysis"
- Fichier: `README.md:13` "**Theoretical Rigor**"
- Fichier: `README.md:14` "A robust Monte Carlo framework"
- Décision: tournures factuelles tolérables mais légèrement assistant. À reformuler à sec.

### C4 — Identité (Phase 7)
- Cas A confirmé (README sans section auteur, notebook metadata.authors=None, git log unique = Dan Allouche).
- Action: ajouter section "## Auteur" en bas du README avec "Dan Allouche".

## Rejected pre-emptively
- Seeds : `np.random.seed(42)` cell 2 ligne 9, BEFORE first MC call. 4/4 runs identiques (140 nombres). REPRODUCTIBILITÉ OK.
- Chiffres README ↔ notebook : tous matchent exactement (mean -0.0949, std 9.5671 → 0.2125, ratio 7.89, slope log-log -0.47 ~= -0.5). REJETÉ COMME NON-FINDING.
- Emojis : 0 dans README et notebook. RAS.
- AI-slop patterns (claude, voici, let me, etc.) : 0 hit. RAS.
- Chemins absolus : 0 hit (pré-scan). RAS.
- Secrets : 0 hit trufflehog/gitleaks. RAS.
