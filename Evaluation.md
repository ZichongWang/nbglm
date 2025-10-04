# Official Metric Calculations: DES, PDS, and MAE

This document describes, in detail, how to compute the **Differential Expression Score (DES)**, **Perturbation Discrimination Score (PDS)**, and **Mean Absolute Error (MAE)** as defined in the official evaluation protocol for single‑cell functional genomics prediction tasks.

---

## Notation and Preliminaries

- Let \(\mathcal{P}=\{1,\dots,N\}\) be the set of perturbations; each perturbation \(k\in\mathcal{P}\) targets a specific **target gene** \(g^*(k)\).

- Let \(\mathcal{G}=\{1,\dots,G\}\) be the set of profiled genes (shared between prediction and ground truth).

- For each perturbation \(k\), let \(\mathcal{C}_k\) denote the set of **perturbed cells**, and \(\mathcal{C}_0\) the set of **control cells**.

- **Pseudobulk expression** for a perturbation is computed by averaging **log1p‑normalized** single‑cell expression across its cells:
  $$
  y_{k,g} \,=\, \frac{1}{|\mathcal{C}_k|} \sum_{i\in\mathcal{C}_k} \log\bigl(1+x_{i,g}\bigr),\qquad
  \hat y_{k,g} \,=\, \text{the corresponding predicted value},\quad g\in\mathcal{G}.
  $$
  The same normalization is applied to control cells to form \(y_{0,g}\) and \(\hat y_{0,g}\).

- **Fold change** for gene \(g\) under perturbation \(k\) is defined relative to control (any consistent fold‑change convention is acceptable; the official text refers to “with respect to control cells”). A common choice in practice is a difference of pseudobulk log1p means: \(\mathrm{FC}_{k,g}=y_{k,g}-y_{0,g}\), with its absolute magnitude \(|\mathrm{FC}_{k,g}|\).

---

## 1) Differential Expression Score (DES)

**Purpose.** Measure how accurately a model predicts the **set** of differentially expressed (DE) genes for each perturbation.

### 1.1 Per‑perturbation DE gene sets

For each perturbation \(k\):

1. **Hypothesis testing.** For both ground truth and prediction, compute gene‑wise differential expression p‑values between the perturbed and control cells using the **Wilcoxon rank‑sum test** (Mann–Whitney U) *with correction*.
2. **Multiple testing.** Apply the **Benjamini–Hochberg** (BH) procedure to control the false discovery rate (FDR) at level \(\alpha=0.05\).
3. **DE sets.**
   - Ground truth DE set: \(G_{k,\text{true}}=\{g\in\mathcal{G}: q_{k,g}^{\text{true}}<\alpha\}\) with size \(n_{k,\text{true}}=|G_{k,\text{true}}|\).
   - Predicted DE set: \(G_{k,\text{pred}}=\{g\in\mathcal{G}: q_{k,g}^{\text{pred}}<\alpha\}\) with size \(n_{k,\text{pred}}=|G_{k,\text{pred}}|\).

### 1.2 Normalized intersection (handling over‑prediction)

The **per‑perturbation DES** depends on the relative sizes of the sets:

- If \(n_{k,\text{pred}}\le n_{k,\text{true}}\):
  $$
  \mathrm{DES}_k \,=\, \frac{|G_{k,\text{pred}}\cap G_{k,\text{true}}|}{n_{k,\text{true}}}.
  $$

- If \(n_{k,\text{pred}}>n_{k,\text{true}}\): to **avoid over‑penalizing** predictions that select too many DE genes, define a truncated predicted set by taking the \(n_{k,\text{true}}\) genes with the **largest absolute fold changes** among \(G_{k,\text{pred}}\):
  $$
  \tilde G_{k,\text{pred}}\;=\;\operatorname*{arg\,top}_{S\subseteq G_{k,\text{pred}},\,|S|=n_{k,\text{true}}}
  \sum_{g\in S} |\mathrm{FC}_{k,g}|,\qquad
  \mathrm{DES}_k \,=\, \frac{|\tilde G_{k,\text{pred}}\cap G_{k,\text{true}}|}{n_{k,\text{true}}}.
  $$

### 1.3 Overall DES

$$
\mathrm{DES} \,=\, \frac{1}{N}\sum_{k=1}^N \mathrm{DES}_k.
$$

> **Remark (edge case).** The official description does not specify a special rule when \(n_{k,\text{true}}=0\). Implementations typically skip such perturbations when averaging or define a convention explicitly; the definition above remains as stated by the specification.

---

## 2) Perturbation Discrimination Score (PDS)

**Purpose.** Measure the model’s ability to **discriminate** between perturbations by ranking predictions according to their similarity to the **true** perturbational effects, regardless of effect size.

### 2.1 Pseudobulk profiles

For every perturbation \(k\), compute pseudobulk vectors \(\hat{\boldsymbol y}_k=(\hat y_{k,g})_{g\in\mathcal{G}}\) and \(\boldsymbol y_k=(y_{k,g})_{g\in\mathcal{G}}\) as the log1p‑mean expressions over cells.

### 2.2 Distance matrix with target‑gene exclusion

For a predicted perturbation \(p\), compute **Manhattan (\(L_1\)) distances** to all true perturbations \(t\):
$$
 d_{p,t} \,=\, \lVert\, \hat{\boldsymbol y}_p - \boldsymbol y_t \,\rVert_1\;\text{ over genes }\mathcal{G}\setminus\{g^*(p)\}.
$$
The **target gene** \(g^*(p)\) of perturbation \(p\) is **excluded** from the distance calculation.

### 2.3 Rank‑based discrimination score

For each \(p\), sort \(\{d_{p,t}\}_{t\in\mathcal{P}}\) in ascending order; let \(r_p\) be the **1‑based index** (rank) of the true perturbation \(t=p\) in this ordered list. The per‑perturbation score is
$$
 \mathrm{PDS}_p \,=\, 1 - \frac{r_p - 1}{N}.
$$
If the true perturbation is the nearest neighbor (\(r_p=1\)), then \(\mathrm{PDS}_p=1\).

### 2.4 Overall PDS

$$
\mathrm{PDS} \,=\, \frac{1}{N}\sum_{p=1}^N \mathrm{PDS}_p.
$$

---

## 3) Mean Absolute Error (MAE)

**Purpose.** Evaluate overall predictive accuracy across **all genes**, not only DE genes.

### 3.1 Per‑perturbation MAE

For each \(k\), with pseudobulk vectors \(\hat{\boldsymbol y}_k\) and \(\boldsymbol y_k\):
$$
 \mathrm{MAE}_k \,=\, \frac{1}{G}\sum_{g\in\mathcal{G}}\bigl|\hat y_{k,g}-y_{k,g}\bigr|.
$$

### 3.2 Overall MAE

$$
\mathrm{MAE} \,=\, \frac{1}{N}\sum_{k=1}^N \mathrm{MAE}_k.
$$

---

## 4) Scaled Scores and Overall Leaderboard Score

Leaderboard scoring averages the **improvements over a baseline** (cell‑mean model). Let the baseline metrics (pre‑computed on the training dataset) be
\(\mathrm{DES}_{\text{base}},\,\mathrm{PDS}_{\text{base}},\,\mathrm{MAE}_{\text{base}}\).

### 4.1 Scaled metrics (each clipped to \([0,1]\))

- **Differential Expression Score**
  $$
  \mathrm{DES}_{\text{scaled}} \,=\, \frac{\mathrm{DES}_{\text{pred}}-\mathrm{DES}_{\text{base}}}{1-\mathrm{DES}_{\text{base}}}\;\;\text{then clip to }[0,1].
  $$

- **Perturbation Discrimination Score**
  $$
  \mathrm{PDS}_{\text{scaled}} \,=\, \frac{\mathrm{PDS}_{\text{pred}}-\mathrm{PDS}_{\text{base}}}{1-\mathrm{PDS}_{\text{base}}}\;\;\text{then clip to }[0,1].
  $$

- **Mean Absolute Error** (smaller is better)
  $$
  \mathrm{MAE}_{\text{scaled}} \,=\, \frac{\mathrm{MAE}_{\text{base}}-\mathrm{MAE}_{\text{pred}}}{\mathrm{MAE}_{\text{base}}}\;\;\text{then clip to }[0,1].
  $$

### 4.2 Overall score

The final overall score reported on the leaderboard is the mean of the three scaled metrics, multiplied by 100:
$$
 \text{Overall} \,=\, 100\times \frac{\mathrm{DES}_{\text{scaled}}+\mathrm{PDS}_{\text{scaled}}+\mathrm{MAE}_{\text{scaled}}}{3}.
$$

---

## 5) End‑to‑End Algorithm (Reference)

For clarity, a concise algorithmic summary is provided below.

1. **Preprocessing (both pred & true):** total‑count normalization (e.g., target sum 1e4), then log1p per cell.
2. **Pseudobulk:** for each perturbation \(k\), average log1p values across cells to form \(\hat{\boldsymbol y}_k\) and \(\boldsymbol y_k\); likewise for control (\(\hat{\boldsymbol y}_0, \boldsymbol y_0\)).
3. **DES:** for each \(k\):
   1) run gene‑wise Wilcoxon rank‑sum test (perturbed vs control) for prediction and ground truth; 2) apply BH (\(\alpha=0.05\)) to get \(G_{k,\text{pred}}\) and \(G_{k,\text{true}}\); 3) if \(|G_{k,\text{pred}}|>|G_{k,\text{true}}|\), truncate predicted set to size \(|G_{k,\text{true}}|\) by largest \(|\mathrm{FC}|\); 4) set \(\mathrm{DES}_k=|G_{k,\text{pred}}\cap G_{k,\text{true}}|/|G_{k,\text{true}}|\); 5) average over \(k\).
4. **PDS:** for each predicted perturbation \(p\):
   1) compute Manhattan distances \(d_{p,t}\) to **true** perturbation profiles over \(\mathcal{G}\setminus\{g^*(p)\}\); 2) rank \(d_{p,t}\) ascending and let \(r_p\) be the 1‑based rank of \(t=p\); 3) set \(\mathrm{PDS}_p=1-(r_p-1)/N\); 4) average over \(p\).
5. **MAE:** for each \(k\), compute \(\mathrm{MAE}_k=(1/G)\sum_{g\in\mathcal{G}}|\hat y_{k,g}-y_{k,g}|\); average over \(k\).
6. **Scaled metrics & Overall:** compute the three scaled metrics w.r.t. the baseline (clipped to \([0,1]\)) and take the mean times 100.

---

### Notes

- The **target gene exclusion** applies **only** to the PDS distance calculation.
- The **Wilcoxon + BH (\(\alpha=0.05\))** thresholding defines DE gene sets for DES.
- Any internally consistent definition of fold change relative to control is acceptable for the DES truncation rule; the official text requires selecting the **largest absolute fold changes** among predicted significant genes.

