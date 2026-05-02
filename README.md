# Cheminformatics Portfolio — Computational Drug Discovery

**Dino Siljdedic | Medicinal Chemistry @ University of Waterloo**

Five computational chemistry projects applying Python, RDKit, and data science to real drug discovery problems. Each project analyzes real pharmaceutical compounds using the same tools and methods used at Pfizer, Roche, and Novartis.

---

## Projects

### Project 1: Drug-Likeness Filter
Filters 106 real molecules through five validated drug-likeness criteria (Lipinski, Veber, Ghose, Lead-likeness, Rule of 3) and visualizes chemical space across 13 therapeutic areas.

**Key finding:** 92.5% of approved drugs pass Lipinski's Ro5, but only 55.7% pass the stricter Ghose filter — showing approved drugs are optimized well beyond simple lead-like properties.

![Project 1](images/project1_full_analysis.png)

---

### Project 2: PROTAC vs Traditional Drug Comparison
Compares PROTACs (the hottest modality in J. Med. Chem. 2026) against traditional drugs, molecular glues, and natural products using the "Beyond Rule of 5" framework.

**Key finding:** PROTACs have 1.9x more rotatable bonds than traditional drugs due to their linker, but molecular glues achieve targeted degradation at just 0.7x the MW.

![Project 2](images/project2_protac_analysis.png)

---

### Project 3: Molecular Similarity Search Engine
Computes ECFP4 Morgan fingerprints for 63 drugs and ranks database compounds by Tanimoto similarity. Includes pairwise kinase inhibitor heatmap and inter-class similarity analysis.

**Key finding:** Querying with Gefitinib correctly returns Osimertinib and Erlotinib as most similar — both EGFR inhibitors sharing the quinazoline scaffold.

![Project 3](images/project3_similarity_search.png)

---

### Project 4: ADMET Property Predictor
Predicts Absorption, Distribution, Metabolism, Excretion, and Toxicity risk flags from molecular structure using literature-validated thresholds across 11 endpoints.

**Key finding:** Rule-based ADMET predictions achieve 50% accuracy — demonstrating exactly why the field is moving toward ML models, the most active area in computational drug discovery.

![Project 4](images/project4_admet_predictions.png)

---

### Project 5: Covalent Drug Warhead Detector
Identifies reactive warhead groups (acrylamides, epoxides, vinyl sulfonamides, etc.) in drugs using SMARTS pattern matching. Compares covalent vs non-covalent drug properties.

**Key finding:** 92% warhead detection accuracy. Acrylamide is the dominant warhead in clinical covalent drugs (7/10 in dataset), consistent with the KRAS G12C inhibitor revolution.

![Project 5](images/project5_covalent_analysis.png)

---

## Tech Stack

- **RDKit** — molecular representation, descriptor calculation, fingerprints, SMARTS matching
- **Pandas** — data manipulation and analysis
- **Matplotlib** — publication-quality scientific visualization
- **NumPy** — numerical computing
- **scikit-learn** — machine learning (used in upcoming QSAR project)
- **ChEMBL API** (optional) — live querying of the world's largest bioactivity database

## How to Run

```bash
# Create environment
conda create -n cheminformatics -c conda-forge rdkit pandas matplotlib numpy
conda activate cheminformatics

# Optional: enable live ChEMBL queries (2.4M+ compounds)
pip install chembl_webresource_client

# Run any project
python project1_final.py
python project2_final.py
python project3_final.py
python project4_final.py
python project5_final.py
```

Each project has a `USE_CHEMBL` flag at the top — set it to `True` to query the live ChEMBL database instead of the built-in curated dataset.

## Data Sources

- **Curated dataset:** 100+ real compounds with SMILES from PubChem, organized by therapeutic area
- **ChEMBL 36** (July 2025): 2.4M bioactive molecules — queryable via API or downloadable

## Journal Context

These projects reflect current trends in medicinal chemistry literature:
- **PROTACs & molecular glues** — dominant theme in J. Med. Chem. 2025-2026
- **Covalent inhibitors & KRAS G12C** — "The Ascension of Targeted Covalent Inhibitors" (J. Med. Chem.)
- **AI-driven ADMET prediction** — "Leveraging ML models in evaluating ADMET" (ADMET & DMPK, 2025)
- **Molecular fingerprints & similarity** — foundational cheminformatics methodology

## Contact

- **LinkedIn:** [linkedin.com/in/dino-siljdedic](https://linkedin.com/in/dino-siljdedic)
- **Email:** dsiljded@uwaterloo.ca
- University of Waterloo — Medicinal Chemistry, Expected 2029
