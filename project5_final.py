#project 5

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, rdMolDescriptors, BRICS
from rdkit.Chem import Draw, rdmolops
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import numpy as np
import warnings, time
warnings.filterwarnings('ignore')
start_time = time.time()
print("=" * 80)
print("  COVALENT DRUG WARHEAD ANALYZER")
print("  Detecting Reactive Groups in Targeted Covalent Inhibitors")
print("  University of Waterloo | Medicinal Chemistry")
print("=" * 80)

# ============================================================================
# WHAT THIS PROJECT DOES:
# ============================================================================
# Targeted Covalent Inhibitors (TCIs) are drugs that form a permanent 
# chemical bond with their target protein. They use "warheads" — small 
# reactive chemical groups that attack specific amino acids (usually cysteine).
#
# This project:
#   1. Defines SMARTS patterns for known warhead types
#   2. Scans drug molecules to detect which warhead they contain
#   3. Compares properties of covalent vs non-covalent drugs
#   4. Analyzes which warheads are most common in clinical drugs
#
# Inspired by:
#   - "Emerging and Re-Emerging Warheads for TCIs" (J. Med. Chem. 2019)
#   - "The Ascension of Targeted Covalent Inhibitors" (J. Med. Chem. 2023)
#   - "Covalent Inhibitors: To Infinity and Beyond" (J. Med. Chem. 2024)
#   - "Covalent-Allosteric Inhibitors" (J. Med. Chem. 2025)
# ============================================================================


# ============================================================================
# WARHEAD DEFINITIONS (SMARTS patterns)
# ============================================================================
# SMARTS is like a search language for chemical structures.
# It lets you describe a pattern and find molecules that match.
#
# For example:
#   [CH2]=[CH]C(=O)   matches an acrylamide warhead
#   This says: "find a CH2 double-bonded to a CH, connected to a C=O"
#
# Think of SMARTS as regex (regular expressions) but for molecules.

WARHEADS = {
    "Acrylamide": {
        "smarts": "[CH2]=[CH]C(=O)[N,O]",
        "description": "Michael acceptor — reacts with cysteine thiol via conjugate addition",
        "example_drug": "Osimertinib, Ibrutinib",
        "target_residue": "Cysteine",
        "reactivity": "Moderate",
    },
    "Vinyl Sulfonamide": {
        "smarts": "[CH2]=[CH]S(=O)(=O)",
        "description": "Stronger Michael acceptor than acrylamide",
        "example_drug": "Research compounds",
        "target_residue": "Cysteine",
        "reactivity": "High",
    },
    "Chloroacetamide": {
        "smarts": "ClCC(=O)N",
        "description": "SN2 alkylation of cysteine — irreversible",
        "example_drug": "Research compounds",
        "target_residue": "Cysteine",
        "reactivity": "High",
    },
    "Cyanoacrylamide": {
        "smarts": "C(=C)C#N",
        "description": "Reversible covalent warhead — weaker Michael acceptor",
        "example_drug": "Research compounds",
        "target_residue": "Cysteine",
        "reactivity": "Low (reversible)",
    },
    "Epoxide": {
        "smarts": "C1OC1",
        "description": "Strained ring opens to form covalent bond",
        "example_drug": "Carfilzomib (proteasome inhibitor)",
        "target_residue": "Threonine",
        "reactivity": "Moderate",
    },
    "Alpha-beta unsaturated ketone": {
        "smarts": "[CH2]=[CH]C(=O)[C,c]",
        "description": "Michael acceptor — enone warhead",
        "example_drug": "Various natural products",
        "target_residue": "Cysteine",
        "reactivity": "High",
    },
    "Nitrile (cyanamide)": {
        "smarts": "[N]C#N",
        "description": "Reacts with catalytic cysteine to form thioimidate",
        "example_drug": "Saxagliptin (DPP-4 inhibitor)",
        "target_residue": "Cysteine/Serine",
        "reactivity": "Moderate",
    },
    "Boronic acid": {
        "smarts": "[B]([OH])([OH])",
        "description": "Reversible covalent — forms tetrahedral adduct with serine",
        "example_drug": "Bortezomib (proteasome inhibitor)",
        "target_residue": "Serine/Threonine",
        "reactivity": "Low (reversible)",
    },
    "Acyl fluoride/activated ester": {
        "smarts": "C(=O)F",
        "description": "Highly reactive — acylates serine or lysine",
        "example_drug": "Research compounds",
        "target_residue": "Serine/Lysine",
        "reactivity": "Very High",
    },
    "Sulfonyl fluoride": {
        "smarts": "S(=O)(=O)F",
        "description": "Reacts with serine, tyrosine, lysine, histidine",
        "example_drug": "PMSF (research tool)",
        "target_residue": "Multiple",
        "reactivity": "Moderate",
    },
}

print(f"\n🔧 Loaded {len(WARHEADS)} warhead SMARTS patterns")
for name, info in WARHEADS.items():
    print(f"   {name}: {info['smarts']} → targets {info['target_residue']}")


# ============================================================================
# COMPOUND DATABASE
# ============================================================================

def get_compounds():
    compounds = [
        # === COVALENT DRUGS (FDA-approved or clinical) ===
        {"name":"Osimertinib","smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)NC(=O)C=C","class":"Covalent","target":"EGFR T790M","indication":"NSCLC","phase":4},
        {"name":"Ibrutinib","smiles":"C=CC(=O)N1CCC(CC1)N2C3=C(C=CC=C3)N=C2C4=CC=C(C=C4)OC5=CC=CC=C5","class":"Covalent","target":"BTK","indication":"CLL","phase":4},
        {"name":"Afatinib","smiles":"CN(C)C/C=C/C(=O)NC1=CC2=C(C=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl","class":"Covalent","target":"EGFR/HER2","indication":"NSCLC","phase":4},
        {"name":"Neratinib","smiles":"CCOC1=CC2=C(C=C1NC(=O)/C=C/CN(C)C)N=CN=C2NC3=CC(=C(C=C3)Cl)OCC4=CC(=CC=C4)Cl","class":"Covalent","target":"HER2","indication":"Breast Ca","phase":4},
        {"name":"Sotorasib","smiles":"C=CC(=O)N1CCN(CC1)C2=NC(=NC3=C2C(=CC=C3)F)NC4=CC(=C(C=C4)F)OC(F)F","class":"Covalent","target":"KRAS G12C","indication":"NSCLC","phase":4},
        {"name":"Adagrasib","smiles":"CC(C)(O)CN1CCN(CC1)C2=NC(=NC3=CC=C(C=C23)OC)NC4=CC(=C(C(=C4)F)C(=O)C=C)F","class":"Covalent","target":"KRAS G12C","indication":"NSCLC","phase":4},
        {"name":"Zanubrutinib","smiles":"C=CC(=O)N1CCC2(CC1)CN(C2)C3=NC(=NC4=CC=CC=C34)NC5=CC=C(C=C5)OC6=CC=CC=C6","class":"Covalent","target":"BTK","indication":"MCL","phase":4},
        {"name":"Acalabrutinib","smiles":"C=CC(=O)N1CCC(CC1)N2C3=C(N=C2C(N)=O)C=CC=C3","class":"Covalent","target":"BTK","indication":"CLL","phase":4},
        {"name":"Mobocertinib","smiles":"C=CC(=O)NC1=CC2=C(C=C1OC)N=CN=C2NC3=CC(=CC=C3)NC(=O)C=C","class":"Covalent","target":"EGFR exon20","indication":"NSCLC","phase":4},
        {"name":"Futibatinib","smiles":"C=CC(=O)NC1=CC2=C(C=C1)N=C(N=C2N)C3=CC=CC=N3","class":"Covalent","target":"FGFR","indication":"Cholangiocarcinoma","phase":4},

        # === NON-COVALENT DRUGS (for comparison) ===
        {"name":"Imatinib","smiles":"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5","class":"Non-covalent","target":"BCR-ABL","indication":"CML","phase":4},
        {"name":"Erlotinib","smiles":"COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC","class":"Non-covalent","target":"EGFR","indication":"NSCLC","phase":4},
        {"name":"Gefitinib","smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4","class":"Non-covalent","target":"EGFR","indication":"NSCLC","phase":4},
        {"name":"Sorafenib","smiles":"CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F","class":"Non-covalent","target":"Multi-kinase","indication":"HCC","phase":4},
        {"name":"Dasatinib","smiles":"CC1=NC(=CC(=N1)NC2=CC(=CC=C2)N3CCN(CC3)CCO)NC4=CC5=C(C=C4)C(=NN5)C(=O)NC6CCCCC6","class":"Non-covalent","target":"BCR-ABL/SRC","indication":"CML","phase":4},
        {"name":"Palbociclib","smiles":"CC(=O)C1=C(C=C2CN=C(N=C2N1)NC3=NC=C(C=C3)N4CCNCC4)C","class":"Non-covalent","target":"CDK4/6","indication":"Breast Ca","phase":4},
        {"name":"Vemurafenib","smiles":"CCCS(=O)(=O)NC1=CC(=C(C=C1F)C(=O)C2=CNC3=NC=C(C=C23)C4=CC=C(C=C4)Cl)F","class":"Non-covalent","target":"BRAF V600E","indication":"Melanoma","phase":4},
        {"name":"Olaparib","smiles":"C1CC1C(=O)N2CCN(CC2)C(=O)C3=C(C=CC=C3F)CC4=NNC(=O)C5=CC=CC=C54","class":"Non-covalent","target":"PARP","indication":"Ovarian Ca","phase":4},
        {"name":"Ruxolitinib","smiles":"N#CCC(C1=CC=CN=C1)N2CC3=C(C2)C(=NN3)C4CCCC4","class":"Non-covalent","target":"JAK1/2","indication":"MPN","phase":4},
        {"name":"Crizotinib","smiles":"CC(C1=C(C=CC(=C1Cl)F)Cl)OC2=CN=C3C=C(C=CC3=C2)C4(CC4)N","class":"Non-covalent","target":"ALK","indication":"NSCLC","phase":4},
        {"name":"Baricitinib","smiles":"CCS(=O)(=O)N1CC(C1)N2C=C(C(=N2)C3=CC=NC=C3)C#N","class":"Non-covalent","target":"JAK1/2","indication":"RA","phase":4},
        {"name":"Fluoxetine","smiles":"CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F","class":"Non-covalent","target":"SERT","indication":"Depression","phase":4},
        {"name":"Metformin","smiles":"CN(C)C(=N)NC(=N)N","class":"Non-covalent","target":"AMPK","indication":"T2DM","phase":4},
        {"name":"Aspirin","smiles":"CC(=O)OC1=CC=CC=C1C(=O)O","class":"Non-covalent","target":"COX-1/2","indication":"Pain","phase":4},
        {"name":"Omeprazole","smiles":"CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2","class":"Non-covalent","target":"H+/K+ ATPase","indication":"GERD","phase":4},
    ]
    return compounds


# ============================================================================
# WARHEAD DETECTION ENGINE
# ============================================================================

def detect_warheads(mol):
    """
    Scan a molecule for known covalent warhead groups using SMARTS matching.
    Returns a list of detected warhead names.
    """
    if mol is None:
        return []
    
    detected = []
    for wh_name, wh_info in WARHEADS.items():
        pattern = Chem.MolFromSmarts(wh_info["smarts"])
        if pattern is None:
            continue
        if mol.HasSubstructMatch(pattern):
            detected.append(wh_name)
    
    return detected


# ============================================================================
# STEP 1: PROCESS ALL COMPOUNDS
# ============================================================================
compounds = get_compounds()
print(f"\n📦 Loaded {len(compounds)} compounds")

results = []
for entry in compounds:
    mol = Chem.MolFromSmiles(entry["smiles"])
    if mol is None:
        continue
    
    warheads = detect_warheads(mol)
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    fsp3 = Descriptors.FractionCSP3(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    
    lip_v = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    
    results.append({
        "Name": entry["name"],
        "Class": entry["class"],
        "Target": entry["target"],
        "Indication": entry["indication"],
        "Warheads_Detected": ", ".join(warheads) if warheads else "None",
        "Num_Warheads": len(warheads),
        "Is_Covalent_Predicted": len(warheads) > 0,
        "Is_Covalent_Actual": entry["class"] == "Covalent",
        "MW": round(mw, 1),
        "LogP": round(logp, 2),
        "TPSA": round(tpsa, 1),
        "HBD": hbd, "HBA": hba,
        "RotBonds": rot_bonds,
        "Fsp3": round(fsp3, 3),
        "HeavyAtoms": heavy_atoms,
        "AromaticRings": aromatic_rings,
        "Lipinski_Violations": lip_v,
    })

df = pd.DataFrame(results)
print(f"   ✅ Processed {len(df)} compounds")


# ============================================================================
# STEP 2: WARHEAD DETECTION RESULTS
# ============================================================================
print(f"\n{'=' * 80}")
print(f"  WARHEAD DETECTION RESULTS")
print(f"{'=' * 80}")

print(f"\n📋 DETECTION BY COMPOUND")
for _, row in df.iterrows():
    wh = row['Warheads_Detected']
    actual = "COVALENT" if row['Is_Covalent_Actual'] else "Non-cov"
    predicted = "✅" if row['Is_Covalent_Predicted'] == row['Is_Covalent_Actual'] else "❌"
    print(f"  {predicted} {row['Name']:<18} [{actual:<8}] Warheads: {wh}")

# Accuracy
correct = (df['Is_Covalent_Predicted'] == df['Is_Covalent_Actual']).sum()
total = len(df)
print(f"\n  Detection accuracy: {correct}/{total} ({correct/total*100:.0f}%)")

# Warhead frequency
print(f"\n📋 WARHEAD FREQUENCY IN COVALENT DRUGS")
covalent_df = df[df['Is_Covalent_Actual']]
all_warheads_found = []
for _, row in covalent_df.iterrows():
    if row['Warheads_Detected'] != "None":
        for wh in row['Warheads_Detected'].split(", "):
            all_warheads_found.append(wh)

wh_counts = pd.Series(all_warheads_found).value_counts()
for wh, count in wh_counts.items():
    bar = "█" * count
    print(f"   {wh:<30} {count:>3}  {bar}")


# ============================================================================
# STEP 3: COVALENT vs NON-COVALENT PROPERTY COMPARISON
# ============================================================================
print(f"\n{'=' * 80}")
print(f"  PROPERTY COMPARISON: COVALENT vs NON-COVALENT")
print(f"{'=' * 80}")

cov = df[df['Is_Covalent_Actual']]
ncov = df[~df['Is_Covalent_Actual']]

props = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds", "Fsp3", "AromaticRings"]
print(f"\n{'Property':<15} {'Covalent (n=' + str(len(cov)) + ')':>22} {'Non-covalent (n=' + str(len(ncov)) + ')':>25}")
print("-" * 65)
for prop in props:
    c_mean = cov[prop].mean()
    c_std = cov[prop].std()
    n_mean = ncov[prop].mean()
    n_std = ncov[prop].std()
    diff_pct = ((c_mean - n_mean) / n_mean * 100) if n_mean != 0 else 0
    direction = "↑" if diff_pct > 5 else ("↓" if diff_pct < -5 else "≈")
    print(f"   {prop:<12} {c_mean:>9.1f} ± {c_std:>5.1f}    {n_mean:>9.1f} ± {n_std:>5.1f}  {direction} {abs(diff_pct):.0f}%")

print(f"""
KEY OBSERVATIONS:
  - Covalent drugs tend to have {'higher' if cov['MW'].mean() > ncov['MW'].mean() else 'lower'} MW 
    ({cov['MW'].mean():.0f} vs {ncov['MW'].mean():.0f} Da)
  - Covalent drugs have {'more' if cov['RotBonds'].mean() > ncov['RotBonds'].mean() else 'fewer'} rotatable bonds
    ({cov['RotBonds'].mean():.1f} vs {ncov['RotBonds'].mean():.1f}) — the warhead adds flexibility
  - The acrylamide warhead (C=CC=O) is by far the most common warhead type
  - KRAS G12C inhibitors (sotorasib, adagrasib) represent the most important
    recent breakthrough in covalent drug design
""")


# ============================================================================
# STEP 4: VISUALIZATIONS
# ============================================================================
print(f"📊 Generating visualizations...")

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 18))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
fig.suptitle(f"Covalent Drug Warhead Analysis — {len(df)} Compounds\nUniversity of Waterloo | Medicinal Chemistry",
             fontsize=17, fontweight='bold', y=0.98)

cov_color = "#e74c3c"
ncov_color = "#3498db"

# Panel 1: Chemical space colored by covalent/non-covalent
ax = fig.add_subplot(gs[0, 0])
ax.scatter(ncov['MW'], ncov['LogP'], c=ncov_color, s=60, alpha=0.7,
          edgecolors='white', linewidth=1, label=f'Non-covalent (n={len(ncov)})', zorder=5)
ax.scatter(cov['MW'], cov['LogP'], c=cov_color, s=80, alpha=0.8,
          edgecolors='white', linewidth=1.5, label=f'Covalent (n={len(cov)})', marker='D', zorder=6)
for _, row in cov.iterrows():
    ax.annotate(row['Name'][:12], (row['MW'], row['LogP']), fontsize=6, alpha=0.7,
               ha='center', va='bottom', fontweight='bold')
ax.axvline(500, color='gray', ls='--', alpha=0.3)
ax.axhline(5, color='gray', ls='--', alpha=0.3)
ax.set_xlabel("Molecular Weight (Da)"); ax.set_ylabel("LogP")
ax.set_title("Chemical space: covalent vs non-covalent drugs", fontweight='bold')
ax.legend(fontsize=9)

# Panel 2: Warhead frequency bar chart
ax = fig.add_subplot(gs[0, 1])
if len(wh_counts) > 0:
    colors_wh = plt.cm.Reds(np.linspace(0.3, 0.8, len(wh_counts)))
    bars = ax.barh(range(len(wh_counts)), wh_counts.values, color=colors_wh,
                   edgecolor='white', linewidth=1)
    ax.set_yticks(range(len(wh_counts)))
    ax.set_yticklabels(wh_counts.index, fontsize=9)
    ax.set_xlabel("Count in dataset")
    for bar, val in zip(bars, wh_counts.values):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
               str(val), va='center', fontweight='bold')
ax.set_title("Warhead type frequency in covalent drugs", fontweight='bold')

# Panel 3: Property box plots — MW
ax = fig.add_subplot(gs[1, 0])
data_box = [cov['MW'].values, ncov['MW'].values]
bp = ax.boxplot(data_box, labels=['Covalent', 'Non-covalent'], patch_artist=True, widths=0.5,
                medianprops=dict(color='black', linewidth=2))
bp['boxes'][0].set_facecolor(cov_color); bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor(ncov_color); bp['boxes'][1].set_alpha(0.7)
for i, (data, color) in enumerate(zip(data_box, [cov_color, ncov_color])):
    jitter = np.random.normal(i+1, 0.05, len(data))
    ax.scatter(jitter, data, c=color, s=30, alpha=0.6, edgecolors='white', linewidth=0.5, zorder=5)
ax.set_ylabel("Molecular Weight (Da)")
ax.set_title("MW comparison", fontweight='bold')

# Panel 4: Property box plots — LogP and TPSA
ax = fig.add_subplot(gs[1, 1])
props_compare = ["LogP", "TPSA", "RotBonds"]
x = np.arange(len(props_compare)); w = 0.35
cov_means = [cov[p].mean() for p in props_compare]
ncov_means = [ncov[p].mean() for p in props_compare]
cov_stds = [cov[p].std() for p in props_compare]
ncov_stds = [ncov[p].std() for p in props_compare]
ax.bar(x - w/2, cov_means, w, yerr=cov_stds, label='Covalent', color=cov_color,
       edgecolor='white', linewidth=1, capsize=4, alpha=0.8)
ax.bar(x + w/2, ncov_means, w, yerr=ncov_stds, label='Non-covalent', color=ncov_color,
       edgecolor='white', linewidth=1, capsize=4, alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(props_compare)
ax.set_ylabel("Mean value"); ax.set_title("Property comparison (mean ± std)", fontweight='bold')
ax.legend(fontsize=9)

# Panel 5: Target distribution for covalent drugs
ax = fig.add_subplot(gs[2, 0])
target_counts = cov['Target'].value_counts()
colors_targets = plt.cm.Reds(np.linspace(0.3, 0.8, len(target_counts)))
ax.bar(range(len(target_counts)), target_counts.values, color=colors_targets,
       edgecolor='white', linewidth=1)
ax.set_xticks(range(len(target_counts)))
ax.set_xticklabels(target_counts.index, rotation=45, ha='right', fontsize=8)
ax.set_ylabel("Number of drugs")
ax.set_title("Targets of covalent drugs in dataset", fontweight='bold')

# Panel 6: Detection accuracy confusion matrix
ax = fig.add_subplot(gs[2, 1])
tp = ((df['Is_Covalent_Predicted']) & (df['Is_Covalent_Actual'])).sum()
fp = ((df['Is_Covalent_Predicted']) & (~df['Is_Covalent_Actual'])).sum()
fn = ((~df['Is_Covalent_Predicted']) & (df['Is_Covalent_Actual'])).sum()
tn = ((~df['Is_Covalent_Predicted']) & (~df['Is_Covalent_Actual'])).sum()

conf = np.array([[tp, fn], [fp, tn]])
im = ax.imshow(conf, cmap='Blues', aspect='auto')
ax.set_xticks([0, 1]); ax.set_xticklabels(['Covalent\n(actual)', 'Non-covalent\n(actual)'])
ax.set_yticks([0, 1]); ax.set_yticklabels(['Predicted\ncovalent', 'Predicted\nnon-covalent'])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(conf[i, j]), ha='center', va='center', fontsize=20, fontweight='bold',
               color='white' if conf[i, j] > conf.max()/2 else 'black')
ax.set_title(f"Warhead detection: {correct}/{total} correct ({correct/total*100:.0f}%)", fontweight='bold')

plt.savefig("project5_covalent_analysis.png", dpi=150, bbox_inches='tight')
print(f"   ✅ Figure saved: project5_covalent_analysis.png")

df.to_csv("project5_covalent_results.csv", index=False)
print(f"   ✅ Data saved: project5_covalent_results.csv")

elapsed = time.time() - start_time
print(f"\n⏱️  Runtime: {elapsed:.1f}s")
print(f"\n{'=' * 70}")
print(f"  PIPELINE COMPLETE")
print(f"{'=' * 70}")
