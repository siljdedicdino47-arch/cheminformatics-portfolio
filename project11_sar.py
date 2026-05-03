#project 11

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, rdMolDescriptors
from rdkit.Chem import rdFMCS, DataStructs, Draw
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import warnings, time
warnings.filterwarnings('ignore')
start_time = time.time()
print("=" * 80)
print("  STRUCTURE-ACTIVITY RELATIONSHIP (SAR) ANALYSIS ENGINE")
print("  Identifying Activity Cliffs and Property Trends Across Drug Series")
print("  University of Waterloo | Medicinal Chemistry")
print("=" * 80)

# ============================================================================
# WHAT THIS PROJECT DOES:
#
# SAR analysis answers the question: "How do small structural changes
# affect biological activity?" This is the bread-and-butter of medicinal
# chemistry.
#
# This project:
#   1. Takes a drug series (compounds targeting the same protein)
#   2. Finds the COMMON SCAFFOLD using Maximum Common Substructure (MCS)
#   3. Identifies what's DIFFERENT between each compound (R-groups)
#   4. Detects ACTIVITY CLIFFS — pairs where small structural changes
#      cause big activity jumps (a hallmark of unusual binding)
#   5. Generates SAR rules: "Adding X improves activity, removing Y hurts"
#
# This is what computational chemists generate for medicinal chemists
# to guide lead optimization.
# ============================================================================


# ============================================================================
# DRUG SERIES DATASETS (kinase inhibitors with known activity)
# ============================================================================

egfr_series = [
    {"name":"Gefitinib","smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4","ic50_nm":15},
    {"name":"Erlotinib","smiles":"COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC","ic50_nm":2},
    {"name":"Lapatinib","smiles":"CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)Cl)OCC5=CC(=CC=C5)F","ic50_nm":10},
    {"name":"Afatinib","smiles":"CN(C)C/C=C/C(=O)NC1=CC2=C(C=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl","ic50_nm":0.5},
    {"name":"Osimertinib","smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)NC(=O)C=C","ic50_nm":12},
    {"name":"Mobocertinib","smiles":"C=CC(=O)NC1=CC2=C(C=C1OC)N=CN=C2NC3=CC(=CC=C3)NC(=O)C=C","ic50_nm":35},
]

bcrabl_series = [
    {"name":"Imatinib","smiles":"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5","ic50_nm":100},
    {"name":"Nilotinib","smiles":"CC1=C(C=C(C=C1)NC(=O)C2=CC(=C(C=C2)C3=CN=C(N=C3)NC4=CC(=CC=C4)C(F)(F)F)C#N)NC5=NC=CC(=N5)C6=CC=CN=C6","ic50_nm":20},
    {"name":"Dasatinib","smiles":"CC1=NC(=CC(=N1)NC2=CC(=CC=C2)N3CCN(CC3)CCO)NC4=CC5=C(C=C4)C(=NN5)C(=O)NC6CCCCC6","ic50_nm":1},
    {"name":"Bosutinib","smiles":"COC1=C(C=C2C(=C1OC)N=CN=C2NC3=CC(=C(C=C3Cl)Cl)OC)OC4=CC(=CC=C4)CN5CCN(CC5)C","ic50_nm":50},
    {"name":"Ponatinib","smiles":"CC1=C(C=CC(=C1)C(=O)NC2=CC(=CC=C2)C(F)(F)F)C#CC3=CC4=C(C=C3)C(=NN4)NC5=CC=C(C=C5)N6CCN(CC6)C","ic50_nm":0.5},
]

cdk_series = [
    {"name":"Palbociclib","smiles":"CC(=O)C1=C(C=C2CN=C(N=C2N1)NC3=NC=C(C=C3)N4CCNCC4)C","ic50_nm":11},
    {"name":"Ribociclib","smiles":"CN(C)C(=O)C1=CC2=CN=C(N=C2N1C3CCCC3)NC4=NC=C(C=C4)N5CCNCC5","ic50_nm":10},
    {"name":"Abemaciclib","smiles":"CCN1C(=NC2=C1N=C(N=C2NC3=CC=C(C=C3)N4CCN(CC4)C)C5=CC(=NC=C5)F)C","ic50_nm":2},
]

all_series = {"EGFR Inhibitors": egfr_series, "BCR-ABL Inhibitors": bcrabl_series, "CDK4/6 Inhibitors": cdk_series}


# ============================================================================
# SAR ANALYSIS FUNCTIONS
# ============================================================================

def compute_pic50(ic50_nm):
    return -np.log10(ic50_nm * 1e-9)

def fingerprint(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

def find_mcs(mols, threshold=0.6):
    """Find the Maximum Common Substructure across a set of molecules."""
    if len(mols) < 2: return None
    try:
        result = rdFMCS.FindMCS(mols, threshold=threshold, timeout=30,
                                 atomCompare=rdFMCS.AtomCompare.CompareAny,
                                 bondCompare=rdFMCS.BondCompare.CompareAny)
        if result.canceled or result.numAtoms < 5:
            return None
        return result.smartsString
    except Exception as e:
        return None

def detect_activity_cliffs(series, similarity_threshold=0.5, activity_threshold=1.0):
    """
    Identify activity cliffs: pairs of similar molecules with very 
    different activities. These pairs are biologically interesting because
    they reveal critical structural features for binding.
    """
    cliffs = []
    for i, c1 in enumerate(series):
        for j, c2 in enumerate(series):
            if i >= j: continue
            mol1 = Chem.MolFromSmiles(c1["smiles"])
            mol2 = Chem.MolFromSmiles(c2["smiles"])
            if mol1 is None or mol2 is None: continue
            
            sim = DataStructs.TanimotoSimilarity(fingerprint(mol1), fingerprint(mol2))
            pic1 = compute_pic50(c1["ic50_nm"])
            pic2 = compute_pic50(c2["ic50_nm"])
            delta_pic = abs(pic1 - pic2)
            
            # Activity cliff: similar structure but big activity difference
            cliff_score = sim * delta_pic if sim > similarity_threshold else 0
            
            cliffs.append({
                "Mol_A": c1["name"], "Mol_B": c2["name"],
                "Similarity": round(sim, 3),
                "pIC50_A": round(pic1, 2), "pIC50_B": round(pic2, 2),
                "ΔpIC50": round(delta_pic, 2),
                "Cliff_Score": round(cliff_score, 3),
                "Is_Cliff": sim > similarity_threshold and delta_pic > activity_threshold,
            })
    
    return pd.DataFrame(cliffs).sort_values("Cliff_Score", ascending=False)


# ============================================================================
# RUN SAR ANALYSIS FOR EACH SERIES
# ============================================================================
all_results = {}

for series_name, series in all_series.items():
    print(f"\n{'='*80}")
    print(f"  ANALYZING SERIES: {series_name}")
    print(f"{'='*80}")
    print(f"   Compounds: {len(series)}")
    
    # Compute properties
    series_data = []
    mols = []
    for c in series:
        mol = Chem.MolFromSmiles(c["smiles"])
        if mol is None: continue
        mols.append(mol)
        series_data.append({
            "Name": c["name"], "SMILES": c["smiles"],
            "IC50_nM": c["ic50_nm"], "pIC50": round(compute_pic50(c["ic50_nm"]), 2),
            "MW": round(Descriptors.MolWt(mol), 1),
            "LogP": round(Descriptors.MolLogP(mol), 2),
            "TPSA": round(Descriptors.TPSA(mol), 1),
            "HBD": Lipinski.NumHDonors(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "HeavyAtoms": mol.GetNumHeavyAtoms(),
        })
    
    series_df = pd.DataFrame(series_data).sort_values("pIC50", ascending=False)
    print(f"\n   POTENCY RANKING:")
    for _, r in series_df.iterrows():
        bar = "█" * int(r['pIC50'])
        print(f"      {r['Name']:<22} pIC50={r['pIC50']:.2f} ({r['IC50_nM']:.1f} nM) {bar}")
    
    # Find MCS
    mcs_smarts = find_mcs(mols)
    if mcs_smarts:
        mcs_mol = Chem.MolFromSmarts(mcs_smarts)
        mcs_atoms = mcs_mol.GetNumAtoms() if mcs_mol else 0
        print(f"\n   COMMON SCAFFOLD (MCS): {mcs_atoms} atoms")
        print(f"      SMARTS: {mcs_smarts[:80]}...")
    else:
        print(f"\n   ⚠️ No common scaffold found (compounds too diverse)")
    
    # Detect activity cliffs
    cliffs_df = detect_activity_cliffs(series)
    print(f"\n   ACTIVITY CLIFFS (similar structure, different activity):")
    cliff_pairs = cliffs_df[cliffs_df['Is_Cliff']]
    if len(cliff_pairs) > 0:
        for _, r in cliff_pairs.head(5).iterrows():
            print(f"      {r['Mol_A']:<20} ↔ {r['Mol_B']:<20} sim={r['Similarity']:.3f} ΔpIC50={r['ΔpIC50']:.2f}")
    else:
        print(f"      No significant cliffs detected")
    
    # Property-activity correlations
    print(f"\n   PROPERTY-ACTIVITY CORRELATIONS:")
    for prop in ["MW", "LogP", "TPSA", "HBD"]:
        if series_df[prop].std() > 0:
            corr = series_df[prop].corr(series_df["pIC50"])
            direction = "↑" if corr > 0.3 else ("↓" if corr < -0.3 else "≈")
            strength = "strong" if abs(corr) > 0.7 else ("moderate" if abs(corr) > 0.4 else "weak")
            print(f"      {prop:<10} vs activity: r={corr:+.3f} {direction} ({strength})")
    
    all_results[series_name] = {
        "data": series_df,
        "cliffs": cliffs_df,
        "mcs": mcs_smarts,
    }


# ============================================================================
# CROSS-SERIES COMPARISON
# ============================================================================
print(f"\n{'='*80}\n  CROSS-SERIES SUMMARY\n{'='*80}")
print(f"\n   {'Series':<25} {'N':<5} {'pIC50 range':<15} {'MW avg':<8} {'LogP avg':<10}")
print(f"   {'-'*65}")
for name, results in all_results.items():
    df = results["data"]
    pic_range = f"{df['pIC50'].min():.1f} - {df['pIC50'].max():.1f}"
    print(f"   {name:<25} {len(df):<5} {pic_range:<15} {df['MW'].mean():<8.0f} {df['LogP'].mean():<10.2f}")


# ============================================================================
# VISUALIZATIONS
# ============================================================================
print(f"\n📊 Generating visualizations...")
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 22)); gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
fig.suptitle(f"SAR Analysis Engine — {sum(len(s) for s in all_series.values())} Compounds in 3 Series\nUniversity of Waterloo | Medicinal Chemistry",
             fontsize=17, fontweight='bold', y=0.99)

series_colors = {"EGFR Inhibitors":"#e74c3c", "BCR-ABL Inhibitors":"#3498db", "CDK4/6 Inhibitors":"#9b59b6"}

# Panel 1: Potency comparison across series
ax = fig.add_subplot(gs[0, 0])
y_pos = 0
for series_name, results in all_results.items():
    df = results["data"].sort_values("pIC50")
    ax.barh(range(y_pos, y_pos+len(df)), df['pIC50'], color=series_colors[series_name],
           edgecolor='white', linewidth=1, alpha=0.8, label=series_name)
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(0.05, y_pos+i, r['Name'], fontsize=7, va='center', color='white', fontweight='bold')
    y_pos += len(df) + 1
ax.set_xlabel("pIC50"); ax.set_ylabel("")
ax.set_yticks([]); ax.set_title("Potency comparison across drug series", fontweight='bold')
ax.legend(fontsize=9)

# Panel 2: Property-activity correlations heatmap
ax = fig.add_subplot(gs[0, 1])
all_corr_data = []
for series_name, results in all_results.items():
    df = results["data"]
    row = {"Series": series_name}
    for prop in ["MW", "LogP", "TPSA", "HBD", "HBA", "RotBonds"]:
        if df[prop].std() > 0:
            row[prop] = df[prop].corr(df["pIC50"])
        else:
            row[prop] = 0
    all_corr_data.append(row)
corr_df = pd.DataFrame(all_corr_data).set_index("Series")
im = ax.imshow(corr_df.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr_df.columns))); ax.set_xticklabels(corr_df.columns, rotation=45, ha='right')
ax.set_yticks(range(len(corr_df.index))); ax.set_yticklabels(corr_df.index)
for i in range(len(corr_df.index)):
    for j in range(len(corr_df.columns)):
        val = corr_df.values[i,j]
        ax.text(j, i, f'{val:+.2f}', ha='center', va='center',
               fontsize=9, color='white' if abs(val) > 0.5 else 'black', fontweight='bold')
plt.colorbar(im, ax=ax, label='Correlation r', shrink=0.7)
ax.set_title("Property-activity correlations by series", fontweight='bold')

# Panel 3-5: Activity cliffs scatter for each series
for idx, (series_name, results) in enumerate(all_results.items()):
    ax = fig.add_subplot(gs[1+idx//2, idx%2] if idx < 2 else gs[2, 0])
    if idx == 2:
        ax = fig.add_subplot(gs[2, 0])
    cliffs = results["cliffs"]
    colors = ['red' if c else 'gray' for c in cliffs['Is_Cliff']]
    sizes = [80 if c else 30 for c in cliffs['Is_Cliff']]
    ax.scatter(cliffs['Similarity'], cliffs['ΔpIC50'], c=colors, s=sizes, alpha=0.7,
              edgecolors='white', linewidth=1)
    ax.axvline(0.5, color='gray', ls='--', alpha=0.4, label='Similarity threshold')
    ax.axhline(1.0, color='gray', ls='--', alpha=0.4, label='Activity threshold')
    # Highlight top cliffs
    top_cliffs = cliffs[cliffs['Is_Cliff']].head(3)
    for _, r in top_cliffs.iterrows():
        ax.annotate(f"{r['Mol_A'][:6]}-{r['Mol_B'][:6]}", (r['Similarity'], r['ΔpIC50']),
                   fontsize=6, alpha=0.8, ha='center', va='bottom')
    ax.set_xlabel("Tanimoto similarity"); ax.set_ylabel("ΔpIC50")
    ax.set_title(f"{series_name}: Activity cliffs", fontweight='bold')
    ax.legend(fontsize=7)

# Panel 6: pIC50 vs LogP across all series (potency-lipophilicity)
ax = fig.add_subplot(gs[2, 1])
for series_name, results in all_results.items():
    df = results["data"]
    ax.scatter(df['LogP'], df['pIC50'], c=series_colors[series_name], s=120, alpha=0.7,
              edgecolors='white', linewidth=1.5, label=series_name)
    for _, r in df.iterrows():
        ax.annotate(r['Name'][:8], (r['LogP'], r['pIC50']), fontsize=6, alpha=0.7)
# LipE iso-lines
for lipe in [3, 5, 7]:
    x = np.linspace(0, 8, 50); y = x + lipe
    ax.plot(x, y, '--', alpha=0.2, color='black')
    ax.text(7.5, 7.5+lipe, f'LipE={lipe}', fontsize=7, alpha=0.5)
ax.set_xlabel("LogP"); ax.set_ylabel("pIC50")
ax.set_title("Potency-lipophilicity across all series", fontweight='bold')
ax.legend(fontsize=8)

# Panel 7: pIC50 distributions
ax = fig.add_subplot(gs[3, 0])
data = [results["data"]["pIC50"].values for results in all_results.values()]
labels = list(all_results.keys())
bp = ax.boxplot(data, labels=[l.replace(' ','\n') for l in labels], patch_artist=True, widths=0.5,
                medianprops=dict(color='black', linewidth=2))
for patch, name in zip(bp['boxes'], labels):
    patch.set_facecolor(series_colors[name]); patch.set_alpha(0.7)
ax.set_ylabel("pIC50"); ax.set_title("Potency distribution by series", fontweight='bold')

# Panel 8: MW distribution
ax = fig.add_subplot(gs[3, 1])
data = [results["data"]["MW"].values for results in all_results.values()]
bp = ax.boxplot(data, labels=[l.replace(' ','\n') for l in labels], patch_artist=True, widths=0.5,
                medianprops=dict(color='black', linewidth=2))
for patch, name in zip(bp['boxes'], labels):
    patch.set_facecolor(series_colors[name]); patch.set_alpha(0.7)
ax.set_ylabel("MW (Da)"); ax.set_title("Molecular weight distribution by series", fontweight='bold')

plt.savefig("project11_sar_analysis.png", dpi=150, bbox_inches='tight')
print(f"   ✅ project11_sar_analysis.png saved")

# Save all data
combined_df = pd.concat([
    results["data"].assign(Series=name) for name, results in all_results.items()
], ignore_index=True)
combined_df.to_csv("project11_sar_results.csv", index=False)
print(f"   ✅ project11_sar_results.csv saved")

elapsed = time.time() - start_time
print(f"\n⏱️  {elapsed:.1f}s\n{'='*70}\n  PIPELINE COMPLETE\n{'='*70}")
