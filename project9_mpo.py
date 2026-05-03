#project 9

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, rdMolDescriptors
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import warnings, time
warnings.filterwarnings('ignore')
start_time = time.time()
print("=" * 80)
print("  MULTI-PARAMETER OPTIMIZATION (MPO) DASHBOARD")
print("  Ranking Drug Candidates Like Real Medicinal Chemists Do")
print("  University of Waterloo | Medicinal Chemistry")
print("=" * 80)

# ============================================================================
# WHAT THIS PROJECT DOES:
#
# Real drug optimization is NEVER one-dimensional. You can't just maximize
# potency — you have to balance potency, selectivity, ADMET, synthesis cost,
# and IP space simultaneously. A drug candidate is good only if it scores
# well across ALL parameters.
#
# This project implements three industry-standard MPO scoring systems:
#
#   1. CNS MPO (Pfizer) — for brain-penetrant drugs
#      Score 0-6, higher is better. Combines LogP, LogD, MW, TPSA, HBD, pKa
#
#   2. Ligand Efficiency (LE) — activity per heavy atom
#      LE = -RT × ln(IC50) / N_heavy_atoms
#      Higher LE means each atom is "earning its keep"
#
#   3. Lipophilic Efficiency (LipE) — activity vs LogP balance
#      LipE = pIC50 - LogP
#      Tells you if potency is "real" or just from making the drug greasy
#
# Inspired by:
#   - Wager et al. (2010) "CNS MPO" (ACS Chem. Neurosci.)
#   - Hopkins et al. (2014) "Ligand efficiency" (Nat. Rev. Drug Disc.)
#   - "Drug Design Strategies for Multi-Parameter Optimization" (J. Med. Chem.)
# ============================================================================


# ============================================================================
# DATASET: Drug candidates with simulated biological activity
# ============================================================================
# In real workflow, IC50 comes from biological assays. We use realistic
# values from published EGFR inhibitor data + plausible estimates.

candidates = [
    # FDA-approved EGFR inhibitors (gold standards)
    {"name":"Gefitinib","smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4","ic50_nm":15,"target":"EGFR","status":"Approved"},
    {"name":"Erlotinib","smiles":"COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC","ic50_nm":2,"target":"EGFR","status":"Approved"},
    {"name":"Osimertinib","smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)NC(=O)C=C","ic50_nm":12,"target":"EGFR T790M","status":"Approved"},
    {"name":"Afatinib","smiles":"CN(C)C/C=C/C(=O)NC1=CC2=C(C=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl","ic50_nm":0.5,"target":"EGFR","status":"Approved"},
    {"name":"Lapatinib","smiles":"CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)Cl)OCC5=CC(=CC=C5)F","ic50_nm":10,"target":"EGFR/HER2","status":"Approved"},
    
    # Other approved kinase inhibitors
    {"name":"Imatinib","smiles":"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5","ic50_nm":100,"target":"BCR-ABL","status":"Approved"},
    {"name":"Dasatinib","smiles":"CC1=NC(=CC(=N1)NC2=CC(=CC=C2)N3CCN(CC3)CCO)NC4=CC5=C(C=C4)C(=NN5)C(=O)NC6CCCCC6","ic50_nm":1,"target":"BCR-ABL","status":"Approved"},
    {"name":"Palbociclib","smiles":"CC(=O)C1=C(C=C2CN=C(N=C2N1)NC3=NC=C(C=C3)N4CCNCC4)C","ic50_nm":11,"target":"CDK4/6","status":"Approved"},
    {"name":"Ibrutinib","smiles":"C=CC(=O)N1CCC(CC1)N2C3=C(C=CC=C3)N=C2C4=CC=C(C=C4)OC5=CC=CC=C5","ic50_nm":0.5,"target":"BTK","status":"Approved"},
    {"name":"Vemurafenib","smiles":"CCCS(=O)(=O)NC1=CC(=C(C=C1F)C(=O)C2=CNC3=NC=C(C=C23)C4=CC=C(C=C4)Cl)F","ic50_nm":31,"target":"BRAF V600E","status":"Approved"},
    
    # CNS drugs (where CNS MPO matters most)
    {"name":"Donepezil","smiles":"COC1=CC2=C(C=C1OC)C(=O)C(C2)CC3CCN(CC3)CC4=CC=CC=C4","ic50_nm":7,"target":"AChE","status":"Approved"},
    {"name":"Memantine","smiles":"C1C2CC3CC1CC(C2)(C3)N","ic50_nm":1000,"target":"NMDA","status":"Approved"},
    {"name":"Fluoxetine","smiles":"CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F","ic50_nm":4,"target":"SERT","status":"Approved"},
    {"name":"Sertraline","smiles":"CNC1CCC(C2=CC=CC=C21)C3=CC(=C(C=C3)Cl)Cl","ic50_nm":3,"target":"SERT","status":"Approved"},
    {"name":"Aripiprazole","smiles":"O=C1CC2=CC=CC=C2N1CCCCOC3=CC=C(C=C3)Cl","ic50_nm":2,"target":"D2","status":"Approved"},
    
    # Hypothetical optimization candidates (showing different MPO profiles)
    {"name":"Candidate_A_potent","smiles":"COC1=CC2=NC=NC(=C2C=C1)NC3=CC=C(C=C3)C(F)(F)F","ic50_nm":5,"target":"EGFR","status":"Investigational"},
    {"name":"Candidate_B_balanced","smiles":"COC1=CC2=NC=NC(=C2C=C1OC)NC3=CC=C(C=C3)F","ic50_nm":50,"target":"EGFR","status":"Investigational"},
    {"name":"Candidate_C_efficient","smiles":"NC1=NC=NC2=CC=CC=C12","ic50_nm":1000,"target":"EGFR","status":"Investigational"},
    {"name":"Candidate_D_greasy","smiles":"CCCCCCC1=CC2=NC=NC(=C2C=C1)NC3=CC=C(C=C3)CCCCCC","ic50_nm":2,"target":"EGFR","status":"Investigational"},
    {"name":"Candidate_E_polar","smiles":"OC1=CC2=NC=NC(=C2C=C1O)NC3=CC=C(C=C3)O","ic50_nm":500,"target":"EGFR","status":"Investigational"},
    {"name":"Candidate_F_clinical","smiles":"COC1=CC2=NC=NC(=C2C=C1OCCO)NC3=CC(=C(C=C3)F)Cl","ic50_nm":8,"target":"EGFR","status":"Investigational"},
]


# ============================================================================
# MPO SCORING FUNCTIONS
# ============================================================================

def cns_mpo_score(mol):
    """
    Pfizer CNS MPO score (Wager et al., 2010).
    Score 0-6, where each parameter contributes 0-1.
    Designed for compounds that need to cross the blood-brain barrier.
    """
    if mol is None: return None, None
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    
    # LogD is approximately LogP for neutral compounds (we'll use LogP)
    logd = logp - 0.5  # rough approximation
    
    # Each parameter scored 0 (worst) to 1 (best) using piecewise linear functions
    
    # MW: optimal ≤ 360, acceptable up to 500
    if mw <= 360: s_mw = 1.0
    elif mw <= 500: s_mw = 1.0 - (mw - 360) / 140
    else: s_mw = 0.0
    
    # LogP: optimal ≤ 3, acceptable up to 5
    if logp <= 3: s_logp = 1.0
    elif logp <= 5: s_logp = 1.0 - (logp - 3) / 2
    else: s_logp = 0.0
    
    # LogD: optimal ≤ 2, acceptable up to 4
    if logd <= 2: s_logd = 1.0
    elif logd <= 4: s_logd = 1.0 - (logd - 2) / 2
    else: s_logd = 0.0
    
    # TPSA: optimal between 40-90, falls off outside
    if 40 <= tpsa <= 90: s_tpsa = 1.0
    elif tpsa < 40: s_tpsa = max(0, tpsa / 40)
    elif tpsa <= 120: s_tpsa = 1.0 - (tpsa - 90) / 30
    else: s_tpsa = 0.0
    
    # HBD: ≤0.5 ideal, ≤3.5 acceptable
    if hbd <= 0.5: s_hbd = 1.0
    elif hbd <= 3.5: s_hbd = 1.0 - (hbd - 0.5) / 3
    else: s_hbd = 0.0
    
    # pKa proxy: assume all are 7.5 (neutral) - simplified
    s_pka = 1.0
    
    total = s_mw + s_logp + s_logd + s_tpsa + s_hbd + s_pka
    components = {"MW":s_mw, "LogP":s_logp, "LogD":s_logd, "TPSA":s_tpsa, "HBD":s_hbd, "pKa":s_pka}
    return round(total, 2), components

def ligand_efficiency(ic50_nm, n_heavy_atoms):
    """
    LE = -RT × ln(IC50) / N_heavy_atoms
    At 298K, RT = 0.5961 kcal/mol
    Convert IC50 from nM to M: divide by 1e9
    pIC50 = -log10(IC50 in M)
    LE in kcal/mol/heavy atom; >0.3 is good
    """
    if ic50_nm <= 0 or n_heavy_atoms == 0: return 0
    pic50 = -np.log10(ic50_nm * 1e-9)
    delta_g = 1.37 * pic50  # kcal/mol (free energy of binding)
    return round(delta_g / n_heavy_atoms, 3)

def lipophilic_efficiency(ic50_nm, logp):
    """
    LipE = pIC50 - LogP
    Tells you if potency is "real" (good binding) or "fake" (just greasy)
    LipE > 5 is excellent; > 3 is good; < 3 is concerning
    """
    if ic50_nm <= 0: return 0
    pic50 = -np.log10(ic50_nm * 1e-9)
    return round(pic50 - logp, 2)


# ============================================================================
# COMPUTE ALL METRICS
# ============================================================================
print(f"\n📦 Analyzing {len(candidates)} drug candidates...")

results = []
for c in candidates:
    mol = Chem.MolFromSmiles(c["smiles"])
    if mol is None: continue
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    fsp3 = Descriptors.FractionCSP3(mol)
    
    cns_score, cns_components = cns_mpo_score(mol)
    le = ligand_efficiency(c["ic50_nm"], heavy)
    lipe = lipophilic_efficiency(c["ic50_nm"], logp)
    pic50 = round(-np.log10(c["ic50_nm"] * 1e-9), 2)
    
    # Overall optimization rank: weighted combination
    # Normalize each metric: CNS_MPO/6, LE/0.5, LipE/8 (rough max values)
    overall = (cns_score/6 * 0.3 + min(le, 0.5)/0.5 * 0.4 + min(max(lipe,0), 8)/8 * 0.3)
    
    results.append({
        "Name": c["name"], "SMILES": c["smiles"], "Target": c["target"], "Status": c["status"],
        "IC50_nM": c["ic50_nm"], "pIC50": pic50,
        "MW": round(mw,1), "LogP": round(logp,2), "TPSA": round(tpsa,1),
        "HBD": hbd, "HBA": hba, "RotBonds": rot, "HeavyAtoms": heavy, "Fsp3": round(fsp3,3),
        "CNS_MPO": cns_score, "LE": le, "LipE": lipe,
        "Overall_Score": round(overall, 3),
        **{f"CNS_{k}": v for k,v in cns_components.items()},
    })

df = pd.DataFrame(results).sort_values("Overall_Score", ascending=False)
print(f"   ✅ {len(df)} candidates scored")


# ============================================================================
# RESULTS
# ============================================================================
print(f"\n{'='*90}")
print(f"  RANKED CANDIDATES (best to worst, by overall MPO score)")
print(f"{'='*90}")
print(f"\n  {'Rank':<5} {'Name':<22} {'IC50':>8} {'CNS_MPO':>8} {'LE':>6} {'LipE':>6} {'Overall':>8}")
print(f"  {'-'*70}")
for i, (_, r) in enumerate(df.iterrows(), 1):
    cns_bar = "★" * int(r['CNS_MPO']) + "☆" * (6 - int(r['CNS_MPO']))
    print(f"  #{i:<4} {r['Name']:<22} {r['IC50_nM']:>6}nM {r['CNS_MPO']:>6.2f}/6 {r['LE']:>6.3f} {r['LipE']:>6.2f} {r['Overall_Score']:>8.3f}")

# Best in each category
print(f"\n{'='*90}\n  CATEGORY WINNERS\n{'='*90}")
print(f"\n  🏆 Best CNS MPO (brain penetrant): {df.iloc[df['CNS_MPO'].idxmax()]['Name']} (CNS_MPO = {df['CNS_MPO'].max():.2f}/6)")
print(f"  🏆 Best Ligand Efficiency:         {df.iloc[df['LE'].idxmax()]['Name']} (LE = {df['LE'].max():.3f})")
print(f"  🏆 Best Lipophilic Efficiency:     {df.iloc[df['LipE'].idxmax()]['Name']} (LipE = {df['LipE'].max():.2f})")
print(f"  🏆 Best Overall:                   {df.iloc[0]['Name']} (Overall = {df.iloc[0]['Overall_Score']:.3f})")


# ============================================================================
# EFFICIENCY ANALYSIS
# ============================================================================
print(f"\n{'='*90}\n  EFFICIENCY INSIGHTS\n{'='*90}")
print(f"""
LIGAND EFFICIENCY (LE) — measures activity per atom
   > 0.30 = good, every atom is contributing
   < 0.20 = bloated molecule, atoms aren't earning their keep

LIPOPHILIC EFFICIENCY (LipE) — measures real potency vs. just being greasy
   > 5 = excellent, real binding interactions
   3-5 = acceptable
   < 3 = potency may be coming from non-specific lipophilic interactions

In our dataset:
   {(df['LE'] > 0.30).sum()} compounds have LE > 0.30 (efficient binding)
   {(df['LipE'] > 5).sum()} compounds have LipE > 5 (high-quality potency)
   {(df['CNS_MPO'] >= 4).sum()} compounds have CNS_MPO ≥ 4 (likely brain-penetrant)
""")


# ============================================================================
# VISUALIZATIONS
# ============================================================================
print(f"📊 Generating visualizations...")
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 22)); gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.3)
fig.suptitle(f"Multi-Parameter Optimization Dashboard — {len(df)} Drug Candidates\nUniversity of Waterloo | Medicinal Chemistry",
             fontsize=17, fontweight='bold', y=0.99)

status_colors = {"Approved":"#27ae60","Investigational":"#9b59b6"}

# Panel 1: Overall ranking bar chart
ax = fig.add_subplot(gs[0, :])
top20 = df.head(20)
colors = [status_colors[s] for s in top20['Status']]
bars = ax.barh(range(len(top20)), top20['Overall_Score'][::-1], color=colors[::-1],
               edgecolor='white', linewidth=1)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20['Name'][::-1], fontsize=8)
ax.set_xlabel("Overall MPO Score (combined CNS_MPO, LE, LipE)")
ax.set_title("Drug candidates ranked by overall optimization score", fontweight='bold', fontsize=13)
for bar, val in zip(bars, top20['Overall_Score'][::-1]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
           va='center', fontsize=7, fontweight='bold')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=c, label=s) for s,c in status_colors.items()], fontsize=9)

# Panel 2: CNS MPO breakdown stacked bar
ax = fig.add_subplot(gs[1, 0])
cns_components = ["MW", "LogP", "LogD", "TPSA", "HBD", "pKa"]
bottom = np.zeros(len(df))
component_colors = plt.cm.viridis(np.linspace(0.2, 0.9, 6))
for i, comp in enumerate(cns_components):
    vals = df[f'CNS_{comp}'].values
    ax.bar(range(len(df)), vals, bottom=bottom, label=comp, color=component_colors[i],
           edgecolor='white', linewidth=0.3)
    bottom += vals
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df['Name'], rotation=90, fontsize=6)
ax.set_ylabel("CNS MPO Score (max 6)")
ax.set_title("CNS MPO score breakdown by component", fontweight='bold')
ax.legend(fontsize=7, loc='upper right')
ax.set_ylim(0, 6.5)
ax.axhline(4, color='red', ls='--', alpha=0.4, label='Acceptable (4)')

# Panel 3: LE vs LipE scatter (the "efficiency map")
ax = fig.add_subplot(gs[1, 1])
for status in df['Status'].unique():
    s = df[df['Status']==status]
    ax.scatter(s['LE'], s['LipE'], c=status_colors[status], s=120, alpha=0.8,
              edgecolors='white', linewidth=1.5, label=status, zorder=5)
    for _, r in s.iterrows():
        ax.annotate(r['Name'][:10], (r['LE'], r['LipE']), fontsize=6, alpha=0.7,
                   ha='center', va='bottom')
ax.axvline(0.30, color='green', ls='--', alpha=0.4, label='LE=0.30 (good)')
ax.axhline(5, color='blue', ls='--', alpha=0.4, label='LipE=5 (good)')
ax.add_patch(Rectangle((0.30, 5), 0.5, 5, alpha=0.06, color='green'))
ax.text(0.4, 7, 'IDEAL\nZONE', ha='center', fontsize=10, fontweight='bold', color='green')
ax.set_xlabel("Ligand Efficiency (LE)")
ax.set_ylabel("Lipophilic Efficiency (LipE)")
ax.set_title("Efficiency map: LE vs LipE", fontweight='bold')
ax.legend(fontsize=8)

# Panel 4: pIC50 vs LogP (potency-lipophilicity plot)
ax = fig.add_subplot(gs[2, 0])
for status in df['Status'].unique():
    s = df[df['Status']==status]
    ax.scatter(s['LogP'], s['pIC50'], c=status_colors[status], s=120, alpha=0.8,
              edgecolors='white', linewidth=1.5, label=status, zorder=5)
# Draw LipE iso-lines
for lipe_val in [3, 5, 7]:
    x = np.linspace(0, 8, 100)
    y = x + lipe_val
    ax.plot(x, y, '--', alpha=0.3, color='gray')
    ax.text(7.5, 7.5+lipe_val, f'LipE={lipe_val}', fontsize=7, alpha=0.5)
ax.set_xlabel("LogP"); ax.set_ylabel("pIC50")
ax.set_title("Potency-lipophilicity plot (with LipE iso-lines)", fontweight='bold')
ax.legend(fontsize=9)

# Panel 5: Radar chart for top 5 candidates
ax = fig.add_subplot(gs[2, 1], polar=True)
top5 = df.head(5)
metrics = ["CNS_MPO", "LE", "LipE", "pIC50"]
norms = {"CNS_MPO": (0, 6), "LE": (0, 0.5), "LipE": (-2, 8), "pIC50": (4, 10)}
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist() + [0]
colors_radar = plt.cm.tab10(np.linspace(0, 1, 5))
for i, (_, r) in enumerate(top5.iterrows()):
    vals = [max(0, min(1, (r[m] - norms[m][0])/(norms[m][1]-norms[m][0]))) for m in metrics]
    vals.append(vals[0])
    ax.plot(angles, vals, 'o-', linewidth=2, color=colors_radar[i], label=r['Name'][:12], markersize=4)
    ax.fill(angles, vals, alpha=0.1, color=colors_radar[i])
ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics, fontsize=9)
ax.set_title("Top 5 candidates radar profile", fontweight='bold', y=1.1)
ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.4, 1.1))

# Panel 6: Heatmap of all metrics
ax = fig.add_subplot(gs[3, :])
metrics_hm = ["MW", "LogP", "TPSA", "HBD", "CNS_MPO", "LE", "LipE", "pIC50", "Overall_Score"]
hm_data = df[metrics_hm].copy()
# Normalize each column to 0-1
for col in metrics_hm:
    mn, mx = hm_data[col].min(), hm_data[col].max()
    if mx != mn:
        hm_data[col] = (hm_data[col] - mn) / (mx - mn)
im = ax.imshow(hm_data.values.T, cmap='RdYlGn', aspect='auto')
ax.set_xticks(range(len(df))); ax.set_xticklabels(df['Name'], rotation=90, fontsize=7)
ax.set_yticks(range(len(metrics_hm))); ax.set_yticklabels(metrics_hm, fontsize=9)
plt.colorbar(im, ax=ax, label='Normalized score (0-1)', shrink=0.6)
ax.set_title("All metrics heatmap (green=better, red=worse)", fontweight='bold')

plt.savefig("project9_mpo_dashboard.png", dpi=150, bbox_inches='tight')
print(f"   ✅ project9_mpo_dashboard.png saved")
df.to_csv("project9_mpo_results.csv", index=False)
print(f"   ✅ project9_mpo_results.csv saved")

elapsed = time.time() - start_time
print(f"\n⏱️  {elapsed:.1f}s\n{'='*70}\n  PIPELINE COMPLETE\n{'='*70}")
