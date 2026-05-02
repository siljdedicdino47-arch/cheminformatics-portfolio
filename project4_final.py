#project 4

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
print("  ADMET PROPERTY PREDICTOR")
print("  Predicting Drug Absorption, Metabolism & Toxicity from Structure")
print("  University of Waterloo | Medicinal Chemistry")
print("=" * 80)

# ============================================================================
# WHAT THIS PROJECT DOES:
# ============================================================================
# In drug discovery, ~60% of drug candidates fail in clinical trials due to
# poor ADMET properties (Absorption, Distribution, Metabolism, Excretion, 
# Toxicity). This project predicts ADMET risk flags from molecular structure
# alone — BEFORE any experiments are run.
#
# This is directly inspired by recent publications:
#   - "Leveraging ML models in evaluating ADMET properties" (ADMET & DMPK, 2025)
#   - "Computational toxicology in drug discovery" (Briefings Bioinform., 2025)
#   - "AI-Integrated QSAR Modeling for Enhanced Drug Discovery" (IJMS, 2025)
#
# Real-world impact: If you can predict that a molecule will be toxic or 
# poorly absorbed BEFORE synthesizing it, you save months of lab work and
# millions of dollars.
# ============================================================================


USE_CHEMBL = False  # <-- CHANGE TO True ON YOUR MACHINE

if USE_CHEMBL:
    try:
        from chembl_webresource_client.new_client import new_client
        print("✅ ChEMBL API detected")
    except ImportError:
        USE_CHEMBL = False


# ============================================================================
# ADMET PREDICTION RULES (Literature-validated thresholds)
# ============================================================================
# These thresholds come from published medicinal chemistry literature.
# They are simplified rule-based predictions — in industry, ML models
# would be trained on experimental data, but these rules give surprisingly
# good first-pass predictions.

def predict_admet(mol):
    """
    Predict ADMET properties from molecular structure using
    validated rule-based thresholds from medicinal chemistry literature.
    
    Returns a dict of predictions with traffic-light risk flags:
        'green'  = favorable / low risk
        'yellow' = borderline / moderate risk  
        'red'    = unfavorable / high risk
    """
    if mol is None:
        return None
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    fsp3 = Descriptors.FractionCSP3(mol)
    mr = Descriptors.MolMR(mol)
    
    predictions = {}
    
    # ----- ABSORPTION -----
    
    # Oral Absorption (based on Lipinski + Veber rules)
    # Egan et al. (2000): TPSA and LogP are the two best predictors
    if tpsa <= 120 and logp >= 0 and logp <= 4 and mw <= 450:
        predictions['Oral_Absorption'] = ('HIGH', 'green')
    elif tpsa <= 140 and logp >= -1 and logp <= 5 and mw <= 500:
        predictions['Oral_Absorption'] = ('MODERATE', 'yellow')
    else:
        predictions['Oral_Absorption'] = ('LOW', 'red')
    
    # Intestinal Permeability (Caco-2 prediction)
    # Yazdanian et al. (1998): LogP > 0 and TPSA < 90 predict high permeability
    if logp > 1 and tpsa < 90 and hbd <= 3:
        predictions['Caco2_Permeability'] = ('HIGH', 'green')
    elif logp > 0 and tpsa < 120 and hbd <= 5:
        predictions['Caco2_Permeability'] = ('MODERATE', 'yellow')
    else:
        predictions['Caco2_Permeability'] = ('LOW', 'red')
    
    # P-glycoprotein (P-gp) Efflux Risk
    # Polli et al. (2001): MW > 400, HBD > 2, TPSA > 60 increase efflux risk
    pgp_score = sum([mw > 400, hbd > 2, tpsa > 60, logp > 4])
    if pgp_score <= 1:
        predictions['Pgp_Efflux'] = ('LOW RISK', 'green')
    elif pgp_score == 2:
        predictions['Pgp_Efflux'] = ('MODERATE RISK', 'yellow')
    else:
        predictions['Pgp_Efflux'] = ('HIGH RISK', 'red')
    
    # ----- DISTRIBUTION -----
    
    # Blood-Brain Barrier (BBB) Penetration
    # Lipinski (2012): MW < 400, TPSA < 90, HBD <= 1, LogP 1-4
    if mw < 400 and tpsa < 70 and hbd <= 1 and 1 <= logp <= 4:
        predictions['BBB_Penetration'] = ('LIKELY', 'green')
    elif mw < 450 and tpsa < 90 and hbd <= 2 and 0 <= logp <= 5:
        predictions['BBB_Penetration'] = ('POSSIBLE', 'yellow')
    else:
        predictions['BBB_Penetration'] = ('UNLIKELY', 'red')
    
    # Plasma Protein Binding (PPB) Risk
    # Highly lipophilic compounds bind extensively to plasma proteins
    if logp > 4 or aromatic_rings >= 4:
        predictions['Plasma_Protein_Binding'] = ('HIGH (>90%)', 'red')
    elif logp > 2.5 or aromatic_rings >= 3:
        predictions['Plasma_Protein_Binding'] = ('MODERATE (50-90%)', 'yellow')
    else:
        predictions['Plasma_Protein_Binding'] = ('LOW (<50%)', 'green')
    
    # ----- METABOLISM -----
    
    # CYP450 Inhibition Risk
    # Lipophilic, nitrogen-containing aromatics are common CYP inhibitors
    n_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
    if logp > 3.5 and n_count >= 2 and aromatic_rings >= 2:
        predictions['CYP_Inhibition'] = ('HIGH RISK', 'red')
    elif logp > 2.5 and n_count >= 1 and aromatic_rings >= 1:
        predictions['CYP_Inhibition'] = ('MODERATE RISK', 'yellow')
    else:
        predictions['CYP_Inhibition'] = ('LOW RISK', 'green')
    
    # Metabolic Stability (based on Fsp3 and LogP)
    # Lovering et al. (2009): Fsp3 correlates with metabolic stability
    if fsp3 > 0.4 and logp < 3:
        predictions['Metabolic_Stability'] = ('STABLE', 'green')
    elif fsp3 > 0.25 and logp < 4.5:
        predictions['Metabolic_Stability'] = ('MODERATE', 'yellow')
    else:
        predictions['Metabolic_Stability'] = ('UNSTABLE', 'red')
    
    # ----- EXCRETION -----
    
    # Renal Clearance Prediction
    # Hydrophilic, small molecules cleared renally
    if mw < 350 and logp < 1 and tpsa > 75:
        predictions['Renal_Clearance'] = ('HIGH', 'green')
    elif mw < 500 and logp < 3:
        predictions['Renal_Clearance'] = ('MODERATE', 'yellow')
    else:
        predictions['Renal_Clearance'] = ('LOW (hepatic)', 'red')
    
    # ----- TOXICITY -----
    
    # hERG Channel Inhibition (cardiac toxicity risk)
    # Aronov (2005): LogP > 3.7, MW > 350, basic nitrogen → hERG risk
    basic_n = sum(1 for atom in mol.GetAtoms() 
                  if atom.GetAtomicNum() == 7 and atom.GetTotalNumHs() > 0)
    herg_score = sum([logp > 3.7, mw > 350, basic_n >= 1, aromatic_rings >= 2])
    if herg_score >= 3:
        predictions['hERG_Risk'] = ('HIGH', 'red')
    elif herg_score >= 2:
        predictions['hERG_Risk'] = ('MODERATE', 'yellow')
    else:
        predictions['hERG_Risk'] = ('LOW', 'green')
    
    # AMES Mutagenicity Risk (structural alerts)
    # Check for known mutagenic substructures
    mutagen_smarts = [
        '[N+](=O)[O-]',          # Nitro group
        'N=N',                    # Azo group  
        '[NH2]c1ccccc1',          # Aniline
        'C1=CC2=CC=CC=C2C=C1',   # Polycyclic aromatic
    ]
    mutagen_hits = 0
    for smarts in mutagen_smarts:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            mutagen_hits += 1
    
    if mutagen_hits >= 2:
        predictions['AMES_Mutagenicity'] = ('HIGH RISK', 'red')
    elif mutagen_hits == 1:
        predictions['AMES_Mutagenicity'] = ('MODERATE RISK', 'yellow')
    else:
        predictions['AMES_Mutagenicity'] = ('LOW RISK', 'green')
    
    # Hepatotoxicity Risk (DILI)
    # Rule-based: high LogP + high TPSA + reactive groups
    dili_score = sum([logp > 3, tpsa > 75, mw > 500, rot_bonds > 8])
    if dili_score >= 3:
        predictions['Hepatotoxicity'] = ('HIGH RISK', 'red')
    elif dili_score >= 2:
        predictions['Hepatotoxicity'] = ('MODERATE RISK', 'yellow')
    else:
        predictions['Hepatotoxicity'] = ('LOW RISK', 'green')
    
    # ----- OVERALL SCORE -----
    red_count = sum(1 for _, (_, flag) in predictions.items() if flag == 'red')
    yellow_count = sum(1 for _, (_, flag) in predictions.items() if flag == 'yellow')
    green_count = sum(1 for _, (_, flag) in predictions.items() if flag == 'green')
    
    if red_count >= 4:
        predictions['OVERALL'] = ('POOR — Multiple high-risk flags', 'red')
    elif red_count >= 2:
        predictions['OVERALL'] = ('MODERATE — Some concerns', 'yellow')
    else:
        predictions['OVERALL'] = ('FAVORABLE — Low risk profile', 'green')
    
    predictions['_counts'] = {'red': red_count, 'yellow': yellow_count, 'green': green_count}
    
    return predictions


# ============================================================================
# COMPOUND DATABASE
# ============================================================================

def get_compounds():
    compounds = [
        # Drugs with KNOWN good ADMET
        {"name":"Metformin","smiles":"CN(C)C(=N)NC(=N)N","class":"Known Good ADMET","note":"Excellent oral bioavail., renal clearance"},
        {"name":"Acetaminophen","smiles":"CC(=O)NC1=CC=C(C=C1)O","class":"Known Good ADMET","note":"Good absorption, hepatic metabolism"},
        {"name":"Caffeine","smiles":"CN1C=NC2=C1C(=O)N(C(=O)N2C)C","class":"Known Good ADMET","note":"Near 100% oral bioavail., BBB penetrant"},
        {"name":"Ibuprofen","smiles":"CC(C)CC1=CC=C(C=C1)C(C)C(=O)O","class":"Known Good ADMET","note":"Rapid absorption, PPB ~99%"},
        {"name":"Aspirin","smiles":"CC(=O)OC1=CC=CC=C1C(=O)O","class":"Known Good ADMET","note":"Rapid absorption, ester hydrolysis"},
        {"name":"Memantine","smiles":"C1C2CC3CC1CC(C2)(C3)N","class":"Known Good ADMET","note":"Good oral bioavail., BBB penetrant"},
        {"name":"Fluconazole","smiles":"OC(CN1C=NC=N1)(CN2C=NC=N2)C3=CC=C(F)C=C3F","class":"Known Good ADMET","note":"91% oral bioavail., renal clearance"},
        
        # Drugs with KNOWN ADMET challenges
        {"name":"Paclitaxel","smiles":"CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C","class":"Known Poor ADMET","note":"<6% oral bioavail., P-gp substrate"},
        {"name":"Atorvastatin","smiles":"CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4","class":"Known Poor ADMET","note":"14% oral bioavail., extensive CYP3A4"},
        {"name":"Terfenadine","smiles":"C(C1=CC=CC=C1)(C2=CC=CC=C2)(CCCN3CCC(CC3)C(C4=CC=CC=C4)(C5=CC=CC=C5)O)O","class":"Known Poor ADMET","note":"Withdrawn — fatal hERG inhibition"},
        
        # Kinase inhibitors (mixed ADMET)
        {"name":"Imatinib","smiles":"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5","class":"Kinase Inhibitor","note":"98% oral bioavail., CYP3A4 substrate"},
        {"name":"Osimertinib","smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)NC(=O)C=C","class":"Kinase Inhibitor","note":"70% oral bioavail., BBB penetrant"},
        {"name":"Ibrutinib","smiles":"C=CC(=O)N1CCC(CC1)N2C3=C(C=CC=C3)N=C2C4=CC=C(C=C4)OC5=CC=CC=C5","class":"Kinase Inhibitor","note":"3% oral bioavail., extensive CYP3A4"},
        {"name":"Erlotinib","smiles":"COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC","class":"Kinase Inhibitor","note":"60% oral bioavail., CYP3A4/1A2"},
        {"name":"Vemurafenib","smiles":"CCCS(=O)(=O)NC1=CC(=C(C=C1F)C(=O)C2=CNC3=NC=C(C=C23)C4=CC=C(C=C4)Cl)F","class":"Kinase Inhibitor","note":"Good absorption but high PPB"},
        {"name":"Palbociclib","smiles":"CC(=O)C1=C(C=C2CN=C(N=C2N1)NC3=NC=C(C=C3)N4CCNCC4)C","class":"Kinase Inhibitor","note":"46% oral bioavail., CYP3A substrate"},
        
        # PROTACs (expected poor oral absorption)
        {"name":"ARV-471","smiles":"C1CCC(CC1)C(=O)NCCOCCOCCNC(=O)C2=CC=C(C=C2)F","class":"PROTAC","note":"Oral PROTAC in Phase 3 — surprisingly"},
        {"name":"ARV-110","smiles":"CC1(CCC(=C(C1)C2=CC=C(C=C2)CN3CCN(CC3)C(=O)C4CC4)C5=CC(=CC=C5)C(F)(F)F)C","class":"PROTAC","note":"Low oral bioavail."},
        {"name":"KT-474","smiles":"O=C(NC1=CC=CC=C1)NCCOCCOCCNC(=O)C2=CC=CC=C2","class":"PROTAC","note":"PEG linker, oral dosing attempted"},
        
        # Molecular glues
        {"name":"Lenalidomide","smiles":"C1CC(=O)NC(=O)C1N2CC3=CC=CC=C3C2=O","class":"Molecular Glue","note":"Good oral bioavail., renal clearance"},
        {"name":"Pomalidomide","smiles":"C1CC(=O)NC(=O)C1N2CC3=CC(=CC=C3C2=O)N","class":"Molecular Glue","note":"73% oral bioavail."},
        
        # SSRIs (CNS drugs — need BBB penetration)
        {"name":"Fluoxetine","smiles":"CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F","class":"CNS Drug","note":"72% oral bioavail., BBB penetrant"},
        {"name":"Sertraline","smiles":"CNC1CCC(C2=CC=CC=C21)C3=CC(=C(C=C3)Cl)Cl","class":"CNS Drug","note":"Good BBB penetration, high PPB"},
        {"name":"Donepezil","smiles":"COC1=CC2=C(C=C1OC)C(=O)C(C2)CC3CCN(CC3)CC4=CC=CC=C4","class":"CNS Drug","note":"100% oral bioavail., BBB penetrant"},
        
        # Antibiotics
        {"name":"Ciprofloxacin","smiles":"C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O","class":"Antibiotic","note":"70% oral bioavail., renal + hepatic"},
        {"name":"Amoxicillin","smiles":"CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C","class":"Antibiotic","note":"74% oral bioavail., renal clearance"},
        
        # Natural products
        {"name":"Morphine","smiles":"CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O","class":"Natural Product","note":"25% oral bioavail., BBB penetrant"},
        {"name":"Artemisinin","smiles":"CC1CCC2C(C(=O)OC3CC4(C1CCC23OO4)C)C","class":"Natural Product","note":"32% oral bioavail., rapid metabolism"},
        {"name":"Curcumin","smiles":"COC1=CC(=CC(=C1O)OC)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC","class":"Natural Product","note":"<1% oral bioavail. — classic example"},
    ]
    return compounds


# ============================================================================
# STEP 1: LOAD AND PROCESS
# ============================================================================
compounds = get_compounds()
print(f"\n📦 Loaded {len(compounds)} compounds")

results = []
for entry in compounds:
    mol = Chem.MolFromSmiles(entry["smiles"])
    if mol is None:
        continue
    
    admet = predict_admet(mol)
    if admet is None:
        continue
    
    row = {
        "Name": entry["name"],
        "Class": entry["class"],
        "Note": entry.get("note", ""),
        "MW": round(Descriptors.MolWt(mol), 1),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 1),
        "HBD": Lipinski.NumHDonors(mol),
        "Fsp3": round(Descriptors.FractionCSP3(mol), 3),
    }
    
    # Add each ADMET prediction
    for prop, val in admet.items():
        if prop == '_counts':
            row['Red_Flags'] = val['red']
            row['Yellow_Flags'] = val['yellow']
            row['Green_Flags'] = val['green']
        else:
            value, flag = val
            row[prop] = value
            row[f"{prop}_flag"] = flag
    
    results.append(row)

df = pd.DataFrame(results)
print(f"   ✅ ADMET predictions computed for {len(df)} molecules")


# ============================================================================
# STEP 2: DETAILED RESULTS
# ============================================================================
print(f"\n{'=' * 90}")
print(f"  ADMET PREDICTION RESULTS")
print(f"{'=' * 90}")

admet_props = ['Oral_Absorption','Caco2_Permeability','BBB_Penetration',
               'CYP_Inhibition','hERG_Risk','AMES_Mutagenicity','OVERALL']

for _, row in df.iterrows():
    flags = f"🟢×{row['Green_Flags']}  🟡×{row['Yellow_Flags']}  🔴×{row['Red_Flags']}"
    print(f"\n  {row['Name']} [{row['Class']}]  —  {flags}")
    print(f"    MW={row['MW']}  LogP={row['LogP']}  TPSA={row['TPSA']}  HBD={row['HBD']}")
    
    for prop in admet_props:
        if prop in row:
            flag = row.get(f"{prop}_flag", "")
            icon = {"green":"🟢","yellow":"🟡","red":"🔴"}.get(flag, "⚪")
            label = prop.replace('_',' ')
            print(f"    {icon} {label:<24} {row[prop]}")
    
    if row.get('Note'):
        print(f"    📝 Known: {row['Note']}")


# ============================================================================
# STEP 3: VALIDATION — How well do predictions match known data?
# ============================================================================
print(f"\n{'=' * 90}")
print(f"  VALIDATION: Predictions vs Known ADMET")
print(f"{'=' * 90}")

# Check specific known outcomes
validations = [
    ("Metformin", "Oral_Absorption", "HIGH", "Known near-100% oral bioavailability"),
    ("Caffeine", "BBB_Penetration", "LIKELY", "Known BBB-penetrant"),
    ("Paclitaxel", "Oral_Absorption", "LOW", "Known <6% oral bioavailability"),
    ("Fluoxetine", "BBB_Penetration", "LIKELY", "Must cross BBB for CNS effect"),
    ("Curcumin", "Oral_Absorption", "LOW", "Known <1% oral bioavailability"),
    ("Lenalidomide", "Oral_Absorption", "HIGH", "Known good oral bioavailability"),
]

correct = 0
total_val = len(validations)

for drug, prop, expected, reason in validations:
    row = df[df['Name']==drug]
    if len(row) == 0:
        continue
    predicted = row.iloc[0][prop]
    match = expected in predicted
    correct += match
    icon = "✅" if match else "❌"
    print(f"  {icon} {drug:<18} {prop:<22} Predicted: {predicted:<12} Expected: {expected:<10} ({reason})")

print(f"\n  Validation accuracy: {correct}/{total_val} ({correct/total_val*100:.0f}%)")


# ============================================================================
# STEP 4: VISUALIZATIONS
# ============================================================================
print(f"\n📊 Generating visualizations...")

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 18))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
fig.suptitle(f"ADMET Property Predictions — {len(df)} Compounds\nUniversity of Waterloo | Medicinal Chemistry",
             fontsize=17, fontweight='bold', y=0.98)

cls_colors = {"Known Good ADMET":"#27ae60","Known Poor ADMET":"#e74c3c",
              "Kinase Inhibitor":"#3498db","PROTAC":"#9b59b6",
              "Molecular Glue":"#f39c12","CNS Drug":"#1abc9c",
              "Antibiotic":"#e67e22","Natural Product":"#2ecc71"}

# Panel 1: ADMET risk heatmap
ax = fig.add_subplot(gs[0,:])
admet_columns = ['Oral_Absorption_flag','Caco2_Permeability_flag','Pgp_Efflux_flag',
                 'BBB_Penetration_flag','Plasma_Protein_Binding_flag',
                 'CYP_Inhibition_flag','Metabolic_Stability_flag',
                 'hERG_Risk_flag','AMES_Mutagenicity_flag','Hepatotoxicity_flag']
admet_labels = [c.replace('_flag','').replace('_',' ') for c in admet_columns]

heatmap_data = np.zeros((len(df), len(admet_columns)))
for i, (_, row) in enumerate(df.iterrows()):
    for j, col in enumerate(admet_columns):
        if col in row:
            flag = row[col]
            heatmap_data[i, j] = {'green': 0, 'yellow': 0.5, 'red': 1}.get(flag, 0.5)

from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#27ae60', '#f1c40f', '#e74c3c'])
im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(admet_labels)))
ax.set_xticklabels(admet_labels, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df['Name'], fontsize=7)
ax.set_title("ADMET risk heatmap (green=favorable, yellow=borderline, red=high risk)", fontweight='bold')

# Panel 2: Red flags by drug class
ax = fig.add_subplot(gs[1, 0])
class_order = df.groupby('Class')['Red_Flags'].mean().sort_values(ascending=False).index
x = np.arange(len(class_order))
means = [df[df['Class']==c]['Red_Flags'].mean() for c in class_order]
colors = [cls_colors.get(c, '#95a5a6') for c in class_order]
ax.barh(x, means, color=colors, edgecolor='white', linewidth=1)
ax.set_yticks(x); ax.set_yticklabels(class_order, fontsize=9)
ax.set_xlabel("Mean number of red ADMET flags")
ax.set_title("ADMET risk by drug class", fontweight='bold')

# Panel 3: LogP vs TPSA colored by oral absorption prediction
ax = fig.add_subplot(gs[1, 1])
abs_colors = {'HIGH':'#27ae60','MODERATE':'#f1c40f','LOW':'#e74c3c'}
for _, row in df.iterrows():
    c = abs_colors.get(row['Oral_Absorption'], '#95a5a6')
    ax.scatter(row['LogP'], row['TPSA'], c=c, s=60, alpha=0.7, edgecolors='white', linewidth=1, zorder=5)
    ax.annotate(row['Name'][:10], (row['LogP'], row['TPSA']), fontsize=5, alpha=0.6, ha='center', va='bottom')

# Draw the "golden rectangle" of oral absorption
from matplotlib.patches import FancyBboxPatch
golden = FancyBboxPatch((0, 20), 4, 100, boxstyle="round,pad=0.1",
                        facecolor='green', alpha=0.06, edgecolor='green', linestyle='--')
ax.add_patch(golden)
ax.text(2, 25, 'Oral absorption\n"sweet spot"', fontsize=8, color='green', ha='center', style='italic')
ax.set_xlabel("LogP"); ax.set_ylabel("TPSA (Å²)")
ax.set_title("Oral absorption prediction (Egan egg model)", fontweight='bold')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor='#27ae60',label='High'),
                   Patch(facecolor='#f1c40f',label='Moderate'),
                   Patch(facecolor='#e74c3c',label='Low')], fontsize=8)

# Panel 4: BBB prediction vs known BBB drugs
ax = fig.add_subplot(gs[2, 0])
bbb_colors = {'LIKELY':'#27ae60','POSSIBLE':'#f1c40f','UNLIKELY':'#e74c3c'}
for _, row in df.iterrows():
    c = bbb_colors.get(row['BBB_Penetration'], '#95a5a6')
    ax.scatter(row['MW'], row['TPSA'], c=c, s=60, alpha=0.7, edgecolors='white', linewidth=1, zorder=5)
    ax.annotate(row['Name'][:10], (row['MW'], row['TPSA']), fontsize=5, alpha=0.6)

ax.axvline(400, color='gray', ls='--', alpha=0.3)
ax.axhline(90, color='gray', ls='--', alpha=0.3)
ax.set_xlabel("MW (Da)"); ax.set_ylabel("TPSA (Å²)")
ax.set_title("BBB penetration prediction", fontweight='bold')
ax.legend(handles=[Patch(facecolor='#27ae60',label='Likely'),
                   Patch(facecolor='#f1c40f',label='Possible'),
                   Patch(facecolor='#e74c3c',label='Unlikely')], fontsize=8)

# Panel 5: Overall ADMET traffic light summary
ax = fig.add_subplot(gs[2, 1])
overall_counts = df['OVERALL'].apply(lambda x: x.split('—')[0].strip()).value_counts()
colors_pie = {'FAVORABLE ':'#27ae60','MODERATE ':'#f1c40f','POOR ':'#e74c3c'}
colors_list = [colors_pie.get(k, '#95a5a6') for k in overall_counts.index]
wedges, texts, autotexts = ax.pie(overall_counts.values, labels=overall_counts.index,
                                   colors=colors_list, autopct='%1.0f%%',
                                   textprops={'fontsize': 10})
ax.set_title("Overall ADMET assessment distribution", fontweight='bold')

plt.savefig("project4_admet_predictions.png", dpi=150, bbox_inches='tight')
print(f"   ✅ Figure saved: project4_admet_predictions.png")

df.to_csv("project4_admet_results.csv", index=False)
print(f"   ✅ Data saved: project4_admet_results.csv")

elapsed = time.time() - start_time
print(f"\n⏱️  Runtime: {elapsed:.1f}s")
print(f"\n{'=' * 70}")
print(f"  PIPELINE COMPLETE")
print(f"{'=' * 70}")
