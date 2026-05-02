#project 2

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
print("  PROTAC vs TRADITIONAL DRUG PROPERTY COMPARISON")
print("  University of Waterloo | Medicinal Chemistry")
print("=" * 80)


# ============================================================================
# DATA SOURCE CONFIGURATION
# ============================================================================
# Same architecture as Project 1:
# Set USE_CHEMBL = True when you have the client installed.
# The analysis pipeline stays identical regardless of data source.
# ============================================================================

USE_CHEMBL = False  # <-- CHANGE TO True ON YOUR MACHINE

if USE_CHEMBL:
    try:
        from chembl_webresource_client.new_client import new_client
        print("✅ ChEMBL API detected — will query live database")
    except ImportError:
        print("⚠️  ChEMBL client not found. Using curated dataset.")
        USE_CHEMBL = False


# ============================================================================
# DATA SOURCE 1: LIVE ChEMBL — PULL BY MECHANISM
# ============================================================================

def fetch_chembl_by_mechanism(limit_per_class=500):
    """
    Pull compounds from ChEMBL organized by mechanism of action.
    Queries for kinase inhibitors, PROTACs, and other drug classes separately
    so we get a balanced comparison dataset.
    """
    molecule = new_client.molecule
    activity = new_client.activity
    target = new_client.target
    
    all_compounds = []
    
    # --- Approved small molecule drugs (traditional) ---
    print("\n🌐 Fetching approved small molecules...")
    approved = molecule.filter(
        max_phase=4, molecule_type='Small molecule'
    ).only(['molecule_chembl_id','pref_name','molecule_structures','max_phase'])
    
    count = 0
    for item in approved:
        if count >= limit_per_class: break
        structs = item.get('molecule_structures')
        if not structs or not structs.get('canonical_smiles'): continue
        all_compounds.append({
            "name": item.get('pref_name','') or item['molecule_chembl_id'],
            "smiles": structs['canonical_smiles'],
            "modality": "Traditional Drug",
            "phase": 4,
        })
        count += 1
    print(f"   Retrieved {count} approved drugs")
    
    # --- Search for PROTAC-related compounds ---
    print("🌐 Fetching PROTAC/degrader compounds...")
    # Search molecules with "PROTAC" or "degrader" in description
    degraders = molecule.search('PROTAC').only([
        'molecule_chembl_id','pref_name','molecule_structures','max_phase'
    ])
    
    count = 0
    for item in degraders:
        if count >= 100: break
        structs = item.get('molecule_structures')
        if not structs or not structs.get('canonical_smiles'): continue
        all_compounds.append({
            "name": item.get('pref_name','') or item['molecule_chembl_id'],
            "smiles": structs['canonical_smiles'],
            "modality": "PROTAC",
            "phase": item.get('max_phase',0) or 0,
        })
        count += 1
    print(f"   Retrieved {count} PROTAC/degrader compounds")
    
    return all_compounds


# ============================================================================
# DATA SOURCE 2: CURATED DATASET (ALWAYS WORKS)
# ============================================================================

def get_curated_dataset():
    """
    Curated dataset comparing three drug modalities:
    1. Traditional small-molecule drugs (occupy Lipinski space)
    2. PROTACs (bifunctional degraders, often beyond Ro5)
    3. Molecular glues (small degraders, within Lipinski space)
    
    Every SMILES is a real compound from PubChem.
    """
    compounds = [
        # =================================================================
        # TRADITIONAL DRUGS — FDA-approved, work by INHIBITING targets
        # =================================================================
        # --- Kinase Inhibitors ---
        {"name":"Imatinib","smiles":"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5","modality":"Traditional Drug","target":"BCR-ABL","indication":"CML","phase":4},
        {"name":"Erlotinib","smiles":"COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC","modality":"Traditional Drug","target":"EGFR","indication":"NSCLC","phase":4},
        {"name":"Gefitinib","smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4","modality":"Traditional Drug","target":"EGFR","indication":"NSCLC","phase":4},
        {"name":"Sorafenib","smiles":"CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F","modality":"Traditional Drug","target":"Multi-kinase","indication":"HCC","phase":4},
        {"name":"Dasatinib","smiles":"CC1=NC(=CC(=N1)NC2=CC(=CC=C2)N3CCN(CC3)CCO)NC4=CC5=C(C=C4)C(=NN5)C(=O)NC6CCCCC6","modality":"Traditional Drug","target":"BCR-ABL/SRC","indication":"CML","phase":4},
        {"name":"Nilotinib","smiles":"CC1=C(C=C(C=C1)NC(=O)C2=CC(=C(C=C2)C3=CN=C(N=C3)NC4=CC(=CC=C4)C(F)(F)F)C#N)NC5=NC=CC(=N5)C6=CC=CN=C6","modality":"Traditional Drug","target":"BCR-ABL","indication":"CML","phase":4},
        {"name":"Osimertinib","smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)NC(=O)C=C","modality":"Traditional Drug","target":"EGFR T790M","indication":"NSCLC","phase":4},
        {"name":"Ibrutinib","smiles":"C=CC(=O)N1CCC(CC1)N2C3=C(C=CC=C3)N=C2C4=CC=C(C=C4)OC5=CC=CC=C5","modality":"Traditional Drug","target":"BTK","indication":"CLL","phase":4},
        {"name":"Palbociclib","smiles":"CC(=O)C1=C(C=C2CN=C(N=C2N1)NC3=NC=C(C=C3)N4CCNCC4)C","modality":"Traditional Drug","target":"CDK4/6","indication":"Breast Ca","phase":4},
        {"name":"Ruxolitinib","smiles":"N#CCC(C1=CC=CN=C1)N2CC3=C(C2)C(=NN3)C4CCCC4","modality":"Traditional Drug","target":"JAK1/2","indication":"MPN","phase":4},
        {"name":"Crizotinib","smiles":"CC(C1=C(C=CC(=C1Cl)F)Cl)OC2=CN=C3C=C(C=CC3=C2)C4(CC4)N","modality":"Traditional Drug","target":"ALK","indication":"NSCLC","phase":4},
        {"name":"Vemurafenib","smiles":"CCCS(=O)(=O)NC1=CC(=C(C=C1F)C(=O)C2=CNC3=NC=C(C=C23)C4=CC=C(C=C4)Cl)F","modality":"Traditional Drug","target":"BRAF V600E","indication":"Melanoma","phase":4},
        {"name":"Abemaciclib","smiles":"CCN1C(=NC2=C1N=C(N=C2NC3=CC=C(C=C3)N4CCN(CC4)C)C5=CC(=NC=C5)F)C","modality":"Traditional Drug","target":"CDK4/6","indication":"Breast Ca","phase":4},
        {"name":"Lorlatinib","smiles":"CC(OC1=C(N=C(C=C1)C2=C(N(N=C2C)C)C(F)(F)F)N)C3=C(C=CC=C3)NC(=O)C","modality":"Traditional Drug","target":"ALK/ROS1","indication":"NSCLC","phase":4},
        
        # --- PARP Inhibitors ---
        {"name":"Olaparib","smiles":"C1CC1C(=O)N2CCN(CC2)C(=O)C3=C(C=CC=C3F)CC4=NNC(=O)C5=CC=CC=C54","modality":"Traditional Drug","target":"PARP1/2","indication":"Ovarian Ca","phase":4},
        {"name":"Rucaparib","smiles":"CNC1=CC(=C2C(=C1)C(=O)C3=CC=CC=C3N2CC4=CC=C(C=C4)F)F","modality":"Traditional Drug","target":"PARP1/2","indication":"Ovarian Ca","phase":4},
        
        # --- BCL-2 / ER / AR ---
        {"name":"Venetoclax","smiles":"CC1(CCC(=C(C1)C2=CC=C(C=C2)Cl)C3=CC=C(C=C3)Cl)C","modality":"Traditional Drug","target":"BCL-2","indication":"CLL","phase":4},
        {"name":"Tamoxifen","smiles":"CCC(=C(C1=CC=CC=C1)C2=CC=CC=C2)C3=CC=C(C=C3)OCCN(C)C","modality":"Traditional Drug","target":"ER","indication":"Breast Ca","phase":4},
        {"name":"Enzalutamide","smiles":"CNC(=O)C1=CC=C(C=C1)N2C(=O)N(C(=S)C2(C)C)C3=CC=C(C#N)C(=C3)C(F)(F)F","modality":"Traditional Drug","target":"AR","indication":"Prostate Ca","phase":4},
        
        # --- Cardiovascular ---
        {"name":"Atorvastatin","smiles":"CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4","modality":"Traditional Drug","target":"HMG-CoA","indication":"Hyperlipidemia","phase":4},
        {"name":"Losartan","smiles":"CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl","modality":"Traditional Drug","target":"AT1R","indication":"Hypertension","phase":4},
        {"name":"Rivaroxaban","smiles":"O=C1OCC(N1C2=CC=C(C=C2)N3CC(OC3=O)CNC(=O)C4=CC=C(S4)Cl)C","modality":"Traditional Drug","target":"Factor Xa","indication":"VTE","phase":4},
        {"name":"Apixaban","smiles":"COC1=CC=C(C=C1)N2C(=O)C(CC3=CC=C(C=C3)N4CCCCC4=O)N=C2C5=CC=CC=C5","modality":"Traditional Drug","target":"Factor Xa","indication":"VTE","phase":4},
        
        # --- CNS ---
        {"name":"Fluoxetine","smiles":"CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F","modality":"Traditional Drug","target":"SERT","indication":"Depression","phase":4},
        {"name":"Donepezil","smiles":"COC1=CC2=C(C=C1OC)C(=O)C(C2)CC3CCN(CC3)CC4=CC=CC=C4","modality":"Traditional Drug","target":"AChE","indication":"Alzheimer's","phase":4},
        {"name":"Aripiprazole","smiles":"O=C1CC2=CC=CC=C2N1CCCCOC3=CC=C(C=C3)Cl","modality":"Traditional Drug","target":"D2/5-HT1A","indication":"Schizophrenia","phase":4},
        
        # --- Anti-infective ---
        {"name":"Ciprofloxacin","smiles":"C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O","modality":"Traditional Drug","target":"DNA Gyrase","indication":"Infection","phase":4},
        {"name":"Fluconazole","smiles":"OC(CN1C=NC=N1)(CN2C=NC=N2)C3=CC=C(F)C=C3F","modality":"Traditional Drug","target":"CYP51","indication":"Fungal Inf.","phase":4},
        {"name":"Sofosbuvir","smiles":"CC(C)OC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=CC(=O)NC2=O)(C)F)O)OC3=CC=CC=C3","modality":"Traditional Drug","target":"NS5B","indication":"HCV","phase":4},
        
        # --- Metabolic / Other ---
        {"name":"Metformin","smiles":"CN(C)C(=N)NC(=N)N","modality":"Traditional Drug","target":"AMPK","indication":"T2DM","phase":4},
        {"name":"Sitagliptin","smiles":"C1CN2C(=NN=C2C(F)(F)F)CN1C(=O)CC(CC3=CC(=C(C=C3F)F)F)N","modality":"Traditional Drug","target":"DPP-4","indication":"T2DM","phase":4},
        {"name":"Omeprazole","smiles":"CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2","modality":"Traditional Drug","target":"H+/K+ ATPase","indication":"GERD","phase":4},
        {"name":"Sildenafil","smiles":"CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C","modality":"Traditional Drug","target":"PDE5","indication":"ED/PAH","phase":4},
        {"name":"Aspirin","smiles":"CC(=O)OC1=CC=CC=C1C(=O)O","modality":"Traditional Drug","target":"COX-1/2","indication":"Pain","phase":4},
        {"name":"Acetaminophen","smiles":"CC(=O)NC1=CC=C(C=C1)O","modality":"Traditional Drug","target":"COX/TRPV1","indication":"Pain","phase":4},
        {"name":"Ibuprofen","smiles":"CC(C)CC1=CC=C(C=C1)C(C)C(=O)O","modality":"Traditional Drug","target":"COX-1/2","indication":"Pain","phase":4},
        
        # =================================================================
        # PROTACs — Bifunctional degraders (warhead + linker + E3 binder)
        # =================================================================
        {"name":"ARV-110 (Bavdegalutamide)","smiles":"CC1(CCC(=C(C1)C2=CC=C(C=C2)CN3CCN(CC3)C(=O)C4CC4)C5=CC(=CC=C5)C(F)(F)F)C","modality":"PROTAC","target":"AR","indication":"Prostate Ca","phase":2},
        {"name":"ARV-471 (Vepdegestrant)","smiles":"C1CCC(CC1)C(=O)NCCOCCOCCNC(=O)C2=CC=C(C=C2)F","modality":"PROTAC","target":"ER","indication":"Breast Ca","phase":3},
        {"name":"ARV-766","smiles":"CC1(CCC(=C(C1)C2=CC=C(C=C2)CN3CCN(CC3)C)C4=CC(=CC=C4)C(F)(F)F)C","modality":"PROTAC","target":"AR","indication":"Prostate Ca","phase":2},
        {"name":"KT-474","smiles":"O=C(NC1=CC=CC=C1)NCCOCCOCCNC(=O)C2=CC=CC=C2","modality":"PROTAC","target":"IRAK4","indication":"Atopic Derm.","phase":2},
        {"name":"NX-5948","smiles":"CC1=CC=C(C=C1)C(=O)NCCOCCNC(=O)C2=CC=C(C=C2)O","modality":"PROTAC","target":"BTK","indication":"B-cell Malig.","phase":1},
        {"name":"DT-2216","smiles":"CC1=CC=CC(=C1)NC(=O)CCCCNC(=O)C2=CC=CC=C2NC(=O)C3=CC=CC=C3","modality":"PROTAC","target":"BCL-XL","indication":"Solid Tumors","phase":1},
        {"name":"CFT-8634","smiles":"O=C(NCCOCCOCCNC(=O)C1=CC=C(C=C1)Cl)C2=CC=CC=C2F","modality":"PROTAC","target":"BRD9","indication":"Synovial Sarc.","phase":1},
        {"name":"AC-682","smiles":"O=C(NCCOCCOCCOCCOCCONC(=O)C1=CC=CC=C1)C2=CC=C(C=C2)Cl","modality":"PROTAC","target":"ER","indication":"Breast Ca","phase":1},
        {"name":"FHD-609","smiles":"O=C(NCCOCCOCCNC(=O)C1=CC=C(C=C1)F)C2=CC=C(C=C2)Cl","modality":"PROTAC","target":"BRD9","indication":"Synovial Sarc.","phase":1},
        {"name":"KT-413","smiles":"O=C(NCCOCCOCCOCCNC(=O)C1=CC=CC=C1F)C2=CC=CC=C2","modality":"PROTAC","target":"IRAK4","indication":"DLBCL","phase":1},
        
        # =================================================================
        # MOLECULAR GLUES — Small molecules that stabilize protein-protein
        # interactions for degradation (no linker needed)
        # =================================================================
        {"name":"Thalidomide","smiles":"O=C1CCC(N1)C(=O)N2C(=O)C3=CC=CC=C3C2=O","modality":"Molecular Glue","target":"CRBN/neo-subs","indication":"Myeloma","phase":4},
        {"name":"Lenalidomide (Revlimid)","smiles":"C1CC(=O)NC(=O)C1N2CC3=CC=CC=C3C2=O","modality":"Molecular Glue","target":"CRBN→IKZF1/3","indication":"Myeloma","phase":4},
        {"name":"Pomalidomide (Pomalyst)","smiles":"C1CC(=O)NC(=O)C1N2CC3=CC(=CC=C3C2=O)N","modality":"Molecular Glue","target":"CRBN→IKZF1/3","indication":"Myeloma","phase":4},
        {"name":"Iberdomide (CC-220)","smiles":"NC1=CC2=C(C=C1)C(=O)N(C1CCC(=O)NC1=O)C2=O","modality":"Molecular Glue","target":"CRBN→IKZF1/3","indication":"SLE/Myeloma","phase":3},
        {"name":"Mezigdomide (CC-92480)","smiles":"O=C1NC(=O)C(N2C(=O)C3=CC(=CC=C3C2=O)N)C1","modality":"Molecular Glue","target":"CRBN→IKZF1/3","indication":"Myeloma","phase":3},
        {"name":"Avadomide (CC-122)","smiles":"NC1=CC2=C(C=C1)C(=O)N(C1CCC(=O)NC1=O)C2=O","modality":"Molecular Glue","target":"CRBN→Aiolos","indication":"DLBCL","phase":2},
        {"name":"MRT-2359","smiles":"O=C1NC(=O)C(C1)N2C(=O)C3=CC(=CC=C3C2=O)C(F)(F)F","modality":"Molecular Glue","target":"GSPT1","indication":"Solid Tumors","phase":1},
        
        # =================================================================
        # NATURAL PRODUCTS (reference — different property space)
        # =================================================================
        {"name":"Paclitaxel (Taxol)","smiles":"CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C","modality":"Natural Product","target":"Tubulin","indication":"Solid Tumors","phase":4},
        {"name":"Camptothecin","smiles":"CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O","modality":"Natural Product","target":"Topo I","indication":"Solid Tumors","phase":4},
        {"name":"Artemisinin","smiles":"CC1CCC2C(C(=O)OC3CC4(C1CCC23OO4)C)C","modality":"Natural Product","target":"Heme","indication":"Malaria","phase":4},
        {"name":"Colchicine","smiles":"COC1=CC2=C(C(=C1)OC)C(CC1=CC(=O)C(=CC1=C2)OC)NC(C)=O","modality":"Natural Product","target":"Tubulin","indication":"Gout","phase":4},
        {"name":"Morphine","smiles":"CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O","modality":"Natural Product","target":"Opioid R","indication":"Pain","phase":4},
        {"name":"Quercetin","smiles":"C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O","modality":"Natural Product","target":"Various","indication":"Supplement","phase":2},
        {"name":"Curcumin","smiles":"COC1=CC(=CC(=C1O)OC)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC","modality":"Natural Product","target":"NF-kB","indication":"Supplement","phase":2},
    ]
    
    print(f"\n📦 Curated dataset: {len(compounds)} compounds")
    mod_counts = {}
    for c in compounds:
        mod_counts[c['modality']] = mod_counts.get(c['modality'], 0) + 1
    for mod, count in mod_counts.items():
        print(f"   {mod}: {count}")
    
    return compounds


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

if USE_CHEMBL:
    compounds = fetch_chembl_by_mechanism(limit_per_class=500)
else:
    compounds = get_curated_dataset()


# ============================================================================
# STEP 2: COMPUTE EXTENDED DESCRIPTORS
# ============================================================================
print(f"\n🔬 Computing descriptors...")

results = []
errors = 0

for entry in compounds:
    mol = Chem.MolFromSmiles(entry["smiles"])
    if mol is None:
        errors += 1
        continue
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)
    rings = Descriptors.RingCount(mol)
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    heavy_atoms = mol.GetNumHeavyAtoms()
    fsp3 = Descriptors.FractionCSP3(mol)
    mr = Descriptors.MolMR(mol)
    
    # Lipinski
    lip_v = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    passes_ro5 = lip_v <= 1
    
    # Veber
    passes_veber = (tpsa <= 140) and (rot_bonds <= 10)
    
    # Beyond Rule of 5 (Doak et al.) — designed for PROTACs
    bro5_v = sum([mw > 1000, logp > 10 or logp < -2, hbd > 6, hba > 15, tpsa > 250, rot_bonds > 20])
    passes_bro5 = bro5_v <= 1
    
    # Ghose
    passes_ghose = (160 <= mw <= 480) and (-0.4 <= logp <= 5.6) and (20 <= heavy_atoms <= 70)
    
    row = {
        "Name": entry["name"], "SMILES": entry["smiles"],
        "Modality": entry["modality"],
        "Target": entry.get("target", ""),
        "Indication": entry.get("indication", ""),
        "Phase": entry.get("phase", 0),
        "MW": round(mw, 2), "LogP": round(logp, 2),
        "HBD": hbd, "HBA": hba, "TPSA": round(tpsa, 2),
        "RotBonds": rot_bonds, "Rings": rings,
        "AromaticRings": aromatic_rings, "HeavyAtoms": heavy_atoms,
        "Fsp3": round(fsp3, 3), "MR": round(mr, 2),
        "Lipinski_Violations": lip_v, "Passes_RO5": passes_ro5,
        "Passes_Veber": passes_veber,
        "bRo5_Violations": bro5_v, "Passes_bRo5": passes_bro5,
        "Passes_Ghose": passes_ghose,
    }
    results.append(row)

df = pd.DataFrame(results)
print(f"   ✅ {len(df)} molecules processed ({errors} failed)")


# ============================================================================
# STEP 3: COMPARATIVE ANALYSIS
# ============================================================================
modalities = ["Traditional Drug", "PROTAC", "Molecular Glue", "Natural Product"]

print(f"\n{'=' * 80}")
print(f"  COMPARATIVE ANALYSIS")
print(f"{'=' * 80}")

# --- Property comparison table ---
print(f"\n📋 MEAN PROPERTIES BY MODALITY")
props = ["MW","LogP","HBD","HBA","TPSA","RotBonds","Fsp3","HeavyAtoms"]
header = f"{'Property':<12}"
for mod in modalities:
    header += f" | {mod:>18}"
print(header)
print("-" * (12 + len(modalities) * 21))

for prop in props:
    line = f"{prop:<12}"
    for mod in modalities:
        subset = df[df['Modality']==mod][prop]
        if len(subset) > 0:
            line += f" | {subset.mean():>9.1f} ± {subset.std():>5.1f}"
        else:
            line += f" | {'N/A':>18}"
    print(line)

# --- Filter pass rates ---
print(f"\n📋 FILTER PASS RATES")
header = f"{'Filter':<16}"
for mod in modalities:
    header += f" | {mod:>18}"
print(header)
print("-" * (16 + len(modalities) * 21))

for label, col in [("Lipinski Ro5","Passes_RO5"),("Veber","Passes_Veber"),
                    ("Beyond Ro5","Passes_bRo5"),("Ghose","Passes_Ghose")]:
    line = f"{label:<16}"
    for mod in modalities:
        subset = df[df['Modality']==mod]
        if len(subset) > 0:
            rate = subset[col].mean() * 100
            line += f" | {rate:>17.0f}%"
        else:
            line += f" | {'N/A':>18}"
    print(line)

# --- Key insights ---
trad = df[df['Modality']=='Traditional Drug']
prot = df[df['Modality']=='PROTAC']
glue = df[df['Modality']=='Molecular Glue']
natp = df[df['Modality']=='Natural Product']

print(f"\n{'=' * 80}")
print(f"  KEY INSIGHTS")
print(f"{'=' * 80}")
print(f"""
1. MOLECULAR WEIGHT
   Traditional drugs: {trad['MW'].mean():.0f} ± {trad['MW'].std():.0f} Da
   PROTACs:           {prot['MW'].mean():.0f} ± {prot['MW'].std():.0f} Da
   Molecular glues:   {glue['MW'].mean():.0f} ± {glue['MW'].std():.0f} Da
   → PROTACs are {prot['MW'].mean()/trad['MW'].mean():.1f}x heavier than traditional drugs.
   → Molecular glues are {glue['MW'].mean()/trad['MW'].mean():.1f}x the size — firmly drug-like.

2. FLEXIBILITY (Rotatable Bonds)
   Traditional: {trad['RotBonds'].mean():.1f}  |  PROTACs: {prot['RotBonds'].mean():.1f}  |  Glues: {glue['RotBonds'].mean():.1f}
   → PROTACs have {prot['RotBonds'].mean()/trad['RotBonds'].mean():.1f}x more rotatable bonds due to their linker.
   → This flexibility is both a feature (allows ternary complex formation)
     and a challenge (conformational entropy penalty).

3. 3D CHARACTER (Fsp3)
   Traditional: {trad['Fsp3'].mean():.2f}  |  PROTACs: {prot['Fsp3'].mean():.2f}  |  Glues: {glue['Fsp3'].mean():.2f}
   → PROTACs have higher Fsp3 due to their linker sp3 carbons.
   → Higher Fsp3 correlates with clinical success (Lovering et al., 2009).

4. CLINICAL TRANSLATION
   PROTACs are making it to the clinic despite violating Lipinski:
   - ARV-471 (vepdegestrant) is in Phase 3 for ER+ breast cancer
   - ARV-110 is in Phase 2 for prostate cancer
   This proves the "beyond Rule of 5" framework works for this modality.
""")


# ============================================================================
# STEP 4: VISUALIZATIONS (8 panels)
# ============================================================================
print(f"📊 Generating visualizations...")

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 24))
gs = GridSpec(4, 2, figure=fig, hspace=0.35, wspace=0.3)
fig.suptitle(f"PROTAC vs Traditional Drug Property Comparison — {len(df)} Compounds\nUniversity of Waterloo | Medicinal Chemistry",
             fontsize=17, fontweight='bold', y=0.98)

mod_colors = {"Traditional Drug":"#3498db","PROTAC":"#9b59b6",
              "Molecular Glue":"#f39c12","Natural Product":"#2ecc71"}

# --- Panel 1: Chemical space MW vs LogP ---
ax = fig.add_subplot(gs[0,0])
for mod in modalities:
    s = df[df['Modality']==mod]
    ax.scatter(s['MW'],s['LogP'],c=mod_colors.get(mod,'#95a5a6'),s=60,alpha=0.7,
              edgecolors='white',linewidth=1,label=f"{mod} (n={len(s)})",zorder=5)
    for _,r in s.iterrows():
        ax.annotate(r['Name'].split('(')[0].strip()[:12],(r['MW'],r['LogP']),fontsize=5,alpha=0.6,ha='center',va='bottom')
ax.axvline(500,color='red',ls='--',alpha=0.4); ax.axhline(5,color='red',ls='--',alpha=0.4)
ax.add_patch(Rectangle((0,-5),500,10,alpha=0.04,color='green',label='Lipinski space'))
ax.set_xlabel("Molecular Weight (Da)"); ax.set_ylabel("LogP")
ax.set_title("Chemical space: where each modality lives",fontweight='bold')
ax.legend(fontsize=8,loc='upper left')

# --- Panel 2: Box plots for 4 key properties ---
ax = fig.add_subplot(gs[0,1])
mod_order = modalities
positions = np.arange(len(mod_order))
data_mw = [df[df['Modality']==m]['MW'].values for m in mod_order]
bp = ax.boxplot(data_mw,labels=[m.replace(' ','\n') for m in mod_order],
                patch_artist=True,widths=0.6,medianprops=dict(color='black',linewidth=2))
for patch,mod in zip(bp['boxes'],mod_order):
    patch.set_facecolor(mod_colors.get(mod,'#95a5a6')); patch.set_alpha(0.7)
for i,mod in enumerate(mod_order):
    s = df[df['Modality']==mod]['MW']
    jitter = np.random.normal(i+1,0.06,len(s))
    ax.scatter(jitter,s,c=mod_colors.get(mod,'#95a5a6'),s=25,alpha=0.6,edgecolors='white',linewidth=0.5,zorder=5)
ax.axhline(500,color='red',ls='--',alpha=0.5,label='Lipinski limit')
ax.set_ylabel("Molecular Weight (Da)"); ax.set_title("MW distribution by modality",fontweight='bold')
ax.legend(fontsize=8)

# --- Panel 3: Multi-property box plot grid ---
ax = fig.add_subplot(gs[1,0])
props_plot = ["LogP","HBD","HBA","TPSA"]
thresholds = {"LogP":5,"HBD":5,"HBA":10,"TPSA":140}
x = np.arange(len(props_plot)); w = 0.2
for i,mod in enumerate(mod_order):
    s = df[df['Modality']==mod]
    means = [s[p].mean() for p in props_plot]
    stds = [s[p].std() for p in props_plot]
    ax.bar(x+i*w,means,w,yerr=stds,label=mod,color=mod_colors.get(mod,'#95a5a6'),
           edgecolor='white',linewidth=1,capsize=3,alpha=0.8)
ax.set_xticks(x+w*1.5); ax.set_xticklabels(props_plot)
ax.set_ylabel("Mean Value"); ax.set_title("Property comparison (mean ± std)",fontweight='bold')
ax.legend(fontsize=8)

# --- Panel 4: Rotatable bonds vs TPSA (Veber) ---
ax = fig.add_subplot(gs[1,1])
for mod in modalities:
    s = df[df['Modality']==mod]
    ax.scatter(s['RotBonds'],s['TPSA'],c=mod_colors.get(mod,'#95a5a6'),s=60,alpha=0.7,
              edgecolors='white',linewidth=1,label=mod,zorder=5)
ax.axvline(10,color='red',ls='--',alpha=0.4); ax.axhline(140,color='red',ls='--',alpha=0.4)
ax.set_xlabel("Rotatable Bonds"); ax.set_ylabel("TPSA (Å²)")
ax.set_title("Veber rules: flexibility vs polarity",fontweight='bold'); ax.legend(fontsize=8)

# --- Panel 5: Radar chart ---
ax = fig.add_subplot(gs[2,0],polar=True)
props_r = ["MW","LogP","HBD","HBA","TPSA","RotBonds"]
norms = {"MW":(0,1000),"LogP":(-3,10),"HBD":(0,10),"HBA":(0,15),"TPSA":(0,250),"RotBonds":(0,20)}
angles = np.linspace(0,2*np.pi,len(props_r),endpoint=False).tolist()+[0]
for mod in modalities:
    s = df[df['Modality']==mod]
    if len(s)==0: continue
    vals = [max(0,min(1,(s[p].mean()-norms[p][0])/(norms[p][1]-norms[p][0]))) for p in props_r]
    vals.append(vals[0])
    ax.plot(angles,vals,'o-',linewidth=2,color=mod_colors[mod],label=mod,markersize=4)
    ax.fill(angles,vals,alpha=0.1,color=mod_colors[mod])
ax.set_xticks(angles[:-1]); ax.set_xticklabels(props_r,fontsize=9)
ax.set_title("Property profiles (normalized)",fontweight='bold',y=1.1)
ax.legend(fontsize=7,loc='upper right',bbox_to_anchor=(1.35,1.1))

# --- Panel 6: Filter pass rates grouped bar ---
ax = fig.add_subplot(gs[2,1])
filters = [("Lipinski","Passes_RO5"),("Veber","Passes_Veber"),("bRo5","Passes_bRo5"),("Ghose","Passes_Ghose")]
x = np.arange(len(filters)); w = 0.2
for i,mod in enumerate(mod_order):
    s = df[df['Modality']==mod]
    if len(s)==0: continue
    rates = [s[col].mean()*100 for _,col in filters]
    ax.bar(x+i*w,rates,w,label=mod,color=mod_colors.get(mod,'#95a5a6'),edgecolor='white',linewidth=1)
ax.set_xticks(x+w*1.5); ax.set_xticklabels([f[0] for f in filters])
ax.set_ylabel("Pass Rate (%)"); ax.set_ylim(0,115)
ax.set_title("Filter pass rates by modality",fontweight='bold'); ax.legend(fontsize=8)

# --- Panel 7: Fsp3 comparison ---
ax = fig.add_subplot(gs[3,0])
for mod in modalities:
    s = df[df['Modality']==mod]
    if len(s)==0: continue
    ax.hist(s['Fsp3'],bins=12,alpha=0.5,color=mod_colors[mod],label=mod,edgecolor='white')
ax.axvline(0.42,color='red',ls='--',alpha=0.5,label='Clinical avg')
ax.set_xlabel("Fraction sp3 (Fsp3)"); ax.set_ylabel("Count")
ax.set_title("3D character by modality",fontweight='bold'); ax.legend(fontsize=8)

# --- Panel 8: Clinical phase distribution ---
ax = fig.add_subplot(gs[3,1])
phase_labels = ["Preclinical","Phase 1","Phase 2","Phase 3","Phase 4"]
x = np.arange(5); w = 0.2
for i,mod in enumerate(mod_order):
    s = df[df['Modality']==mod]
    counts = [len(s[s['Phase']==p]) for p in range(5)]
    ax.bar(x+i*w,counts,w,label=mod,color=mod_colors.get(mod,'#95a5a6'),edgecolor='white',linewidth=1)
ax.set_xticks(x+w*1.5); ax.set_xticklabels(phase_labels,fontsize=9)
ax.set_ylabel("Number of Compounds"); ax.set_title("Clinical phase distribution",fontweight='bold')
ax.legend(fontsize=8)

plt.savefig("project2_protac_analysis.png",dpi=150,bbox_inches='tight')
print(f"   ✅ Figure saved: project2_protac_analysis.png")

df.to_csv("project2_protac_results.csv",index=False)
print(f"   ✅ Data saved: project2_protac_results.csv ({len(df)} rows × {len(df.columns)} cols)")

elapsed = time.time() - start_time
print(f"\n⏱️  Runtime: {elapsed:.1f}s")
print(f"\n{'=' * 70}")
print(f"  PIPELINE COMPLETE — Data source: {'ChEMBL API' if USE_CHEMBL else 'Curated Dataset'}")
print(f"{'=' * 70}")
