#project 3

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, rdMolDescriptors, BRICS
from rdkit.Chem import Draw, rdmolops
from rdkit.Chem import DataStructs
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Patch
from matplotlib.gridspec import GridSpec
import numpy as np
import warnings, time
warnings.filterwarnings('ignore')
start_time = time.time()
print("=" * 80)
print("  MOLECULAR SIMILARITY SEARCH ENGINE")
print("  University of Waterloo | Medicinal Chemistry")
print("=" * 80)


# ============================================================================
# DATA SOURCE CONFIGURATION
# ============================================================================

USE_CHEMBL = False  # <-- CHANGE TO True ON YOUR MACHINE

CHEMBL_FILE_PATH = None  # <-- SET THIS to your ChEMBL file path

if USE_CHEMBL:
    try:
        from chembl_webresource_client.new_client import new_client
        print("✅ ChEMBL API detected")
    except ImportError:
        print("⚠️  ChEMBL client not found. Using curated dataset.")
        USE_CHEMBL = False


# ============================================================================
# DATA SOURCE 1: LIVE ChEMBL
# ============================================================================

def fetch_chembl_compounds(limit=3000):
    """Fetch approved drugs from ChEMBL for the similarity database."""
    molecule = new_client.molecule
    print(f"\n🌐 Building database from ChEMBL (limit={limit})...")
    
    results = molecule.filter(
        max_phase=4, molecule_type='Small molecule'
    ).only(['molecule_chembl_id','pref_name','molecule_structures','indication_class'])
    
    compounds = []
    count = 0
    for item in results:
        if count >= limit: break
        structs = item.get('molecule_structures')
        if not structs or not structs.get('canonical_smiles'): continue
        indication = item.get('indication_class','') or 'Unclassified'
        compounds.append({
            "name": item.get('pref_name','') or item['molecule_chembl_id'],
            "smiles": structs['canonical_smiles'],
            "class": indication[:25],
        })
        count += 1
        if count % 500 == 0: print(f"   ... {count} compounds")
    
    print(f"   ✅ {len(compounds)} compounds loaded")
    return compounds


# ============================================================================
# DATA SOURCE 2: FILE
# ============================================================================

def load_from_file(filepath, limit=5000):
    """Load from downloaded ChEMBL file."""
    if filepath.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(filepath)
        compounds = []
        for mol in supplier:
            if len(compounds) >= limit: break
            if mol is None: continue
            smiles = Chem.MolToSmiles(mol)
            cid = mol.GetProp('chembl_id') if mol.HasProp('chembl_id') else f'MOL_{len(compounds)}'
            compounds.append({"name":cid,"smiles":smiles,"class":"ChEMBL"})
        return compounds
    else:
        df = pd.read_csv(filepath, sep='\t', nrows=limit)
        smiles_col = [c for c in df.columns if 'smiles' in c.lower()][0]
        id_col = [c for c in df.columns if 'chembl' in c.lower()][0]
        return [{"name":str(r[id_col]),"smiles":str(r[smiles_col]),"class":"ChEMBL"}
                for _,r in df.iterrows() if pd.notna(r[smiles_col])]


# ============================================================================
# DATA SOURCE 3: CURATED DATASET
# ============================================================================

def get_curated_database():
    """Curated database of 70+ drugs organized by therapeutic class."""
    
    compounds = [
        # === KINASE INHIBITORS ===
        {"name":"Imatinib","smiles":"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5","class":"Kinase Inhibitor"},
        {"name":"Erlotinib","smiles":"COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC","class":"Kinase Inhibitor"},
        {"name":"Gefitinib","smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4","class":"Kinase Inhibitor"},
        {"name":"Sorafenib","smiles":"CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F","class":"Kinase Inhibitor"},
        {"name":"Sunitinib","smiles":"CCN(CC)CCNC(=O)C1=C(C(=C(S1)/C=C\\2/C3=CC=CC=C3NC2=O)C)C","class":"Kinase Inhibitor"},
        {"name":"Lapatinib","smiles":"CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)Cl)OCC5=CC(=CC=C5)F","class":"Kinase Inhibitor"},
        {"name":"Dasatinib","smiles":"CC1=NC(=CC(=N1)NC2=CC(=CC=C2)N3CCN(CC3)CCO)NC4=CC5=C(C=C4)C(=NN5)C(=O)NC6CCCCC6","class":"Kinase Inhibitor"},
        {"name":"Nilotinib","smiles":"CC1=C(C=C(C=C1)NC(=O)C2=CC(=C(C=C2)C3=CN=C(N=C3)NC4=CC(=CC=C4)C(F)(F)F)C#N)NC5=NC=CC(=N5)C6=CC=CN=C6","class":"Kinase Inhibitor"},
        {"name":"Bosutinib","smiles":"COC1=C(C=C2C(=C1OC)N=CN=C2NC3=CC(=C(C=C3Cl)Cl)OC)OC4=CC(=CC=C4)CN5CCN(CC5)C","class":"Kinase Inhibitor"},
        {"name":"Ponatinib","smiles":"CC1=C(C=CC(=C1)C(=O)NC2=CC(=CC=C2)C(F)(F)F)C#CC3=CC4=C(C=C3)C(=NN4)NC5=CC=C(C=C5)N6CCN(CC6)C","class":"Kinase Inhibitor"},
        {"name":"Osimertinib","smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)NC(=O)C=C","class":"Kinase Inhibitor"},
        {"name":"Crizotinib","smiles":"CC(C1=C(C=CC(=C1Cl)F)Cl)OC2=CN=C3C=C(C=CC3=C2)C4(CC4)N","class":"Kinase Inhibitor"},
        {"name":"Vemurafenib","smiles":"CCCS(=O)(=O)NC1=CC(=C(C=C1F)C(=O)C2=CNC3=NC=C(C=C23)C4=CC=C(C=C4)Cl)F","class":"Kinase Inhibitor"},
        {"name":"Ibrutinib","smiles":"C=CC(=O)N1CCC(CC1)N2C3=C(C=CC=C3)N=C2C4=CC=C(C=C4)OC5=CC=CC=C5","class":"Kinase Inhibitor"},
        {"name":"Palbociclib","smiles":"CC(=O)C1=C(C=C2CN=C(N=C2N1)NC3=NC=C(C=C3)N4CCNCC4)C","class":"Kinase Inhibitor"},
        {"name":"Abemaciclib","smiles":"CCN1C(=NC2=C1N=C(N=C2NC3=CC=C(C=C3)N4CCN(CC4)C)C5=CC(=NC=C5)F)C","class":"Kinase Inhibitor"},
        
        # === STATINS ===
        {"name":"Atorvastatin","smiles":"CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4","class":"Statin"},
        {"name":"Rosuvastatin","smiles":"CC(C1=NC(=NC(=C1C=CC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)N(C)S(=O)(=O)C)C","class":"Statin"},
        {"name":"Simvastatin","smiles":"CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C","class":"Statin"},
        
        # === NSAIDs ===
        {"name":"Aspirin","smiles":"CC(=O)OC1=CC=CC=C1C(=O)O","class":"NSAID"},
        {"name":"Ibuprofen","smiles":"CC(C)CC1=CC=C(C=C1)C(C)C(=O)O","class":"NSAID"},
        {"name":"Naproxen","smiles":"COC1=CC2=CC(=CC=C2C=C1)C(C)C(=O)O","class":"NSAID"},
        {"name":"Diclofenac","smiles":"OC(=O)CC1=CC=CC=C1NC2=C(Cl)C=CC=C2Cl","class":"NSAID"},
        {"name":"Celecoxib","smiles":"CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F","class":"NSAID/COX-2"},
        {"name":"Indomethacin","smiles":"CC1=C(C2=CC(=CC=C2N1C(=O)C3=CC=C(C=C3)Cl)OC)CC(=O)O","class":"NSAID"},
        
        # === SSRIs / ANTIDEPRESSANTS ===
        {"name":"Fluoxetine","smiles":"CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F","class":"SSRI"},
        {"name":"Sertraline","smiles":"CNC1CCC(C2=CC=CC=C21)C3=CC(=C(C=C3)Cl)Cl","class":"SSRI"},
        {"name":"Paroxetine","smiles":"C1CC(OC2=CC3=C(C=C2)OCO3)C(C1COC4=CC=C(C=C4)F)NC","class":"SSRI"},
        {"name":"Citalopram","smiles":"CN(C)CCCC1(C2=CC=C(C=C2)F)OCC3=CC(=CC=C31)C#N","class":"SSRI"},
        {"name":"Venlafaxine","smiles":"COC1=CC=C(C=C1)C(CN(C)C)C2(CCCCC2)O","class":"SNRI"},
        {"name":"Duloxetine","smiles":"CNCC(C1=CC=CS1)OC2=CC=C3C=CC=CC3=C2","class":"SNRI"},
        
        # === ANTIVIRALS ===
        {"name":"Oseltamivir","smiles":"CCOC(=O)C1=CC(CC(C1NC(=O)C)N)OC(CC)CC","class":"Antiviral"},
        {"name":"Acyclovir","smiles":"C1=NC2=C(N1COCCO)NC(=NC2=O)N","class":"Antiviral"},
        {"name":"Sofosbuvir","smiles":"CC(C)OC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=CC(=O)NC2=O)(C)F)O)OC3=CC=CC=C3","class":"Antiviral"},
        {"name":"Remdesivir","smiles":"CCC(CC)COC(=O)C(C)NP(=O)(OCC1C(C(C(O1)N2C=CC3=C2N=CN=C3N)O)O)OC4=CC=CC=C4","class":"Antiviral"},
        
        # === ANTIBIOTICS ===
        {"name":"Ciprofloxacin","smiles":"C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O","class":"Antibiotic"},
        {"name":"Levofloxacin","smiles":"CC1COC2=C3N1C=C(C(=O)C3=CC(=C2N4CCN(CC4)C)F)C(=O)O","class":"Antibiotic"},
        {"name":"Amoxicillin","smiles":"CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C","class":"Antibiotic"},
        
        # === PPIs ===
        {"name":"Omeprazole","smiles":"CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2","class":"PPI"},
        {"name":"Lansoprazole","smiles":"CC1=C(C=CN=C1CS(=O)C2=NC3=CC=CC=C3N2)OCC(F)(F)F","class":"PPI"},
        {"name":"Pantoprazole","smiles":"COC1=CC=NC(=C1OC)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC(F)F","class":"PPI"},
        
        # === CARDIOVASCULAR ===
        {"name":"Losartan","smiles":"CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl","class":"ARB"},
        {"name":"Amlodipine","smiles":"CCOC(=O)C1=C(NC(=C(C1C2=CC=CC=C2Cl)C(=O)OC)C)COCCN","class":"CCB"},
        {"name":"Metoprolol","smiles":"CC(C)NCC(COC1=CC=C(C=C1)CCOC)O","class":"Beta Blocker"},
        {"name":"Warfarin","smiles":"CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O","class":"Anticoagulant"},
        {"name":"Rivaroxaban","smiles":"O=C1OCC(N1C2=CC=C(C=C2)N3CC(OC3=O)CNC(=O)C4=CC=C(S4)Cl)C","class":"Factor Xa Inh."},
        {"name":"Apixaban","smiles":"COC1=CC=C(C=C1)N2C(=O)C(CC3=CC=C(C=C3)N4CCCCC4=O)N=C2C5=CC=CC=C5","class":"Factor Xa Inh."},
        
        # === METABOLIC ===
        {"name":"Metformin","smiles":"CN(C)C(=N)NC(=N)N","class":"Biguanide"},
        {"name":"Sitagliptin","smiles":"C1CN2C(=NN=C2C(F)(F)F)CN1C(=O)CC(CC3=CC(=C(C=C3F)F)F)N","class":"DPP-4 Inh."},
        {"name":"Empagliflozin","smiles":"OCC1OC(C(C(C1O)O)O)C2=CC=C(C=C2Cl)CC3=CC(=C(C=C3)OC4CCCC4)O","class":"SGLT2 Inh."},
        
        # === IMMUNOLOGY ===
        {"name":"Baricitinib","smiles":"CCS(=O)(=O)N1CC(C1)N2C=C(C(=N2)C3=CC=NC=C3)C#N","class":"JAK Inhibitor"},
        {"name":"Methotrexate","smiles":"CN(CC1=CN=C2N=C(N=C(N)C2=N1)N)C3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O","class":"DHFR Inh."},
        
        # === OTHER ===
        {"name":"Sildenafil","smiles":"CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C","class":"PDE5 Inh."},
        {"name":"Tadalafil","smiles":"CN1CC(=O)N2C(C1=O)CC3=C(C2C4=CC5=C(C=C4)OCO5)NC6=CC=CC=C36","class":"PDE5 Inh."},
        {"name":"Donepezil","smiles":"COC1=CC2=C(C=C1OC)C(=O)C(C2)CC3CCN(CC3)CC4=CC=CC=C4","class":"AChE Inh."},
        {"name":"Olaparib","smiles":"C1CC1C(=O)N2CCN(CC2)C(=O)C3=C(C=CC=C3F)CC4=NNC(=O)C5=CC=CC=C54","class":"PARP Inh."},
        {"name":"Enzalutamide","smiles":"CNC(=O)C1=CC=C(C=C1)N2C(=O)N(C(=S)C2(C)C)C3=CC=C(C#N)C(=C3)C(F)(F)F","class":"AR Antagonist"},
        {"name":"Tamoxifen","smiles":"CCC(=C(C1=CC=CC=C1)C2=CC=CC=C2)C3=CC=C(C=C3)OCCN(C)C","class":"SERM"},
        {"name":"Letrozole","smiles":"C1=CC=C(C=C1)C(C2=CC=C(C=C2)C#N)N3C=NC=N3","class":"Aromatase Inh."},
        {"name":"Acetaminophen","smiles":"CC(=O)NC1=CC=C(C=C1)O","class":"Analgesic"},
        {"name":"Caffeine","smiles":"CN1C=NC2=C1C(=O)N(C(=O)N2C)C","class":"Methylxanthine"},
        {"name":"Morphine","smiles":"CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O","class":"Opioid"},
        {"name":"Memantine","smiles":"C1C2CC3CC1CC(C2)(C3)N","class":"NMDA Ant."},
    ]
    
    print(f"\n📦 Curated database: {len(compounds)} compounds")
    class_counts = {}
    for c in compounds:
        class_counts[c['class']] = class_counts.get(c['class'], 0) + 1
    for cls in sorted(class_counts, key=class_counts.get, reverse=True)[:8]:
        print(f"   {cls}: {class_counts[cls]}")
    if len(class_counts) > 8:
        print(f"   ... and {len(class_counts)-8} more classes")
    
    return compounds


# ============================================================================
# STEP 1: BUILD THE DATABASE
# ============================================================================

import os

if USE_CHEMBL:
    raw_compounds = fetch_chembl_compounds(limit=3000)
elif CHEMBL_FILE_PATH and os.path.exists(CHEMBL_FILE_PATH):
    raw_compounds = load_from_file(CHEMBL_FILE_PATH, limit=5000)
else:
    raw_compounds = get_curated_database()


# ============================================================================
# STEP 2: GENERATE FINGERPRINTS
# ============================================================================
print(f"\n🔧 Generating Morgan fingerprints (ECFP4, 2048 bits)...")

database = []
fp_errors = 0

for entry in raw_compounds:
    mol = Chem.MolFromSmiles(entry["smiles"])
    if mol is None:
        fp_errors += 1
        continue
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    
    database.append({
        "name": entry["name"],
        "smiles": entry["smiles"],
        "mol": mol,
        "fp": fp,
        "class": entry["class"],
        "mw": round(Descriptors.MolWt(mol), 1),
        "logp": round(Descriptors.MolLogP(mol), 2),
    })

print(f"   ✅ {len(database)} fingerprints generated ({fp_errors} errors)")


# ============================================================================
# STEP 3: SIMILARITY SEARCH FUNCTION
# ============================================================================

def similarity_search(query_smiles, query_name, db, top_n=20):
    """
    Find the most similar molecules to a query using Tanimoto similarity
    on Morgan (ECFP4) fingerprints.
    """
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        print(f"   ❌ Could not parse: {query_smiles}")
        return None
    
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, radius=2, nBits=2048)
    
    results = []
    for entry in db:
        if entry["name"] == query_name:
            continue
        sim = DataStructs.TanimotoSimilarity(query_fp, entry["fp"])
        results.append({
            "Name": entry["name"],
            "Class": entry["class"],
            "Tanimoto": round(sim, 4),
            "MW": entry["mw"],
            "LogP": entry["logp"],
        })
    
    results_df = pd.DataFrame(results).sort_values("Tanimoto", ascending=False)
    return results_df.head(top_n)


# ============================================================================
# STEP 4: RUN QUERIES
# ============================================================================

queries = [
    {
        "name": "Imatinib",
        "smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "desc": "BCR-ABL kinase inhibitor — first targeted cancer therapy (CML)",
    },
    {
        "name": "Gefitinib",
        "smiles": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
        "desc": "EGFR kinase inhibitor for non-small cell lung cancer",
    },
    {
        "name": "Aspirin",
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "desc": "Classic NSAID — cyclooxygenase inhibitor",
    },
    {
        "name": "Omeprazole",
        "smiles": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2",
        "desc": "Proton pump inhibitor for GERD",
    },
    {
        "name": "Fluoxetine",
        "smiles": "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",
        "desc": "SSRI antidepressant (Prozac)",
    },
]

all_query_results = {}

for q in queries:
    print(f"\n{'=' * 70}")
    print(f"  QUERY: {q['name']}")
    print(f"  {q['desc']}")
    print(f"{'=' * 70}")
    
    results = similarity_search(q["smiles"], q["name"], database)
    all_query_results[q["name"]] = results
    
    if results is not None:
        print(f"\n{'Rank':<5} {'Name':<20} {'Class':<18} {'Tanimoto':>10} {'MW':>8} {'LogP':>8}")
        print("-" * 75)
        for i, (_, row) in enumerate(results.head(15).iterrows(), 1):
            if row["Tanimoto"] >= 0.7: ind = "🟢"
            elif row["Tanimoto"] >= 0.4: ind = "🟡"
            else: ind = "🔴"
            print(f"{i:<5} {row['Name']:<20} {row['Class']:<18} {row['Tanimoto']:>8.4f} {ind} {row['MW']:>6.1f} {row['LogP']:>8.2f}")
        
        # Check if top hits are from the same class
        top5_classes = results.head(5)['Class'].tolist()
        query_class = next((e['class'] for e in database if e['name']==q['name']), 'Unknown')
        matches = sum(1 for c in top5_classes if c == query_class)
        print(f"\n   Top-5 class match rate: {matches}/5 match query class '{query_class}'")


# ============================================================================
# STEP 5: PAIRWISE SIMILARITY HEATMAP (Kinase Inhibitors)
# ============================================================================
print(f"\n{'=' * 70}")
print(f"  PAIRWISE SIMILARITY ANALYSIS: KINASE INHIBITOR FAMILY")
print(f"{'=' * 70}")

kinase_entries = [e for e in database if e["class"] == "Kinase Inhibitor"]
n_kin = len(kinase_entries)
sim_matrix = np.zeros((n_kin, n_kin))

for i in range(n_kin):
    for j in range(n_kin):
        sim_matrix[i][j] = DataStructs.TanimotoSimilarity(
            kinase_entries[i]["fp"], kinase_entries[j]["fp"]
        )

kin_names = [e["name"] for e in kinase_entries]

# Find the most similar pairs (excluding self)
pairs = []
for i in range(n_kin):
    for j in range(i+1, n_kin):
        pairs.append((kin_names[i], kin_names[j], sim_matrix[i][j]))
pairs.sort(key=lambda x: x[2], reverse=True)

print(f"\n📋 TOP 10 MOST SIMILAR KINASE INHIBITOR PAIRS")
print(f"   {'Drug A':<18} {'Drug B':<18} {'Tanimoto':>10}")
print(f"   {'-'*50}")
for a, b, sim in pairs[:10]:
    print(f"   {a:<18} {b:<18} {sim:>10.4f}")


# ============================================================================
# STEP 6: FULL DATABASE SIMILARITY MATRIX (all classes)
# ============================================================================
print(f"\n📋 INTER-CLASS SIMILARITY (mean Tanimoto between drug classes)")

classes = list(set(e['class'] for e in database))
classes = [c for c in classes if sum(1 for e in database if e['class']==c) >= 3]
classes.sort()

print(f"\n   {'':>18}", end="")
for c in classes[:8]:
    print(f" {c[:10]:>12}", end="")
print()

for c1 in classes[:8]:
    entries_1 = [e for e in database if e['class']==c1]
    print(f"   {c1[:18]:<18}", end="")
    for c2 in classes[:8]:
        entries_2 = [e for e in database if e['class']==c2]
        sims = []
        for e1 in entries_1:
            for e2 in entries_2:
                if e1['name'] != e2['name']:
                    sims.append(DataStructs.TanimotoSimilarity(e1['fp'], e2['fp']))
        mean_sim = np.mean(sims) if sims else 0
        print(f" {mean_sim:>12.3f}", end="")
    print()


# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================
print(f"\n📊 Generating visualizations...")

plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 20))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)
fig.suptitle(f"Molecular Similarity Search — {len(database)} Compound Database\nUniversity of Waterloo | Medicinal Chemistry",
             fontsize=17, fontweight='bold', y=0.98)

class_colors = {
    "Kinase Inhibitor":"#e74c3c","Statin":"#3498db","NSAID":"#e67e22",
    "NSAID/COX-2":"#d35400","SSRI":"#9b59b6","SNRI":"#8e44ad",
    "Antiviral":"#2ecc71","Antibiotic":"#1abc9c","PPI":"#f39c12",
    "ARB":"#16a085","CCB":"#2980b9","Beta Blocker":"#34495e",
    "Anticoagulant":"#7f8c8d","Factor Xa Inh.":"#95a5a6",
    "Biguanide":"#c0392b","DPP-4 Inh.":"#d35400","SGLT2 Inh.":"#27ae60",
    "JAK Inhibitor":"#e74c3c","PDE5 Inh.":"#8e44ad","PARP Inh.":"#2c3e50",
    "Opioid":"#7f8c8d","AChE Inh.":"#1abc9c",
}

# --- Panel 1-3: Similarity search results for first 3 queries ---
for idx, q_name in enumerate(list(all_query_results.keys())[:3]):
    ax = fig.add_subplot(gs[0, 0] if idx==0 else (gs[0,1] if idx==1 else gs[1,0]))
    res = all_query_results[q_name].head(12)
    colors = [class_colors.get(c, '#95a5a6') for c in res['Class']]
    y_pos = range(len(res)-1, -1, -1)
    bars = ax.barh(list(y_pos), res['Tanimoto'], color=colors, edgecolor='white', linewidth=1, height=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(res['Name'], fontsize=8)
    ax.set_xlabel("Tanimoto Similarity")
    ax.set_title(f"Query: {q_name}", fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.axvline(0.7, color='red', ls='--', alpha=0.4, lw=1.5)
    ax.axvline(0.4, color='orange', ls='--', alpha=0.3, lw=1)
    for bar, val in zip(bars, res['Tanimoto']):
        ax.text(bar.get_width()+0.01, bar.get_y()+bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=7, fontweight='bold')

# --- Panel 4: Kinase inhibitor heatmap ---
ax = fig.add_subplot(gs[1, 1])
im = ax.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(n_kin))
ax.set_yticks(range(n_kin))
ax.set_xticklabels(kin_names, rotation=45, ha='right', fontsize=7)
ax.set_yticklabels(kin_names, fontsize=7)
for i in range(n_kin):
    for j in range(n_kin):
        color = 'white' if sim_matrix[i][j] > 0.5 or sim_matrix[i][j] < 0.15 else 'black'
        ax.text(j, i, f'{sim_matrix[i][j]:.2f}', ha='center', va='center', fontsize=5, color=color)
plt.colorbar(im, ax=ax, label='Tanimoto', shrink=0.8)
ax.set_title("Kinase inhibitor pairwise similarity", fontweight='bold')

# --- Panel 5: Similarity distribution histogram ---
ax = fig.add_subplot(gs[2, 0])
# Compute all pairwise similarities
all_sims = []
same_class_sims = []
diff_class_sims = []
for i in range(len(database)):
    for j in range(i+1, len(database)):
        sim = DataStructs.TanimotoSimilarity(database[i]['fp'], database[j]['fp'])
        all_sims.append(sim)
        if database[i]['class'] == database[j]['class']:
            same_class_sims.append(sim)
        else:
            diff_class_sims.append(sim)

ax.hist(diff_class_sims, bins=50, alpha=0.6, color='#3498db', label=f'Different class (n={len(diff_class_sims)})', edgecolor='white')
ax.hist(same_class_sims, bins=50, alpha=0.6, color='#e74c3c', label=f'Same class (n={len(same_class_sims)})', edgecolor='white')
ax.axvline(0.7, color='red', ls='--', alpha=0.7, label='Similarity threshold (0.7)')
ax.set_xlabel("Tanimoto Similarity"); ax.set_ylabel("Pair Count")
ax.set_title("Similarity distribution: same vs different class", fontweight='bold')
ax.legend(fontsize=8)

mean_same = np.mean(same_class_sims) if same_class_sims else 0
mean_diff = np.mean(diff_class_sims) if diff_class_sims else 0
ax.text(0.95, 0.95, f'Mean same-class: {mean_same:.3f}\nMean diff-class: {mean_diff:.3f}',
        transform=ax.transAxes, fontsize=9, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# --- Panel 6: Query molecule property context ---
ax = fig.add_subplot(gs[2, 1])
for entry in database:
    color = class_colors.get(entry['class'], '#95a5a6')
    ax.scatter(entry['mw'], entry['logp'], c=color, s=20, alpha=0.3, edgecolors='none')

# Highlight query molecules
for q in queries:
    mol = Chem.MolFromSmiles(q['smiles'])
    if mol:
        qmw = Descriptors.MolWt(mol)
        qlp = Descriptors.MolLogP(mol)
        ax.scatter(qmw, qlp, c='black', s=150, marker='*', zorder=10)
        ax.annotate(q['name'], (qmw, qlp), fontsize=8, fontweight='bold',
                   xytext=(5, 5), textcoords='offset points')

ax.axvline(500, color='red', ls='--', alpha=0.3)
ax.axhline(5, color='red', ls='--', alpha=0.3)
ax.set_xlabel("Molecular Weight (Da)"); ax.set_ylabel("LogP")
ax.set_title("Query molecules in chemical space (★)", fontweight='bold')

plt.savefig("project3_similarity_search.png", dpi=150, bbox_inches='tight')
print(f"   ✅ Figure saved: project3_similarity_search.png")

# Save results
for q_name, res in all_query_results.items():
    fname = f"project3_{q_name.lower().replace(' ','_')}_results.csv"
    res.to_csv(fname, index=False)
print(f"   ✅ Query results saved to CSV files")

elapsed = time.time() - start_time
print(f"\n⏱️  Runtime: {elapsed:.1f}s")
print(f"\n{'=' * 70}")
print(f"  PIPELINE COMPLETE — {len(database)} compounds in database")
print(f"  Data source: {'ChEMBL API' if USE_CHEMBL else 'Curated Dataset'}")
print(f"  To switch to ChEMBL: set USE_CHEMBL = True at the top")
print(f"{'=' * 70}")
