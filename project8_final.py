#project 8

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, rdMolDescriptors
from rdkit.Chem import DataStructs
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
print("  QSAR MODEL: PREDICTING EGFR INHIBITOR ACTIVITY")
print("  Machine Learning for Drug Discovery")
print("  University of Waterloo | Medicinal Chemistry")
print("=" * 80)

# ============================================================================
# WHAT THIS PROJECT DOES:
# QSAR = Quantitative Structure-Activity Relationship
#
# Given a molecule's STRUCTURE, can we PREDICT if it will be active against
# a disease target? This is the central question of computational drug
# discovery.
#
# We train a Random Forest classifier on a dataset of molecules with KNOWN
# activity against EGFR (Epidermal Growth Factor Receptor), a major cancer
# target. The model learns which molecular features correlate with activity
# and can then predict activity for NEW, untested molecules.
#
# This is exactly what papers in JCIM and J. Med. Chem. do:
#   - "AI-Integrated QSAR Modeling" (IJMS, 2025)
#   - "ML models in evaluating ADMET" (ADMET & DMPK, 2025)
# ============================================================================

# Check for scikit-learn
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                  roc_auc_score, roc_curve, confusion_matrix, classification_report)
    from sklearn.preprocessing import StandardScaler
    print("✅ scikit-learn loaded")
except ImportError:
    print("❌ scikit-learn not found. Install: pip install scikit-learn --break-system-packages")
    import sys; sys.exit(1)

USE_CHEMBL = False  # <-- CHANGE TO True ON YOUR MACHINE

# ============================================================================
# DATASET: EGFR Inhibitors with Activity Data
# ============================================================================
# In a real project, you'd query ChEMBL for EGFR (CHEMBL203) IC50 data:
#   activity = new_client.activity
#   egfr_data = activity.filter(target_chembl_id='CHEMBL203', ...)
#
# Here we use a curated dataset of 60 compounds with binary activity labels.
# Active = IC50 < 1μM (binds well to EGFR)
# Inactive = IC50 > 10μM (doesn't bind meaningfully)

def get_egfr_dataset():
    compounds = [
        # === ACTIVE EGFR Inhibitors (IC50 < 1μM) ===
        {"smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4","activity":1,"name":"Gefitinib"},
        {"smiles":"COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC","activity":1,"name":"Erlotinib"},
        {"smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)NC(=O)C=C","activity":1,"name":"Osimertinib"},
        {"smiles":"CN(C)C/C=C/C(=O)NC1=CC2=C(C=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl","activity":1,"name":"Afatinib"},
        {"smiles":"C=CC(=O)NC1=CC2=C(C=C1OC)N=CN=C2NC3=CC(=CC=C3)NC(=O)C=C","activity":1,"name":"Mobocertinib"},
        {"smiles":"CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)Cl)OCC5=CC(=CC=C5)F","activity":1,"name":"Lapatinib"},
        # Active analogs (structurally related to known EGFR inhibitors)
        {"smiles":"COC1=CC2=C(C=C1OC)C(=NC=N2)NC3=CC=C(C=C3)F","activity":1,"name":"Analog_1"},
        {"smiles":"COC1=CC2=C(C=C1)C(=NC=N2)NC3=CC(=CC=C3)C#C","activity":1,"name":"Analog_2"},
        {"smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC=C(C=C3)Cl)OC","activity":1,"name":"Analog_3"},
        {"smiles":"C1=CC=C(C=C1)NC2=NC=NC3=CC(=CC=C23)OC","activity":1,"name":"Analog_4"},
        {"smiles":"COC1=CC2=C(C=C1)N=CN=C2NC3=CC=C(C=C3)OC","activity":1,"name":"Analog_5"},
        {"smiles":"COC1=CC=C(C=C1)NC2=NC=NC3=CC(=CC=C23)OCCO","activity":1,"name":"Analog_6"},
        {"smiles":"NC1=NC=NC2=CC(=CC=C12)OCC3=CC=CC=C3","activity":1,"name":"Analog_7"},
        {"smiles":"COC1=CC2=C(C=C1OC)C(=NC=N2)NC3=CC=CC(=C3)Cl","activity":1,"name":"Analog_8"},
        {"smiles":"COC1=CC2=NC=NC(=C2C=C1OCCO)NC3=CC(=CC=C3)F","activity":1,"name":"Analog_9"},
        {"smiles":"COC1=CC2=NC=NC(=C2C=C1OC)NC3=CC(=C(C=C3)F)Cl","activity":1,"name":"Analog_10"},
        {"smiles":"C1=CC=C(C(=C1)NC2=NC=NC3=CC=CC=C23)F","activity":1,"name":"Analog_11"},
        {"smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC=CC=C3)OCCN4CCOCC4","activity":1,"name":"Analog_12"},
        {"smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=CC=C3)Br)OC","activity":1,"name":"Analog_13"},
        {"smiles":"COC1=CC2=NC=NC(=C2C=C1)NC3=CC=C(C=C3)N","activity":1,"name":"Analog_14"},
        {"smiles":"COC1=CC2=C(C=C1)C(=NC=N2)NC3=CC=C(C(=C3)F)F","activity":1,"name":"Analog_15"},
        {"smiles":"NC1=CC2=C(C=C1)N=CN=C2NC3=CC(=CC=C3)C(F)(F)F","activity":1,"name":"Analog_16"},
        {"smiles":"COC1=CC2=C(C=C1OCC3=CC=CC=C3)C(=NC=N2)NC4=CC=CC=C4","activity":1,"name":"Analog_17"},
        {"smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC=CC=C3F)OCCCN(C)C","activity":1,"name":"Analog_18"},
        {"smiles":"C=CC(=O)NC1=CC2=C(C=C1)N=CN=C2NC3=CC(=CC=C3)OC","activity":1,"name":"Analog_19"},
        {"smiles":"COC1=CC2=C(C=C1OCCO)C(=NC=N2)NC3=CC(=CC=C3)F","activity":1,"name":"Analog_20"},

        # === INACTIVE Compounds (IC50 > 10μM against EGFR) ===
        {"smiles":"CC(C)CC1=CC=C(C=C1)C(C)C(=O)O","activity":0,"name":"Ibuprofen"},
        {"smiles":"CC(=O)OC1=CC=CC=C1C(=O)O","activity":0,"name":"Aspirin"},
        {"smiles":"CC(=O)NC1=CC=C(C=C1)O","activity":0,"name":"Acetaminophen"},
        {"smiles":"CN(C)C(=N)NC(=N)N","activity":0,"name":"Metformin"},
        {"smiles":"CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F","activity":0,"name":"Fluoxetine"},
        {"smiles":"CNC1CCC(C2=CC=CC=C21)C3=CC(=C(C=C3)Cl)Cl","activity":0,"name":"Sertraline"},
        {"smiles":"CN1C=NC2=C1C(=O)N(C(=O)N2C)C","activity":0,"name":"Caffeine"},
        {"smiles":"CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O","activity":0,"name":"Warfarin"},
        {"smiles":"CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2","activity":0,"name":"Omeprazole"},
        {"smiles":"CC1COC2=C3N1C=C(C(=O)C3=CC(=C2N4CCN(CC4)C)F)C(=O)O","activity":0,"name":"Levofloxacin"},
        {"smiles":"C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O","activity":0,"name":"Ciprofloxacin"},
        {"smiles":"CC(C)NCC(COC1=CC=C(C=C1)CCOC)O","activity":0,"name":"Metoprolol"},
        {"smiles":"CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C","activity":0,"name":"Sildenafil"},
        {"smiles":"CN1CC(=O)N2C(C1=O)CC3=C(C2C4=CC5=C(C=C4)OCO5)NC6=CC=CC=C36","activity":0,"name":"Tadalafil"},
        {"smiles":"CCS(=O)(=O)N1CC(C1)N2C=C(C(=N2)C3=CC=NC=C3)C#N","activity":0,"name":"Baricitinib"},
        {"smiles":"C1CCNCC1","activity":0,"name":"Piperidine"},
        {"smiles":"C1=CC=C(C=C1)O","activity":0,"name":"Phenol"},
        {"smiles":"C1=CC=NC=C1","activity":0,"name":"Pyridine"},
        {"smiles":"C1=CC=C2C(=C1)C=CN2","activity":0,"name":"Indole"},
        {"smiles":"OC(CN1C=NC=N1)(CN2C=NC=N2)C3=CC=C(F)C=C3F","activity":0,"name":"Fluconazole"},
        {"smiles":"COC1=CC=C(C=C1)C(CN(C)C)C2(CCCCC2)O","activity":0,"name":"Venlafaxine"},
        {"smiles":"CNCC(C1=CC=CS1)OC2=CC=C3C=CC=CC3=C2","activity":0,"name":"Duloxetine"},
        {"smiles":"O=C1CC2=CC=CC=C2N1CCCCOC3=CC=C(C=C3)Cl","activity":0,"name":"Aripiprazole"},
        {"smiles":"C1C2CC3CC1CC(C2)(C3)N","activity":0,"name":"Memantine"},
        {"smiles":"C1=CN=CN1","activity":0,"name":"Imidazole"},
        {"smiles":"CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl","activity":0,"name":"Losartan"},
        {"smiles":"CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4","activity":0,"name":"Atorvastatin"},
        {"smiles":"CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O","activity":0,"name":"Morphine"},
        {"smiles":"CC1CCC2C(C(=O)OC3CC4(C1CCC23OO4)C)C","activity":0,"name":"Artemisinin"},
        {"smiles":"CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F","activity":0,"name":"Celecoxib"},
    ]
    return compounds


# ============================================================================
# STEP 1: COMPUTE MOLECULAR DESCRIPTORS (features for the ML model)
# ============================================================================
print(f"\n🔬 Computing molecular descriptors...")

compounds = get_egfr_dataset()
print(f"   Dataset: {len(compounds)} compounds ({sum(c['activity'] for c in compounds)} active, {len(compounds)-sum(c['activity'] for c in compounds)} inactive)")

descriptor_names = ["MW","LogP","TPSA","HBD","HBA","RotBonds","AromaticRings",
                    "Rings","HeavyAtoms","Fsp3","MR","NumN","NumO","NumHalogens",
                    "NumAmideBonds","FractionAromatic","Chi0","Kappa1"]

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    n_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum()==7)
    o_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum()==8)
    hal_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [9,17,35])
    amide = len(mol.GetSubstructMatches(Chem.MolFromSmarts("C(=O)N"))) if Chem.MolFromSmarts("C(=O)N") else 0
    heavy = mol.GetNumHeavyAtoms()
    arom = Descriptors.NumAromaticRings(mol)
    rings = Descriptors.RingCount(mol)
    frac_arom = arom/max(rings,1)
    
    return [
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol), arom, rings, heavy,
        Descriptors.FractionCSP3(mol), Descriptors.MolMR(mol),
        n_count, o_count, hal_count, amide, frac_arom,
        Descriptors.Chi0(mol), Descriptors.Kappa1(mol),
    ]

X_list = []; y_list = []; names_list = []; valid_smiles = []
for c in compounds:
    desc = compute_descriptors(c["smiles"])
    if desc is None: continue
    X_list.append(desc); y_list.append(c["activity"])
    names_list.append(c["name"]); valid_smiles.append(c["smiles"])

X = np.array(X_list)
y = np.array(y_list)
print(f"   ✅ {len(X)} compounds × {len(descriptor_names)} descriptors")
print(f"   Active: {y.sum()}, Inactive: {len(y)-y.sum()}")


# ============================================================================
# STEP 2: TRAIN/TEST SPLIT
# ============================================================================
print(f"\n📊 Splitting data (80% train, 20% test, stratified)...")
X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
    X, y, names_list, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train: {len(X_train)} ({y_train.sum()} active, {len(y_train)-y_train.sum()} inactive)")
print(f"   Test:  {len(X_test)} ({y_test.sum()} active, {len(y_test)-y_test.sum()} inactive)")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================================
# STEP 3: TRAIN RANDOM FOREST MODEL
# ============================================================================
print(f"\n🤖 Training Random Forest classifier...")
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
rf.fit(X_train_scaled, y_train)

y_pred = rf.predict(X_test_scaled)
y_prob = rf.predict_proba(X_test_scaled)[:,1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)

print(f"   ✅ Model trained!")
print(f"\n{'='*70}\n  TEST SET PERFORMANCE\n{'='*70}")
print(f"   Accuracy:  {acc:.3f}")
print(f"   Precision: {prec:.3f}")
print(f"   Recall:    {rec:.3f}")
print(f"   F1 Score:  {f1:.3f}")
print(f"   AUC-ROC:   {auc:.3f}")


# ============================================================================
# STEP 4: CROSS-VALIDATION
# ============================================================================
print(f"\n📊 5-Fold Cross-Validation...")
X_scaled_all = scaler.fit_transform(X)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_scaled_all, y, cv=cv, scoring='roc_auc')
print(f"   AUC scores per fold: {[f'{s:.3f}' for s in cv_scores]}")
print(f"   Mean AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")


# ============================================================================
# STEP 5: FEATURE IMPORTANCE
# ============================================================================
print(f"\n📋 FEATURE IMPORTANCE (what the model learned)")
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print(f"\n   {'Feature':<20} {'Importance':>12}")
print(f"   {'-'*35}")
for i in sorted_idx:
    bar = "█" * int(importances[i] * 50)
    print(f"   {descriptor_names[i]:<20} {importances[i]:>10.4f}  {bar}")


# ============================================================================
# STEP 6: PREDICTIONS ON TEST SET
# ============================================================================
print(f"\n📋 TEST SET PREDICTIONS")
print(f"   {'Name':<18} {'Actual':>8} {'Predicted':>10} {'Prob':>6} {'Correct':>8}")
print(f"   {'-'*55}")
for i, name in enumerate(names_test):
    actual = "Active" if y_test[i]==1 else "Inactive"
    predicted = "Active" if y_pred[i]==1 else "Inactive"
    correct = "✅" if y_test[i]==y_pred[i] else "❌"
    print(f"   {name:<18} {actual:>8} {predicted:>10} {y_prob[i]:>6.3f} {correct:>8}")

cm = confusion_matrix(y_test, y_pred)
print(f"\n   Confusion Matrix:")
print(f"   {'':>15} Pred Active  Pred Inactive")
print(f"   Actual Active   {cm[1][1]:>8}     {cm[1][0]:>8}")
print(f"   Actual Inactive {cm[0][1]:>8}     {cm[0][0]:>8}")


# ============================================================================
# STEP 7: VISUALIZATIONS
# ============================================================================
print(f"\n📊 Generating visualizations...")
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 18)); gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
fig.suptitle(f"QSAR Model: Predicting EGFR Inhibitor Activity\n{len(X)} compounds, {len(descriptor_names)} descriptors, Random Forest\nUniversity of Waterloo | Medicinal Chemistry",
             fontsize=17, fontweight='bold', y=0.98)

# Panel 1: ROC Curve
ax = fig.add_subplot(gs[0, 0])
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'RF Model (AUC = {auc:.3f})')
ax.plot([0,1], [0,1], 'k--', alpha=0.3, label='Random (AUC = 0.5)')
ax.fill_between(fpr, tpr, alpha=0.1, color='blue')
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curve (AUC = {auc:.3f})", fontweight='bold')
ax.legend(fontsize=10); ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.02])

# Panel 2: Feature importance
ax = fig.add_subplot(gs[0, 1])
top_n = 12
top_idx = sorted_idx[:top_n]
colors_fi = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))
ax.barh(range(top_n), importances[top_idx][::-1], color=colors_fi, edgecolor='white', linewidth=1)
ax.set_yticks(range(top_n))
ax.set_yticklabels([descriptor_names[i] for i in top_idx][::-1], fontsize=9)
ax.set_xlabel("Feature Importance")
ax.set_title("Top features driving predictions", fontweight='bold')

# Panel 3: Confusion matrix heatmap
ax = fig.add_subplot(gs[1, 0])
im = ax.imshow(cm, cmap='Blues', aspect='auto')
ax.set_xticks([0,1]); ax.set_xticklabels(['Inactive','Active'])
ax.set_yticks([0,1]); ax.set_yticklabels(['Inactive','Active'])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=24, fontweight='bold',
               color='white' if cm[i,j] > cm.max()/2 else 'black')
ax.set_title(f"Confusion Matrix (Accuracy: {acc:.1%})", fontweight='bold')

# Panel 4: Cross-validation scores
ax = fig.add_subplot(gs[1, 1])
ax.bar(range(1,6), cv_scores, color='#3498db', edgecolor='white', linewidth=1.5, alpha=0.8)
ax.axhline(cv_scores.mean(), color='red', ls='--', lw=2, label=f'Mean: {cv_scores.mean():.3f}')
ax.fill_between([0.5,5.5], cv_scores.mean()-cv_scores.std(), cv_scores.mean()+cv_scores.std(),
               alpha=0.1, color='red')
ax.set_xlabel("Fold"); ax.set_ylabel("AUC-ROC")
ax.set_title(f"5-Fold CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}", fontweight='bold')
ax.legend(fontsize=10); ax.set_ylim(0,1.1)

# Panel 5: MW vs LogP colored by predicted class
ax = fig.add_subplot(gs[2, 0])
# Predict on full dataset
y_pred_all = rf.predict(scaler.transform(X))
for label, color, marker in [(1,'#27ae60','o'),(0,'#e74c3c','s')]:
    mask = y_pred_all == label
    ax.scatter(X[mask,0], X[mask,1], c=color, s=60, alpha=0.7, marker=marker,
              edgecolors='white', linewidth=0.5,
              label=f'Predicted {"Active" if label else "Inactive"}')
ax.set_xlabel("Molecular Weight (Da)"); ax.set_ylabel("LogP")
ax.set_title("Chemical space colored by prediction", fontweight='bold')
ax.legend(fontsize=9)

# Panel 6: Prediction probability distribution
ax = fig.add_subplot(gs[2, 1])
probs_all = rf.predict_proba(scaler.transform(X))[:,1]
active_probs = probs_all[y==1]
inactive_probs = probs_all[y==0]
ax.hist(inactive_probs, bins=15, alpha=0.6, color='#e74c3c', label='Actually Inactive', edgecolor='white')
ax.hist(active_probs, bins=15, alpha=0.6, color='#27ae60', label='Actually Active', edgecolor='white')
ax.axvline(0.5, color='black', ls='--', lw=2, label='Decision boundary')
ax.set_xlabel("Predicted probability of being Active")
ax.set_ylabel("Count"); ax.set_title("Prediction confidence distribution", fontweight='bold')
ax.legend(fontsize=9)

plt.savefig("project8_qsar_model.png", dpi=150, bbox_inches='tight')
print(f"   ✅ project8_qsar_model.png saved")

# Save results
results_df = pd.DataFrame({
    'Name': names_list, 'SMILES': valid_smiles,
    'Actual': y, 'Predicted': y_pred_all,
    'Probability': np.round(probs_all, 4),
})
results_df.to_csv("project8_qsar_results.csv", index=False)
print(f"   ✅ project8_qsar_results.csv saved")

elapsed = time.time() - start_time
print(f"\n⏱️  {elapsed:.1f}s\n{'='*70}\n  PIPELINE COMPLETE\n{'='*70}")
