#project 10

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, rdMolDescriptors
from rdkit.Chem import DataStructs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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
print("  CHEMICAL SPACE VISUALIZATION")
print("  PCA & t-SNE Dimensionality Reduction on Drug Fingerprints")
print("  University of Waterloo | Medicinal Chemistry")
print("=" * 80)

# ============================================================================
# WHAT THIS PROJECT DOES:
#
# A molecular fingerprint is a 2048-dimensional vector. You can't visualize
# 2048 dimensions. So we use DIMENSIONALITY REDUCTION to project molecules
# onto a 2D plot where similar molecules cluster together.
#
# Two techniques:
#   1. PCA (Principal Component Analysis) — linear, fast, finds main axes
#      of variation. Good for first look.
#   2. t-SNE (t-distributed Stochastic Neighbor Embedding) — non-linear,
#      slow but powerful. Reveals tight clusters that PCA misses.
#
# Then we use K-MEANS CLUSTERING to identify natural groupings without
# knowing the labels — and compare against actual drug class labels.
#
# This is a real data science workflow used in pharma "chemical space
# analysis" — used to:
#   - Identify gaps in chemical libraries
#   - Find unexplored regions for new drug discovery
#   - Visualize HTS results
# ============================================================================


def get_drug_dataset():
    """Diverse dataset spanning multiple therapeutic areas and drug classes."""
    drugs = [
        # Kinase inhibitors
        ("Imatinib","CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5","Kinase Inhibitor","Oncology"),
        ("Erlotinib","COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC","Kinase Inhibitor","Oncology"),
        ("Gefitinib","COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4","Kinase Inhibitor","Oncology"),
        ("Sorafenib","CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F","Kinase Inhibitor","Oncology"),
        ("Dasatinib","CC1=NC(=CC(=N1)NC2=CC(=CC=C2)N3CCN(CC3)CCO)NC4=CC5=C(C=C4)C(=NN5)C(=O)NC6CCCCC6","Kinase Inhibitor","Oncology"),
        ("Nilotinib","CC1=C(C=C(C=C1)NC(=O)C2=CC(=C(C=C2)C3=CN=C(N=C3)NC4=CC(=CC=C4)C(F)(F)F)C#N)NC5=NC=CC(=N5)C6=CC=CN=C6","Kinase Inhibitor","Oncology"),
        ("Osimertinib","COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)NC(=O)C=C","Kinase Inhibitor","Oncology"),
        ("Ibrutinib","C=CC(=O)N1CCC(CC1)N2C3=C(C=CC=C3)N=C2C4=CC=C(C=C4)OC5=CC=CC=C5","Kinase Inhibitor","Oncology"),
        ("Palbociclib","CC(=O)C1=C(C=C2CN=C(N=C2N1)NC3=NC=C(C=C3)N4CCNCC4)C","Kinase Inhibitor","Oncology"),
        ("Ribociclib","CN(C)C(=O)C1=CC2=CN=C(N=C2N1C3CCCC3)NC4=NC=C(C=C4)N5CCNCC5","Kinase Inhibitor","Oncology"),
        ("Crizotinib","CC(C1=C(C=CC(=C1Cl)F)Cl)OC2=CN=C3C=C(C=CC3=C2)C4(CC4)N","Kinase Inhibitor","Oncology"),
        ("Vemurafenib","CCCS(=O)(=O)NC1=CC(=C(C=C1F)C(=O)C2=CNC3=NC=C(C=C23)C4=CC=C(C=C4)Cl)F","Kinase Inhibitor","Oncology"),
        ("Lapatinib","CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)N=CN=C3NC4=CC(=C(C=C4)Cl)OCC5=CC(=CC=C5)F","Kinase Inhibitor","Oncology"),
        ("Sunitinib","CCN(CC)CCNC(=O)C1=C(C(=C(S1)/C=C\\2/C3=CC=CC=C3NC2=O)C)C","Kinase Inhibitor","Oncology"),
        ("Bosutinib","COC1=C(C=C2C(=C1OC)N=CN=C2NC3=CC(=C(C=C3Cl)Cl)OC)OC4=CC(=CC=C4)CN5CCN(CC5)C","Kinase Inhibitor","Oncology"),
        ("Ruxolitinib","N#CCC(C1=CC=CN=C1)N2CC3=C(C2)C(=NN3)C4CCCC4","Kinase Inhibitor","Oncology"),
        ("Abemaciclib","CCN1C(=NC2=C1N=C(N=C2NC3=CC=C(C=C3)N4CCN(CC4)C)C5=CC(=NC=C5)F)C","Kinase Inhibitor","Oncology"),
        # Statins
        ("Atorvastatin","CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4","Statin","Cardiovascular"),
        ("Rosuvastatin","CC(C1=NC(=NC(=C1C=CC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)N(C)S(=O)(=O)C)C","Statin","Cardiovascular"),
        ("Simvastatin","CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C","Statin","Cardiovascular"),
        ("Lovastatin","CCC(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C","Statin","Cardiovascular"),
        # NSAIDs
        ("Aspirin","CC(=O)OC1=CC=CC=C1C(=O)O","NSAID","Pain"),
        ("Ibuprofen","CC(C)CC1=CC=C(C=C1)C(C)C(=O)O","NSAID","Pain"),
        ("Naproxen","COC1=CC2=CC(=CC=C2C=C1)C(C)C(=O)O","NSAID","Pain"),
        ("Diclofenac","OC(=O)CC1=CC=CC=C1NC2=C(Cl)C=CC=C2Cl","NSAID","Pain"),
        ("Celecoxib","CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F","NSAID","Pain"),
        ("Indomethacin","CC1=C(C2=CC(=CC=C2N1C(=O)C3=CC=C(C=C3)Cl)OC)CC(=O)O","NSAID","Pain"),
        # SSRIs
        ("Fluoxetine","CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F","SSRI","CNS"),
        ("Sertraline","CNC1CCC(C2=CC=CC=C21)C3=CC(=C(C=C3)Cl)Cl","SSRI","CNS"),
        ("Paroxetine","C1CC(OC2=CC3=C(C=C2)OCO3)C(C1COC4=CC=C(C=C4)F)NC","SSRI","CNS"),
        ("Citalopram","CN(C)CCCC1(C2=CC=C(C=C2)F)OCC3=CC(=CC=C31)C#N","SSRI","CNS"),
        ("Venlafaxine","COC1=CC=C(C=C1)C(CN(C)C)C2(CCCCC2)O","SNRI","CNS"),
        ("Duloxetine","CNCC(C1=CC=CS1)OC2=CC=C3C=CC=CC3=C2","SNRI","CNS"),
        # Antibiotics
        ("Ciprofloxacin","C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O","Fluoroquinolone","Anti-infective"),
        ("Levofloxacin","CC1COC2=C3N1C=C(C(=O)C3=CC(=C2N4CCN(CC4)C)F)C(=O)O","Fluoroquinolone","Anti-infective"),
        ("Amoxicillin","CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C","Beta-Lactam","Anti-infective"),
        # PPIs
        ("Omeprazole","CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2","PPI","GI"),
        ("Lansoprazole","CC1=C(C=CN=C1CS(=O)C2=NC3=CC=CC=C3N2)OCC(F)(F)F","PPI","GI"),
        # Cardiovascular
        ("Losartan","CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl","ARB","Cardiovascular"),
        ("Amlodipine","CCOC(=O)C1=C(NC(=C(C1C2=CC=CC=C2Cl)C(=O)OC)C)COCCN","CCB","Cardiovascular"),
        ("Warfarin","CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O","Anticoagulant","Cardiovascular"),
        # Metabolic
        ("Metformin","CN(C)C(=N)NC(=N)N","Biguanide","Metabolic"),
        ("Sitagliptin","C1CN2C(=NN=C2C(F)(F)F)CN1C(=O)CC(CC3=CC(=C(C=C3F)F)F)N","DPP-4","Metabolic"),
        # Other
        ("Sildenafil","CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C","PDE5","Urology"),
        ("Caffeine","CN1C=NC2=C1C(=O)N(C(=O)N2C)C","Methylxanthine","OTC"),
        ("Acetaminophen","CC(=O)NC1=CC=C(C=C1)O","Analgesic","Pain"),
        ("Donepezil","COC1=CC2=C(C=C1OC)C(=O)C(C2)CC3CCN(CC3)CC4=CC=CC=C4","AChE","CNS"),
        # PROTACs (different chemical space)
        ("ARV-471","C1CCC(CC1)C(=O)NCCOCCOCCNC(=O)C2=CC=C(C=C2)F","PROTAC","Oncology"),
        ("ARV-110","CC1(CCC(=C(C1)C2=CC=C(C=C2)CN3CCN(CC3)C(=O)C4CC4)C5=CC(=CC=C5)C(F)(F)F)C","PROTAC","Oncology"),
        # Molecular glues
        ("Lenalidomide","C1CC(=O)NC(=O)C1N2CC3=CC=CC=C3C2=O","Molecular Glue","Oncology"),
        ("Pomalidomide","C1CC(=O)NC(=O)C1N2CC3=CC(=CC=C3C2=O)N","Molecular Glue","Oncology"),
        ("Thalidomide","O=C1CCC(N1)C(=O)N2C(=O)C3=CC=CC=C3C2=O","Molecular Glue","Oncology"),
    ]
    return [{"name":n,"smiles":s,"class":c,"area":a} for n,s,c,a in drugs]


# ============================================================================
# COMPUTE FINGERPRINTS AND DESCRIPTORS
# ============================================================================
print(f"\n📦 Loading dataset...")
drugs = get_drug_dataset()
print(f"   {len(drugs)} drugs across {len(set(d['class'] for d in drugs))} classes")

print(f"\n🔧 Computing Morgan fingerprints (ECFP4, 2048-bit)...")
fps = []
desc_array = []
valid_drugs = []

for d in drugs:
    mol = Chem.MolFromSmiles(d["smiles"])
    if mol is None: continue
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros((2048,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fps.append(arr)
    desc_array.append([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol), Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol), Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol), Descriptors.FractionCSP3(mol),
    ])
    valid_drugs.append(d)

X_fp = np.array(fps)
X_desc = np.array(desc_array)
print(f"   ✅ Fingerprint matrix: {X_fp.shape}")
print(f"   ✅ Descriptor matrix: {X_desc.shape}")


# ============================================================================
# DIMENSIONALITY REDUCTION
# ============================================================================
print(f"\n🧮 Running PCA on fingerprints...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_fp)
print(f"   PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
print(f"   PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")
print(f"   Total: {sum(pca.explained_variance_ratio_)*100:.1f}%")

print(f"\n🧮 Running t-SNE on fingerprints (non-linear, slower)...")
perplexity = min(15, len(X_fp)-1)
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
X_tsne = tsne.fit_transform(X_fp)
print(f"   ✅ t-SNE complete (perplexity={perplexity})")


# ============================================================================
# K-MEANS CLUSTERING (unsupervised — find clusters without knowing labels)
# ============================================================================
print(f"\n🎯 Running K-Means clustering (k=6, unsupervised)...")
n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_fp)

# Examine cluster composition
print(f"\n📋 CLUSTER COMPOSITION (does K-Means recover real drug classes?)")
df = pd.DataFrame({
    "Name": [d["name"] for d in valid_drugs],
    "Class": [d["class"] for d in valid_drugs],
    "Area": [d["area"] for d in valid_drugs],
    "Cluster": clusters,
    "PC1": X_pca[:,0], "PC2": X_pca[:,1],
    "tSNE1": X_tsne[:,0], "tSNE2": X_tsne[:,1],
})

for c in range(n_clusters):
    members = df[df['Cluster']==c]
    classes_in_cluster = members['Class'].value_counts()
    print(f"\n   Cluster {c} ({len(members)} compounds):")
    for cls, count in classes_in_cluster.items():
        print(f"      {cls}: {count}")
    # Show examples
    examples = members['Name'].tolist()[:5]
    print(f"      Examples: {', '.join(examples)}")


# ============================================================================
# CLUSTERING QUALITY: Are similar drugs grouped together?
# ============================================================================
print(f"\n📊 CLUSTERING QUALITY METRICS")
# Purity: most common class in each cluster / cluster size
total_correct = 0
for c in range(n_clusters):
    members = df[df['Cluster']==c]
    if len(members) == 0: continue
    most_common_count = members['Class'].value_counts().iloc[0]
    total_correct += most_common_count
purity = total_correct / len(df)
print(f"   Cluster purity: {purity:.3f} ({total_correct}/{len(df)} compounds in correct class majority)")


# ============================================================================
# VISUALIZATIONS
# ============================================================================
print(f"\n📊 Generating visualizations...")
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20, 18)); gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
fig.suptitle(f"Chemical Space Visualization — {len(df)} Drugs\nPCA & t-SNE Dimensionality Reduction on 2048-bit Fingerprints\nUniversity of Waterloo | Medicinal Chemistry",
             fontsize=16, fontweight='bold', y=0.99)

# Color palette for drug classes
classes = df['Class'].unique()
class_colors = dict(zip(classes, plt.cm.tab20(np.linspace(0, 1, len(classes)))))

# Panel 1: PCA colored by class
ax = fig.add_subplot(gs[0, 0])
for cls in classes:
    sub = df[df['Class']==cls]
    ax.scatter(sub['PC1'], sub['PC2'], c=[class_colors[cls]]*len(sub), s=100, alpha=0.7,
              edgecolors='white', linewidth=1, label=cls)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA on fingerprints (colored by drug class)", fontweight='bold')
ax.legend(fontsize=6, loc='upper right', ncol=2)

# Panel 2: t-SNE colored by class
ax = fig.add_subplot(gs[0, 1])
for cls in classes:
    sub = df[df['Class']==cls]
    ax.scatter(sub['tSNE1'], sub['tSNE2'], c=[class_colors[cls]]*len(sub), s=100, alpha=0.7,
              edgecolors='white', linewidth=1, label=cls)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.set_title("t-SNE on fingerprints (colored by drug class)", fontweight='bold')
ax.legend(fontsize=6, loc='upper right', ncol=2)

# Panel 3: PCA colored by K-Means cluster
ax = fig.add_subplot(gs[1, 0])
cluster_colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
for c in range(n_clusters):
    sub = df[df['Cluster']==c]
    ax.scatter(sub['PC1'], sub['PC2'], c=[cluster_colors[c]]*len(sub), s=100, alpha=0.7,
              edgecolors='white', linewidth=1, label=f'Cluster {c}')
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title(f"PCA colored by K-Means cluster (purity: {purity:.2f})", fontweight='bold')
ax.legend(fontsize=8)

# Panel 4: t-SNE colored by K-Means cluster
ax = fig.add_subplot(gs[1, 1])
for c in range(n_clusters):
    sub = df[df['Cluster']==c]
    ax.scatter(sub['tSNE1'], sub['tSNE2'], c=[cluster_colors[c]]*len(sub), s=100, alpha=0.7,
              edgecolors='white', linewidth=1, label=f'Cluster {c}')
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.set_title("t-SNE colored by K-Means cluster", fontweight='bold')
ax.legend(fontsize=8)

# Panel 5: t-SNE with drug names (most informative view)
ax = fig.add_subplot(gs[2, 0])
for cls in classes:
    sub = df[df['Class']==cls]
    ax.scatter(sub['tSNE1'], sub['tSNE2'], c=[class_colors[cls]]*len(sub), s=120, alpha=0.7,
              edgecolors='white', linewidth=1.5)
    for _, r in sub.iterrows():
        ax.annotate(r['Name'][:10], (r['tSNE1'], r['tSNE2']), fontsize=6, alpha=0.7,
                   ha='center', va='bottom', fontweight='bold')
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.set_title("t-SNE with drug names", fontweight='bold')

# Panel 6: Cluster size distribution
ax = fig.add_subplot(gs[2, 1])
cluster_sizes = df['Cluster'].value_counts().sort_index()
bars = ax.bar(cluster_sizes.index, cluster_sizes.values,
              color=[cluster_colors[c] for c in cluster_sizes.index],
              edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, cluster_sizes.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
           str(val), ha='center', fontweight='bold')
ax.set_xlabel("Cluster"); ax.set_ylabel("Number of compounds")
ax.set_title("K-Means cluster sizes", fontweight='bold')

plt.savefig("project10_chemical_space.png", dpi=150, bbox_inches='tight')
print(f"   ✅ project10_chemical_space.png saved")
df.to_csv("project10_chemical_space_results.csv", index=False)
print(f"   ✅ project10_chemical_space_results.csv saved")

elapsed = time.time() - start_time
print(f"\n⏱️  {elapsed:.1f}s\n{'='*70}\n  PIPELINE COMPLETE\n{'='*70}")
