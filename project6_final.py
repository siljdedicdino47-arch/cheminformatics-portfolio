#project 6

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, rdMolDescriptors
from rdkit.Chem import rdFMCS, DataStructs
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
print("  MATCHED MOLECULAR PAIR (MMP) ANALYSIS")
print("  Quantifying How Structural Changes Affect Drug Properties")
print("  University of Waterloo | Medicinal Chemistry")
print("=" * 80)

# ============================================================================
# WHAT THIS PROJECT DOES:
# Medicinal chemists optimize drugs by making small structural changes.
# An MMP is two molecules differing by ONE transformation. By analyzing
# many MMPs, we learn rules like: "Adding fluorine typically changes LogP
# by X." This mirrors the J. Med. Chem. 2026 paper "What Happens in
# Successful Optimizations?" by Paul Leeson.
# ============================================================================

pairs = [
    {"mol_a":{"name":"Gefitinib","smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"},
     "mol_b":{"name":"Erlotinib","smiles":"COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"},
     "transformation":"Morpholine→ethynyl substituents","series":"EGFR"},
    {"mol_a":{"name":"Erlotinib","smiles":"COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC"},
     "mol_b":{"name":"Osimertinib","smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)NC(=O)C=C"},
     "transformation":"Added covalent acrylamide warhead","series":"EGFR"},
    {"mol_a":{"name":"Imatinib","smiles":"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"},
     "mol_b":{"name":"Nilotinib","smiles":"CC1=C(C=C(C=C1)NC(=O)C2=CC(=C(C=C2)C3=CN=C(N=C3)NC4=CC(=CC=C4)C(F)(F)F)C#N)NC5=NC=CC(=N5)C6=CC=CN=C6"},
     "transformation":"Added CF3 and CN for potency","series":"BCR-ABL"},
    {"mol_a":{"name":"Imatinib","smiles":"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5"},
     "mol_b":{"name":"Dasatinib","smiles":"CC1=NC(=CC(=N1)NC2=CC(=CC=C2)N3CCN(CC3)CCO)NC4=CC5=C(C=C4)C(=NN5)C(=O)NC6CCCCC6"},
     "transformation":"Scaffold hop to aminothiazole","series":"BCR-ABL"},
    {"mol_a":{"name":"Palbociclib","smiles":"CC(=O)C1=C(C=C2CN=C(N=C2N1)NC3=NC=C(C=C3)N4CCNCC4)C"},
     "mol_b":{"name":"Abemaciclib","smiles":"CCN1C(=NC2=C1N=C(N=C2NC3=CC=C(C=C3)N4CCN(CC4)C)C5=CC(=NC=C5)F)C"},
     "transformation":"Core modification, added fluoropyridine","series":"CDK4/6"},
    {"mol_a":{"name":"Aspirin","smiles":"CC(=O)OC1=CC=CC=C1C(=O)O"},
     "mol_b":{"name":"Ibuprofen","smiles":"CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"},
     "transformation":"Salicylate→propionic acid scaffold","series":"NSAID"},
    {"mol_a":{"name":"Ibuprofen","smiles":"CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"},
     "mol_b":{"name":"Naproxen","smiles":"COC1=CC2=CC(=CC=C2C=C1)C(C)C(=O)O"},
     "transformation":"Phenyl→naphthyl, added methoxy","series":"NSAID"},
    {"mol_a":{"name":"Omeprazole","smiles":"CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2"},
     "mol_b":{"name":"Lansoprazole","smiles":"CC1=C(C=CN=C1CS(=O)C2=NC3=CC=CC=C3N2)OCC(F)(F)F"},
     "transformation":"Methyl/methoxy→trifluoroethoxy","series":"PPI"},
    {"mol_a":{"name":"Fluoxetine","smiles":"CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F"},
     "mol_b":{"name":"Sertraline","smiles":"CNC1CCC(C2=CC=CC=C21)C3=CC(=C(C=C3)Cl)Cl"},
     "transformation":"Open chain→fused ring, CF3→dichloro","series":"SSRI"},
    {"mol_a":{"name":"Atorvastatin","smiles":"CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4"},
     "mol_b":{"name":"Rosuvastatin","smiles":"CC(C1=NC(=NC(=C1C=CC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)N(C)S(=O)(=O)C)C"},
     "transformation":"Pyrrole→pyrimidine, added sulfonamide","series":"Statin"},
    {"mol_a":{"name":"Ibrutinib","smiles":"C=CC(=O)N1CCC(CC1)N2C3=C(C=CC=C3)N=C2C4=CC=C(C=C4)OC5=CC=CC=C5"},
     "mol_b":{"name":"Acalabrutinib","smiles":"C=CC(=O)N1CCC(CC1)N2C3=C(N=C2C(N)=O)C=CC=C3"},
     "transformation":"Removed phenoxyphenyl, added amide","series":"BTK"},
    {"mol_a":{"name":"Ibrutinib","smiles":"C=CC(=O)N1CCC(CC1)N2C3=C(C=CC=C3)N=C2C4=CC=C(C=C4)OC5=CC=CC=C5"},
     "mol_b":{"name":"Zanubrutinib","smiles":"C=CC(=O)N1CCC2(CC1)CN(C2)C3=NC(=NC4=CC=CC=C34)NC5=CC=C(C=C5)OC6=CC=CC=C6"},
     "transformation":"Added spiro ring for improved PK","series":"BTK"},
    {"mol_a":{"name":"Losartan","smiles":"CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl"},
     "mol_b":{"name":"Valsartan","smiles":"CCCCC(=O)N(CC1=CC=C(C=C1)C2=CC=CC=C2C3=NNN=N3)C(C(C)C)C(=O)O"},
     "transformation":"Imidazole→valine-based, different binding mode","series":"ARB"},
]

def compute_props(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    return {"MW":round(Descriptors.MolWt(mol),1),"LogP":round(Descriptors.MolLogP(mol),2),
            "TPSA":round(Descriptors.TPSA(mol),1),"HBD":Lipinski.NumHDonors(mol),
            "HBA":Lipinski.NumHAcceptors(mol),"RotBonds":Descriptors.NumRotatableBonds(mol),
            "Fsp3":round(Descriptors.FractionCSP3(mol),3),"AromaticRings":Descriptors.NumAromaticRings(mol),
            "HeavyAtoms":mol.GetNumHeavyAtoms()}

def compute_similarity(sa, sb):
    ma, mb = Chem.MolFromSmiles(sa), Chem.MolFromSmiles(sb)
    if ma is None or mb is None: return 0
    fa = AllChem.GetMorganFingerprintAsBitVect(ma,2,nBits=2048)
    fb = AllChem.GetMorganFingerprintAsBitVect(mb,2,nBits=2048)
    return round(DataStructs.TanimotoSimilarity(fa,fb),4)

print(f"\n📦 Analyzing {len(pairs)} molecular pairs...")
results = []
for pair in pairs:
    pa = compute_props(pair["mol_a"]["smiles"])
    pb = compute_props(pair["mol_b"]["smiles"])
    if pa is None or pb is None: continue
    sim = compute_similarity(pair["mol_a"]["smiles"], pair["mol_b"]["smiles"])
    row = {"Mol_A":pair["mol_a"]["name"],"Mol_B":pair["mol_b"]["name"],
           "Transformation":pair["transformation"],"Series":pair["series"],"Tanimoto":sim}
    for prop in pa:
        row[f"A_{prop}"] = pa[prop]; row[f"B_{prop}"] = pb[prop]
        row[f"Δ{prop}"] = round(pb[prop]-pa[prop],2)
    results.append(row)

df = pd.DataFrame(results)
print(f"   ✅ {len(df)} pairs analyzed")

# Results
print(f"\n{'='*90}\n  RESULTS\n{'='*90}")
delta_props = ["ΔMW","ΔLogP","ΔTPSA","ΔHBD","ΔHBA","ΔRotBonds","ΔFsp3"]
for _,row in df.iterrows():
    print(f"\n  {row['Mol_A']} → {row['Mol_B']}  (Tanimoto: {row['Tanimoto']:.3f})")
    print(f"    {row['Transformation']}")
    changes = [f"{dp[1:]} {'↑' if row[dp]>0 else '↓'}{abs(row[dp]):.1f}" for dp in delta_props if row[dp]!=0]
    print(f"    Deltas: {', '.join(changes[:5])}")

print(f"\n{'='*90}\n  SUMMARY STATISTICS\n{'='*90}")
print(f"\n{'Property':<12} {'Mean Δ':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-"*50)
for dp in delta_props:
    print(f"   {dp:<10} {df[dp].mean():>8.1f} {df[dp].std():>8.1f} {df[dp].min():>8.1f} {df[dp].max():>8.1f}")

# Visualizations
print(f"\n📊 Generating visualizations...")
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20,18)); gs = GridSpec(3,2,figure=fig,hspace=0.4,wspace=0.3)
fig.suptitle(f"Matched Molecular Pair Analysis — {len(df)} Drug Pairs\nUniversity of Waterloo | Medicinal Chemistry",fontsize=17,fontweight='bold',y=0.98)
sc = {"EGFR":"#e74c3c","BCR-ABL":"#3498db","CDK4/6":"#9b59b6","NSAID":"#e67e22","PPI":"#1abc9c","SSRI":"#16a085","Statin":"#f39c12","BTK":"#d35400","ARB":"#2ecc71"}

ax=fig.add_subplot(gs[0,0])
for s in df['Series'].unique():
    sub=df[df['Series']==s]
    ax.scatter(sub['ΔMW'],sub['ΔLogP'],c=sc.get(s,'#95a5a6'),s=100,alpha=0.8,edgecolors='white',linewidth=1.5,label=s,zorder=5)
ax.axhline(0,color='gray',ls='-',alpha=0.3);ax.axvline(0,color='gray',ls='-',alpha=0.3)
ax.set_xlabel("ΔMW (Da)");ax.set_ylabel("ΔLogP");ax.set_title("Property changes: ΔMW vs ΔLogP",fontweight='bold');ax.legend(fontsize=7,ncol=2)

ax=fig.add_subplot(gs[0,1])
x=np.arange(len(delta_props[:4]));means=[df[p].mean() for p in delta_props[:4]];stds=[df[p].std() for p in delta_props[:4]]
colors_bar=['#3498db' if m>=0 else '#e74c3c' for m in means]
ax.bar(x,means,yerr=stds,color=colors_bar,edgecolor='white',linewidth=1.5,capsize=5,alpha=0.8)
ax.axhline(0,color='black',ls='-',lw=0.5);ax.set_xticks(x);ax.set_xticklabels(delta_props[:4])
ax.set_ylabel("Mean Δ");ax.set_title("Average property changes",fontweight='bold')

ax=fig.add_subplot(gs[1,0])
colors_sim=[sc.get(s,'#95a5a6') for s in df['Series']]
ax.barh(range(len(df)),df['Tanimoto'],color=colors_sim,edgecolor='white',linewidth=1)
ax.set_yticks(range(len(df)));ax.set_yticklabels([f"{r['Mol_A'][:7]}→{r['Mol_B'][:7]}" for _,r in df.iterrows()],fontsize=7)
ax.axvline(0.7,color='red',ls='--',alpha=0.4);ax.set_xlabel("Tanimoto");ax.set_title("Pair similarity",fontweight='bold')

ax=fig.add_subplot(gs[1,1])
hm=df[delta_props].copy()
for c in delta_props: mx=max(hm[c].abs().max(),0.01);hm[c]=hm[c]/mx
im=ax.imshow(hm.values,cmap='RdBu_r',aspect='auto',vmin=-1,vmax=1)
ax.set_xticks(range(len(delta_props)));ax.set_xticklabels(delta_props,fontsize=8,rotation=45,ha='right')
ax.set_yticks(range(len(df)));ax.set_yticklabels([f"{r['Mol_A'][:7]}→{r['Mol_B'][:7]}" for _,r in df.iterrows()],fontsize=7)
plt.colorbar(im,ax=ax,label='Normalized Δ',shrink=0.8);ax.set_title("Property change heatmap",fontweight='bold')

ax=fig.add_subplot(gs[2,0])
ax.scatter(df['ΔMW'],df['ΔTPSA'],c=[sc.get(s,'#95a5a6') for s in df['Series']],s=100,alpha=0.8,edgecolors='white',linewidth=1.5)
if len(df)>2:
    z=np.polyfit(df['ΔMW'],df['ΔTPSA'],1);p=np.poly1d(z);xl=np.linspace(df['ΔMW'].min()-10,df['ΔMW'].max()+10,100)
    ax.plot(xl,p(xl),'--',color='red',alpha=0.5,label=f'slope={z[0]:.2f}')
ax.axhline(0,color='gray',ls='-',alpha=0.3);ax.axvline(0,color='gray',ls='-',alpha=0.3)
ax.set_xlabel("ΔMW");ax.set_ylabel("ΔTPSA");ax.set_title("ΔMW vs ΔTPSA correlation",fontweight='bold');ax.legend(fontsize=9)

ax=fig.add_subplot(gs[2,1])
series_means=df.groupby('Series')['Tanimoto'].mean().sort_values()
ax.barh(range(len(series_means)),series_means.values,color=[sc.get(s,'#95a5a6') for s in series_means.index],edgecolor='white',linewidth=1)
ax.set_yticks(range(len(series_means)));ax.set_yticklabels(series_means.index)
ax.set_xlabel("Mean Tanimoto");ax.set_title("Avg similarity by drug series",fontweight='bold')

plt.savefig("project6_mmp_analysis.png",dpi=150,bbox_inches='tight')
print(f"   ✅ project6_mmp_analysis.png saved")
df.to_csv("project6_mmp_results.csv",index=False)
elapsed=time.time()-start_time
print(f"\n⏱️  {elapsed:.1f}s\n{'='*70}\n  PIPELINE COMPLETE\n{'='*70}")
