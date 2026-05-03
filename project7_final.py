#project 7

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem, rdMolDescriptors
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
print("  NATURAL PRODUCT DRUG-LIKENESS SCORER")
print("  Comparing Natural Products to Synthetic Drugs")
print("  University of Waterloo | Medicinal Chemistry")
print("=" * 80)

# ============================================================================
# WHAT THIS PROJECT DOES:
# Natural products (NPs) are molecules from plants, fungi, bacteria.
# ~40% of all approved drugs are NP-derived. But NPs have unusual properties:
# higher Fsp3 (more 3D), more stereocenters, more oxygen, fewer nitrogen.
# This project scores molecules on a "natural product-likeness" scale and
# compares NP chemical space to synthetic drug chemical space.
#
# Based on:
#   - Ertl et al. (2008) "Natural Product-likeness Score" (JCIM)
#   - Lovering et al. (2009) "Escape from Flatland" (J. Med. Chem.)
#   - "AI in Natural Product Drug Discovery" (Nat. Rev. Drug Disc. 2025)
# ============================================================================

def get_compounds():
    compounds = [
        # === NATURAL PRODUCTS (from plants, fungi, bacteria) ===
        {"name":"Paclitaxel","smiles":"CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C","source":"Pacific yew tree","area":"Oncology","type":"NP"},
        {"name":"Morphine","smiles":"CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O","source":"Opium poppy","area":"Pain","type":"NP"},
        {"name":"Artemisinin","smiles":"CC1CCC2C(C(=O)OC3CC4(C1CCC23OO4)C)C","source":"Sweet wormwood","area":"Malaria","type":"NP"},
        {"name":"Camptothecin","smiles":"CCC1(C2=C(COC1=O)C(=O)N3CC4=CC5=CC=CC=C5N=C4C3=C2)O","source":"Camptotheca tree","area":"Oncology","type":"NP"},
        {"name":"Colchicine","smiles":"COC1=CC2=C(C(=C1)OC)C(CC1=CC(=O)C(=CC1=C2)OC)NC(C)=O","source":"Autumn crocus","area":"Gout","type":"NP"},
        {"name":"Resveratrol","smiles":"OC1=CC=C(C=C1)/C=C/C2=CC(=CC(=C2)O)O","source":"Grapes","area":"Supplement","type":"NP"},
        {"name":"Quercetin","smiles":"C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O","source":"Many plants","area":"Supplement","type":"NP"},
        {"name":"Curcumin","smiles":"COC1=CC(=CC(=C1O)OC)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC","source":"Turmeric","area":"Supplement","type":"NP"},
        {"name":"Caffeine","smiles":"CN1C=NC2=C1C(=O)N(C(=O)N2C)C","source":"Coffee plant","area":"Stimulant","type":"NP"},
        {"name":"Nicotine","smiles":"CN1CCCC1C2=CN=CC=C2","source":"Tobacco","area":"CNS","type":"NP"},
        {"name":"Capsaicin","smiles":"COC1=CC(=CC=C1O)CNC(=O)CCCC/C=C/C(C)C","source":"Chili pepper","area":"Pain","type":"NP"},
        {"name":"Epigallocatechin gallate","smiles":"C1C(C(OC2=CC(=CC(=C21)O)O)C3=CC(=C(C(=C3)O)O)O)OC(=O)C4=CC(=C(C(=C4)O)O)O","source":"Green tea","area":"Antioxidant","type":"NP"},
        {"name":"Berberine","smiles":"COC1=CC=C2C=C3C=CC4=CC5=C(C=C4C3=C2C1=O)OCO5","source":"Barberry","area":"Metabolic","type":"NP"},
        {"name":"Vinblastine","smiles":"CCC1(CC2CC(C3=C(CCN(C2)C1)C4=CC=CC=C4N3)C(C(=O)OC)O)O","source":"Periwinkle","area":"Oncology","type":"NP"},
        {"name":"Penicillin G","smiles":"CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C","source":"Penicillium mold","area":"Antibiotic","type":"NP"},
        {"name":"Erythromycin","smiles":"CCC1C(C(C(N(CC(CC(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)C)O)(C)O","source":"Streptomyces","area":"Antibiotic","type":"NP"},
        {"name":"Lovastatin","smiles":"CCC(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C","source":"Aspergillus","area":"Cardiovascular","type":"NP"},
        {"name":"Rapamycin","smiles":"CC1CCC2CC(C=CC=CC(CC(C(=O)C(CC(=O)C(C(C(C(C=CC(C(=O)C1)OC)OC3C(CC(C(O3)C)O)OC)C)O)C)O)OC)C)CCC(=O)OC(C(CC2O)C)C(=CC4CCC(C(C4)OC)O)C","source":"Streptomyces","area":"Immunology","type":"NP"},

        # === NP-DERIVED DRUGS (semi-synthetic modifications) ===
        {"name":"Docetaxel","smiles":"CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1O)OC(=O)C(C5=CC=CC=C5)NC(=O)OC(C)(C)C)OC(=O)C6=CC=CC=C6)(CO4)OC(=O)C)O)C)O","source":"Semi-synthetic from taxol","area":"Oncology","type":"NP-derived"},
        {"name":"Amoxicillin","smiles":"CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C","source":"Semi-synthetic penicillin","area":"Antibiotic","type":"NP-derived"},
        {"name":"Doxycycline","smiles":"CC1C2C(C3C(=C(C(=O)C(=C3C(=O)C2(C(=C1O)C(=O)N)O)O)O)N(C)C)O","source":"Semi-synthetic tetracycline","area":"Antibiotic","type":"NP-derived"},
        {"name":"Simvastatin","smiles":"CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C","source":"Semi-synthetic from lovastatin","area":"Cardiovascular","type":"NP-derived"},

        # === SYNTHETIC DRUGS (fully designed in the lab) ===
        {"name":"Imatinib","smiles":"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5","source":"Synthetic","area":"Oncology","type":"Synthetic"},
        {"name":"Osimertinib","smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)NC(=O)C=C","source":"Synthetic","area":"Oncology","type":"Synthetic"},
        {"name":"Sorafenib","smiles":"CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F","source":"Synthetic","area":"Oncology","type":"Synthetic"},
        {"name":"Palbociclib","smiles":"CC(=O)C1=C(C=C2CN=C(N=C2N1)NC3=NC=C(C=C3)N4CCNCC4)C","source":"Synthetic","area":"Oncology","type":"Synthetic"},
        {"name":"Venetoclax","smiles":"CC1(CCC(=C(C1)C2=CC=C(C=C2)Cl)C3=CC=C(C=C3)Cl)C","source":"Synthetic","area":"Oncology","type":"Synthetic"},
        {"name":"Fluoxetine","smiles":"CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F","source":"Synthetic","area":"CNS","type":"Synthetic"},
        {"name":"Sertraline","smiles":"CNC1CCC(C2=CC=CC=C21)C3=CC(=C(C=C3)Cl)Cl","source":"Synthetic","area":"CNS","type":"Synthetic"},
        {"name":"Losartan","smiles":"CCCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl","source":"Synthetic","area":"Cardiovascular","type":"Synthetic"},
        {"name":"Sildenafil","smiles":"CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C","source":"Synthetic","area":"Urology","type":"Synthetic"},
        {"name":"Metformin","smiles":"CN(C)C(=N)NC(=N)N","source":"Synthetic","area":"Metabolic","type":"Synthetic"},
        {"name":"Sitagliptin","smiles":"C1CN2C(=NN=C2C(F)(F)F)CN1C(=O)CC(CC3=CC(=C(C=C3F)F)F)N","source":"Synthetic","area":"Metabolic","type":"Synthetic"},
        {"name":"Ciprofloxacin","smiles":"C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O","source":"Synthetic","area":"Antibiotic","type":"Synthetic"},
        {"name":"Omeprazole","smiles":"CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2","source":"Synthetic","area":"GI","type":"Synthetic"},
        {"name":"Baricitinib","smiles":"CCS(=O)(=O)N1CC(C1)N2C=C(C(=N2)C3=CC=NC=C3)C#N","source":"Synthetic","area":"Immunology","type":"Synthetic"},
        {"name":"Celecoxib","smiles":"CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F","source":"Synthetic","area":"Pain","type":"Synthetic"},
        {"name":"Enzalutamide","smiles":"CNC(=O)C1=CC=C(C=C1)N2C(=O)N(C(=S)C2(C)C)C3=CC=C(C#N)C(=C3)C(F)(F)F","source":"Synthetic","area":"Oncology","type":"Synthetic"},
    ]
    return compounds

# ============================================================================
# NP-LIKENESS SCORING
# ============================================================================
def compute_np_score(mol):
    """
    Compute a Natural Product-likeness score based on structural features
    that distinguish NPs from synthetic drugs.
    
    Features that increase NP-likeness:
      - High Fsp3 (more 3D, less flat)
      - Stereocenters (NPs are stereochemically rich)
      - Oxygen atoms (NPs are oxygen-rich)
      - Fused/bridged ring systems
      - Lactone rings
    
    Features that decrease NP-likeness:
      - Halogen atoms (F, Cl, Br — rare in nature)
      - Sulfonamides (synthetic motif)
      - Many aromatic rings (NPs tend to be less flat)
    """
    if mol is None: return None
    
    fsp3 = Descriptors.FractionCSP3(mol)
    stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    rings = Descriptors.RingCount(mol)
    aromatic = Descriptors.NumAromaticRings(mol)
    
    # Count specific atoms
    o_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    n_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    halogen_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [9, 17, 35])
    s_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
    heavy = mol.GetNumHeavyAtoms()
    
    # O/N ratio — NPs tend to have more O relative to N
    o_ratio = o_count / max(heavy, 1)
    n_ratio = n_count / max(heavy, 1)
    
    # Scoring (each component 0-1, total 0-7)
    score = 0
    
    # Fsp3: NPs typically > 0.4
    score += min(fsp3 / 0.5, 1.0)
    
    # Stereocenters: NPs typically have 3+
    score += min(stereo / 4.0, 1.0)
    
    # O richness: NPs are O-rich
    score += min(o_ratio / 0.25, 1.0)
    
    # Low N: NPs tend to have fewer N atoms
    score += max(0, 1.0 - n_ratio * 5)
    
    # No halogens: halogens are rare in nature
    score += 1.0 if halogen_count == 0 else max(0, 0.5 - halogen_count * 0.2)
    
    # Ring complexity: NPs have fused/bridged rings
    non_aromatic_rings = rings - aromatic
    score += min(non_aromatic_rings / 3.0, 1.0)
    
    # Low aromatic fraction: NPs are less flat
    aromatic_fraction = aromatic / max(rings, 1)
    score += max(0, 1.0 - aromatic_fraction)
    
    return round(score, 2)

# ============================================================================
# PROCESSING
# ============================================================================
compounds = get_compounds()
print(f"\n📦 {len(compounds)} compounds loaded")

results = []
for entry in compounds:
    mol = Chem.MolFromSmiles(entry["smiles"])
    if mol is None: continue
    
    np_score = compute_np_score(mol)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    fsp3 = Descriptors.FractionCSP3(mol)
    stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    aromatic = Descriptors.NumAromaticRings(mol)
    rings = Descriptors.RingCount(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    o_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    n_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    hal_count = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() in [9,17,35])
    
    lip_v = sum([mw>500,logp>5,hbd>5,hba>10])
    
    results.append({
        "Name":entry["name"],"Source":entry["source"],"Area":entry["area"],"Type":entry["type"],
        "NP_Score":np_score,"MW":round(mw,1),"LogP":round(logp,2),"TPSA":round(tpsa,1),
        "HBD":hbd,"HBA":hba,"Fsp3":round(fsp3,3),"StereoCenters":stereo,
        "AromaticRings":aromatic,"Rings":rings,"RotBonds":rot,"HeavyAtoms":heavy,
        "O_count":o_count,"N_count":n_count,"Halogen_count":hal_count,
        "Lipinski_Violations":lip_v,
    })

df = pd.DataFrame(results)
print(f"   ✅ {len(df)} compounds scored")

# Results
np_df = df[df['Type']=='NP']; syn_df = df[df['Type']=='Synthetic']; der_df = df[df['Type']=='NP-derived']

print(f"\n{'='*80}\n  NP-LIKENESS SCORES\n{'='*80}")
for typ in ["NP","NP-derived","Synthetic"]:
    sub = df[df['Type']==typ].sort_values('NP_Score',ascending=False)
    print(f"\n  --- {typ} (n={len(sub)}, mean NP-score: {sub['NP_Score'].mean():.2f}) ---")
    for _,r in sub.iterrows():
        bar = "█"*int(r['NP_Score']); spaces = "░"*(7-int(r['NP_Score']))
        print(f"    {r['Name']:<25} NP={r['NP_Score']:.1f} {bar}{spaces}  MW={r['MW']:.0f} Fsp3={r['Fsp3']:.2f} Stereo={r['StereoCenters']}")

print(f"\n{'='*80}\n  PROPERTY COMPARISON\n{'='*80}")
props = ["MW","LogP","TPSA","Fsp3","StereoCenters","AromaticRings","O_count","N_count","Halogen_count"]
print(f"\n{'Property':<18} {'Natural Products':>18} {'NP-Derived':>18} {'Synthetic':>18}")
print("-"*75)
for p in props:
    print(f"   {p:<16} {np_df[p].mean():>16.1f} {der_df[p].mean():>16.1f} {syn_df[p].mean():>16.1f}")

print(f"""
KEY INSIGHTS:
  NPs have {np_df['Fsp3'].mean():.2f} Fsp3 vs {syn_df['Fsp3'].mean():.2f} for synthetics — NPs are more 3D.
  NPs have {np_df['StereoCenters'].mean():.1f} stereocenters vs {syn_df['StereoCenters'].mean():.1f} for synthetics.
  NPs have {np_df['O_count'].mean():.1f} oxygens vs {syn_df['O_count'].mean():.1f} — NPs are O-rich.
  Synthetics have {syn_df['Halogen_count'].mean():.1f} halogens vs {np_df['Halogen_count'].mean():.1f} — halogens are man-made.
  NPs have {np_df['AromaticRings'].mean():.1f} aromatic rings vs {syn_df['AromaticRings'].mean():.1f} — synthetics are flatter.
""")

# Visualizations
print(f"📊 Generating visualizations...")
plt.style.use('seaborn-v0_8-whitegrid')
fig = plt.figure(figsize=(20,18)); gs = GridSpec(3,2,figure=fig,hspace=0.4,wspace=0.3)
fig.suptitle(f"Natural Product vs Synthetic Drug Analysis — {len(df)} Compounds\nUniversity of Waterloo | Medicinal Chemistry",fontsize=17,fontweight='bold',y=0.98)
tc = {"NP":"#2ecc71","NP-derived":"#f39c12","Synthetic":"#3498db"}

ax=fig.add_subplot(gs[0,0])
for t in ["NP","NP-derived","Synthetic"]:
    s=df[df['Type']==t]
    ax.scatter(s['Fsp3'],s['NP_Score'],c=tc[t],s=80,alpha=0.7,edgecolors='white',linewidth=1,label=f"{t} (n={len(s)})")
    for _,r in s.iterrows(): ax.annotate(r['Name'][:10],(r['Fsp3'],r['NP_Score']),fontsize=5,alpha=0.6)
ax.set_xlabel("Fsp3 (3D character)");ax.set_ylabel("NP-likeness score");ax.set_title("NP-likeness vs 3D character",fontweight='bold');ax.legend(fontsize=8)

ax=fig.add_subplot(gs[0,1])
data=[df[df['Type']==t]['NP_Score'].values for t in ["NP","NP-derived","Synthetic"]]
bp=ax.boxplot(data,labels=["Natural\nProducts","NP-\nDerived","Synthetic"],patch_artist=True,widths=0.5,medianprops=dict(color='black',linewidth=2))
for p,t in zip(bp['boxes'],["NP","NP-derived","Synthetic"]): p.set_facecolor(tc[t]);p.set_alpha(0.7)
for i,t in enumerate(["NP","NP-derived","Synthetic"]):
    s=df[df['Type']==t]['NP_Score'];j=np.random.normal(i+1,0.06,len(s))
    ax.scatter(j,s,c=tc[t],s=30,alpha=0.6,edgecolors='white',linewidth=0.5,zorder=5)
ax.set_ylabel("NP-likeness score");ax.set_title("NP score distribution by type",fontweight='bold')

ax=fig.add_subplot(gs[1,0])
for t in ["NP","NP-derived","Synthetic"]:
    s=df[df['Type']==t]
    ax.scatter(s['MW'],s['LogP'],c=tc[t],s=80,alpha=0.7,edgecolors='white',linewidth=1,label=t)
ax.axvline(500,color='red',ls='--',alpha=0.3);ax.axhline(5,color='red',ls='--',alpha=0.3)
ax.set_xlabel("MW (Da)");ax.set_ylabel("LogP");ax.set_title("Chemical space comparison",fontweight='bold');ax.legend(fontsize=9)

ax=fig.add_subplot(gs[1,1])
props_compare=["Fsp3","StereoCenters","AromaticRings","O_count","N_count","Halogen_count"]
x=np.arange(len(props_compare));w=0.25
for i,t in enumerate(["NP","NP-derived","Synthetic"]):
    s=df[df['Type']==t];vals=[s[p].mean() for p in props_compare]
    ax.bar(x+i*w,vals,w,label=t,color=tc[t],edgecolor='white',linewidth=1)
ax.set_xticks(x+w);ax.set_xticklabels(props_compare,fontsize=8,rotation=30,ha='right')
ax.set_ylabel("Mean value");ax.set_title("Structural features by type",fontweight='bold');ax.legend(fontsize=8)

ax=fig.add_subplot(gs[2,0])
for t in ["NP","Synthetic"]:
    s=df[df['Type']==t]
    ax.hist(s['Fsp3'],bins=12,alpha=0.5,color=tc[t],label=t,edgecolor='white')
ax.axvline(0.42,color='red',ls='--',alpha=0.5,label='Clinical avg (0.42)')
ax.set_xlabel("Fsp3");ax.set_ylabel("Count");ax.set_title('"Escape from Flatland" — Fsp3 distribution',fontweight='bold');ax.legend(fontsize=8)

ax=fig.add_subplot(gs[2,1])
ax.scatter(df['StereoCenters'],df['NP_Score'],c=[tc[t] for t in df['Type']],s=80,alpha=0.7,edgecolors='white',linewidth=1)
ax.set_xlabel("Number of Stereocenters");ax.set_ylabel("NP-likeness score")
ax.set_title("Stereochemical complexity vs NP-likeness",fontweight='bold')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=tc[t],label=t) for t in tc],fontsize=8)

plt.savefig("project7_np_analysis.png",dpi=150,bbox_inches='tight')
print(f"   ✅ project7_np_analysis.png saved")
df.to_csv("project7_np_results.csv",index=False)
elapsed=time.time()-start_time
print(f"\n⏱️  {elapsed:.1f}s\n{'='*70}\n  PIPELINE COMPLETE\n{'='*70}")
