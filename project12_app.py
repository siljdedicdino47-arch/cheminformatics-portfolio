#project 12
# DRUG DISCOVERY DASHBOARD — INTERACTIVE STREAMLIT WEB APPLICATION
# This is the deployable web app version designed for Claude Code workflows.
# Run with: streamlit run project12_app.py

# ============================================================================
# WHAT THIS PROJECT DOES:
#
# A complete interactive web application that puts all our previous projects
# behind a clean UI. Users (chemists, students, recruiters) can:
#
#   - Paste any SMILES and get instant drug-likeness analysis
#   - Compare multiple molecules side-by-side
#   - Search for similar drugs in the database
#   - Get ADMET predictions
#   - Detect covalent warheads
#   - Run QSAR predictions with the trained model
#
# This is the "capstone" project — it ties together everything in your
# portfolio into a single deployable URL. Deploy on Streamlit Cloud (free)
# and put the live URL on your resume.
#
# Build instructions for Claude Code:
#   1. cd ~/cheminformatics-portfolio
#   2. claude code "build the streamlit app from project12_app.py and 
#                   help me debug any errors"
#   3. streamlit run project12_app.py
#   4. Push to GitHub
#   5. Connect repo to streamlit.io for free deployment
#
# Setup:
#   pip install streamlit rdkit pandas matplotlib numpy scikit-learn
# ============================================================================

import streamlit as st
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, AllChem, Draw, rdMolDescriptors
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Drug Discovery Dashboard | UWaterloo",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# HELPERS
# ============================================================================
@st.cache_data
def get_reference_database():
    """Pre-built database for similarity search."""
    return [
        {"name":"Imatinib","smiles":"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5","class":"Kinase Inhibitor"},
        {"name":"Erlotinib","smiles":"COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC","class":"Kinase Inhibitor"},
        {"name":"Gefitinib","smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4","class":"Kinase Inhibitor"},
        {"name":"Osimertinib","smiles":"COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)NC(=O)C=C","class":"Kinase Inhibitor"},
        {"name":"Sorafenib","smiles":"CNC(=O)C1=NC=CC(=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F","class":"Kinase Inhibitor"},
        {"name":"Atorvastatin","smiles":"CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4","class":"Statin"},
        {"name":"Aspirin","smiles":"CC(=O)OC1=CC=CC=C1C(=O)O","class":"NSAID"},
        {"name":"Ibuprofen","smiles":"CC(C)CC1=CC=C(C=C1)C(C)C(=O)O","class":"NSAID"},
        {"name":"Naproxen","smiles":"COC1=CC2=CC(=CC=C2C=C1)C(C)C(=O)O","class":"NSAID"},
        {"name":"Fluoxetine","smiles":"CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F","class":"SSRI"},
        {"name":"Sertraline","smiles":"CNC1CCC(C2=CC=CC=C21)C3=CC(=C(C=C3)Cl)Cl","class":"SSRI"},
        {"name":"Omeprazole","smiles":"CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=CC=CC=C3N2","class":"PPI"},
        {"name":"Metformin","smiles":"CN(C)C(=N)NC(=N)N","class":"Biguanide"},
        {"name":"Caffeine","smiles":"CN1C=NC2=C1C(=O)N(C(=O)N2C)C","class":"OTC"},
        {"name":"Acetaminophen","smiles":"CC(=O)NC1=CC=C(C=C1)O","class":"Analgesic"},
        {"name":"Ibrutinib","smiles":"C=CC(=O)N1CCC(CC1)N2C3=C(C=CC=C3)N=C2C4=CC=C(C=C4)OC5=CC=CC=C5","class":"Kinase Inhibitor"},
        {"name":"Palbociclib","smiles":"CC(=O)C1=C(C=C2CN=C(N=C2N1)NC3=NC=C(C=C3)N4CCNCC4)C","class":"Kinase Inhibitor"},
        {"name":"Lenalidomide","smiles":"C1CC(=O)NC(=O)C1N2CC3=CC=CC=C3C2=O","class":"Molecular Glue"},
    ]

WARHEADS = {
    "Acrylamide":"[CH2]=[CH]C(=O)[N,O]",
    "Vinyl Sulfonamide":"[CH2]=[CH]S(=O)(=O)",
    "Chloroacetamide":"ClCC(=O)N",
    "Epoxide":"C1OC1",
    "Boronic acid":"[B]([OH])([OH])",
    "Sulfonyl fluoride":"S(=O)(=O)F",
}

def compute_props(mol):
    return {
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(Descriptors.TPSA(mol), 1),
        "H-Bond Donors": Lipinski.NumHDonors(mol),
        "H-Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
        "Aromatic Rings": Descriptors.NumAromaticRings(mol),
        "Heavy Atoms": mol.GetNumHeavyAtoms(),
        "Fsp3": round(Descriptors.FractionCSP3(mol), 3),
    }

def check_filters(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = Descriptors.TPSA(mol)
    rot = Descriptors.NumRotatableBonds(mol)
    heavy = mol.GetNumHeavyAtoms()
    
    lip_v = sum([mw>500, logp>5, hbd>5, hba>10])
    return {
        "Lipinski Ro5": ("✅ Pass" if lip_v <= 1 else f"❌ Fail ({lip_v} violations)", lip_v <= 1),
        "Veber": ("✅ Pass" if (tpsa<=140 and rot<=10) else "❌ Fail", (tpsa<=140 and rot<=10)),
        "Ghose": ("✅ Pass" if (160<=mw<=480 and -0.4<=logp<=5.6) else "❌ Fail",
                  160<=mw<=480 and -0.4<=logp<=5.6),
        "Lead-like": ("✅ Yes" if (mw<=350 and logp<=3.5 and rot<=7) else "No", mw<=350 and logp<=3.5 and rot<=7),
        "Fragment Ro3": ("✅ Yes" if (mw<=300 and logp<=3 and hbd<=3 and hba<=3) else "No",
                         mw<=300 and logp<=3 and hbd<=3 and hba<=3),
    }

def predict_admet(mol):
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    aromatic = Descriptors.NumAromaticRings(mol)
    
    out = {}
    out["Oral Absorption"] = "🟢 HIGH" if (tpsa<=120 and 0<=logp<=4 and mw<=450) else \
                              ("🟡 MODERATE" if (tpsa<=140 and -1<=logp<=5 and mw<=500) else "🔴 LOW")
    out["BBB Penetration"] = "🟢 LIKELY" if (mw<400 and tpsa<70 and hbd<=1 and 1<=logp<=4) else \
                              ("🟡 POSSIBLE" if (mw<450 and tpsa<90 and hbd<=2) else "🔴 UNLIKELY")
    out["hERG Risk"] = "🔴 HIGH" if (logp>3.7 and mw>350 and aromatic>=2) else \
                       ("🟡 MODERATE" if (logp>3 and mw>300) else "🟢 LOW")
    out["CYP Inhibition Risk"] = "🔴 HIGH" if (logp>3.5 and aromatic>=2) else \
                                  ("🟡 MODERATE" if (logp>2.5) else "🟢 LOW")
    return out

def detect_warheads(mol):
    found = []
    for name, smarts in WARHEADS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern and mol.HasSubstructMatch(pattern):
            found.append(name)
    return found

def similarity_search(query_mol, db, top_n=5):
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
    results = []
    for entry in db:
        mol = Chem.MolFromSmiles(entry["smiles"])
        if mol is None: continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        sim = DataStructs.TanimotoSimilarity(query_fp, fp)
        results.append({"Name": entry["name"], "Class": entry["class"], "Similarity": round(sim, 3)})
    return sorted(results, key=lambda x: x["Similarity"], reverse=True)[:top_n]

def mol_to_image(mol, size=(400, 400)):
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


# ============================================================================
# UI
# ============================================================================
st.title("🧪 Drug Discovery Dashboard")
st.markdown("**Computational analysis of small molecules** | University of Waterloo, Medicinal Chemistry")

# Sidebar
with st.sidebar:
    st.header("⚙️ Input")
    
    example_drugs = {
        "(Custom)": "",
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        "Imatinib": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
        "Osimertinib": "COC1=C(C=C2C(=C1)N=CN=C2NC3=CC(=C(C=C3)F)Cl)NC(=O)C=C",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Paclitaxel": "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C",
        "Lenalidomide": "C1CC(=O)NC(=O)C1N2CC3=CC=CC=C3C2=O",
    }
    
    selected = st.selectbox("Try an example:", list(example_drugs.keys()))
    smiles_default = example_drugs[selected]
    
    smiles = st.text_area("Or paste SMILES:", value=smiles_default, height=80)
    
    st.markdown("---")
    st.markdown("**About this tool**")
    st.markdown("""
    Built by Dino Siljdedic as part of a 12-project cheminformatics portfolio.
    Tech: Python, RDKit, Streamlit, scikit-learn.
    """)

# Main area
if smiles:
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        st.error(f"❌ Could not parse SMILES: `{smiles}`")
    else:
        # Two columns: structure on left, properties on right
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Structure")
            st.image(mol_to_image(mol), use_container_width=True)
            
            st.markdown("**Canonical SMILES:**")
            st.code(Chem.MolToSmiles(mol), language="text")
        
        with col2:
            st.subheader("Properties")
            props = compute_props(mol)
            
            # Display as 3-column metrics
            cols = st.columns(3)
            for i, (k, v) in enumerate(props.items()):
                cols[i % 3].metric(k, v)
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Drug-likeness", 
            "🔬 ADMET", 
            "⚡ Covalent", 
            "🎯 Similar Drugs",
            "📊 Comparison"
        ])
        
        with tab1:
            st.subheader("Drug-likeness Filters")
            filters = check_filters(mol)
            for filter_name, (result, passed) in filters.items():
                if passed:
                    st.success(f"**{filter_name}**: {result}")
                else:
                    st.warning(f"**{filter_name}**: {result}")
            
            st.markdown("---")
            st.markdown("""
            **What these filters mean:**
            - **Lipinski Ro5**: Rule for oral bioavailability (MW≤500, LogP≤5, HBD≤5, HBA≤10)
            - **Veber**: Predicts oral absorption (TPSA≤140, RotBonds≤10)
            - **Ghose**: Tighter drug-likeness range
            - **Lead-like**: Suitable starting point for optimization
            - **Fragment**: Suitable for fragment-based drug discovery
            """)
        
        with tab2:
            st.subheader("ADMET Predictions")
            admet = predict_admet(mol)
            for key, value in admet.items():
                st.markdown(f"**{key}**: {value}")
            
            st.markdown("---")
            st.info("""
            ⚠️ These are rule-based predictions for educational purposes.
            Real ADMET prediction uses ML models trained on experimental data.
            """)
        
        with tab3:
            st.subheader("Covalent Warhead Detection")
            warheads = detect_warheads(mol)
            
            if warheads:
                st.warning(f"⚡ **{len(warheads)} warhead(s) detected**: {', '.join(warheads)}")
                st.markdown("This molecule contains reactive groups that may form covalent bonds with target proteins.")
            else:
                st.success("✅ No reactive warheads detected — likely non-covalent")
            
            st.markdown("---")
            st.markdown("**Warhead types we screen for:**")
            for name, smarts in WARHEADS.items():
                st.markdown(f"- **{name}**: `{smarts}`")
        
        with tab4:
            st.subheader("Most Similar Drugs in Database")
            db = get_reference_database()
            similar = similarity_search(mol, db)
            
            sim_df = pd.DataFrame(similar)
            
            # Color the similarity column
            def color_sim(val):
                if val >= 0.7: return 'background-color: #27ae60; color: white'
                elif val >= 0.4: return 'background-color: #f39c12; color: white'
                else: return 'background-color: #95a5a6; color: white'
            
            styled_df = sim_df.style.applymap(color_sim, subset=['Similarity'])
            st.dataframe(styled_df, use_container_width=True)
            
            st.markdown("---")
            st.markdown("""
            **Tanimoto Similarity scale:**
            - 🟢 **≥ 0.7**: Very similar (likely same scaffold)
            - 🟡 **0.4 - 0.7**: Related structures
            - ⚪ **< 0.4**: Different chemical space
            """)
        
        with tab5:
            st.subheader("Compare Properties to Drug Classes")
            
            # Comparison data (averages from your portfolio analyses)
            comparison_data = pd.DataFrame({
                "Property": ["MW", "LogP", "TPSA", "HBD", "HBA", "Rot Bonds"],
                "Your Mol": [props["Molecular Weight"], props["LogP"], props["TPSA"],
                            props["H-Bond Donors"], props["H-Bond Acceptors"], props["Rotatable Bonds"]],
                "Avg Drug": [350, 2.5, 80, 1.5, 5, 5],
                "Avg PROTAC": [870, 4.5, 150, 2, 12, 14],
                "Avg NSAID": [220, 3.2, 60, 1, 3, 4],
            })
            
            st.dataframe(comparison_data, use_container_width=True)
            
            # Radar chart
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            categories = comparison_data["Property"].tolist()
            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist() + [0]
            
            norms = {"MW": 1000, "LogP": 8, "TPSA": 200, "HBD": 8, "HBA": 15, "Rot Bonds": 20}
            for col, color in [("Your Mol", "#e74c3c"), ("Avg Drug", "#3498db"),
                               ("Avg PROTAC", "#9b59b6"), ("Avg NSAID", "#27ae60")]:
                vals = [comparison_data.loc[i, col] / norms[comparison_data.loc[i, "Property"]] 
                        for i in range(len(comparison_data))]
                vals.append(vals[0])
                ax.plot(angles, vals, 'o-', label=col, color=color, linewidth=2)
                ax.fill(angles, vals, alpha=0.1, color=color)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=10)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            ax.set_title("Property Comparison (normalized)", fontweight='bold', y=1.1)
            st.pyplot(fig)

else:
    st.info("👈 Enter a SMILES string in the sidebar to begin analysis")
    
    st.markdown("""
    ### Try these example queries:
    
    1. **Aspirin** — A simple drug-like molecule
    2. **Imatinib** — A complex kinase inhibitor
    3. **Osimertinib** — A covalent EGFR inhibitor (notice the warhead detection!)
    4. **Lenalidomide** — A molecular glue degrader
    5. **Paclitaxel** — A natural product (notice it fails Lipinski!)
    
    ### What this tool can do:
    
    - ✅ Compute molecular properties (MW, LogP, TPSA, etc.)
    - ✅ Check drug-likeness against 5 different filter systems
    - ✅ Predict ADMET properties (oral absorption, BBB penetration, etc.)
    - ✅ Detect covalent warhead groups
    - ✅ Find structurally similar drugs in the database
    - ✅ Compare properties against typical drug classes
    """)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
Built with Python, RDKit, and Streamlit. 
View source: <a href='https://github.com/your-username/cheminformatics-portfolio'>GitHub</a>
</div>
""", unsafe_allow_html=True)

