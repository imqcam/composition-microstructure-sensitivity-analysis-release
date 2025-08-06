# Composition-to-Microstructure-sensitivity-analysis
The code conducts a global sensitivity analysis of microstructural features to the variation of elemental concentration in alloy composition.
The required dataset for this analysis is the chemical composition of the alloy (concentration of the alloying elements) and their corresponding microstructural features. These features could be phase fractions, grain size distribution, thermo-physical properties, or mechanical properties of the material. 
The provided code focuses on the Nickel-based alloy 718 and the sensitivity of its microstructural phase fraction to the variation of alloying elements. 

An open-source package, Chaospy, is used to build a Polynomial Chaos Expansion (PCE)-based surrogate model that predicts the phase fractions of microstructural features based on alloy composition. In addition to prediction, the surrogate model quantifies the sensitivity of each element by computing both the main effect and total effect Sobol indices. These indices help identify how strongly each element influences the variability in phase fractions. The elements with higher index values introduce higher variability in phase fractions when their concentrations vary. Therefore, controlling the concentrations of these high-sensitivity elements is essential to minimize uncertainty in the resulting microstructural phase fractions.
The code for the visualization of the sensitivity analysis is also provided.

The dataset Fig_1_dataset.xlsx includes reported yield strength values for alloy 718 samples produced using various manufacturing techniques, including cast, wrought, directed energy deposition (DED), laser powder bed fusion (LPBF), and electron beam melting (EBM), along with their corresponding references.

Figure_1.py script reproduces the figure that visualizes the variability in yield strength for both as-built and fully heat-treated alloy 718 samples across these different manufacturing methods.

