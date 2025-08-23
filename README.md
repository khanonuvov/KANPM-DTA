<h1 id="gsik-dta">GSIK-DTA: Graph-Sequence Integration with Kolmogorov–Arnold Networks for Improving Generalizability of Drug–Target Affinity Prediction</h1>

<p align="center">
  <!-- Adjust width as needed (e.g., 800–1100) -->
  <img src="images/architecture.png" alt="Model Architecture" width="900">
</p>
<p align="center"><em>Figure 1. GSIK-DTA model architecture.</em></p>

<h2 id="requirements">Requirements</h2>
<ul>
  <li>Python 3.9.21</li>
  <li>numpy==2.0.2</li>
  <li>pandas==2.2.3</li>
  <li>torch==2.6.0</li>
  <li>transformers==4.49.0</li>
  <li>rdkit==2024.3.2</li>
  <li>fair-esm==2.0.0</li>
</ul>

<h2 id="step-1-clone-repository">Step 1: Clone Repository</h2>
<pre><code>git clone https://github.com/khanonuvov/GSIK-DTA.git
cd GSIK-DTA
</code></pre>

<h2 id="step-2-Generate Pretrained Models">Step 2: Generate Pretrained Models</h2>
<p>For each dataset (<code>davis</code>, <code>kiba</code>, <code>metz</code>) run these three commands:</p>

<h4>Drug Sequence (SMILES) → Chembarta-2</h4>
<pre><code>python pretrained/chemberta_pretraiend.py</code></pre>

<h4>Protein sequence (FASTA) → ESM-Cambrian</h4>
<pre><code>python pretrained/esmC_pretraiend.py</code></pre>

<h4>Protein contact map (from same FASTA) → ESM-2</h4>
<pre><code>python pretrained/esm2_map.py</code></pre>

<h4>Lung Cancer (EGFR) Test</h4>
<pre><code>python pretrained/chemberta_pretraiend.py</code></pre>
<pre><code>python pretrained/esmC_pretraiend.py</code></pre>
<pre><code>python pretrained/esm2_map.py</code></pre>

<h2 id="step-3-Train the Model">Step 3: Train the Model</h2>
<pre><code>python main.py
</code></pre>

<h2 id="contact">Contact</h2>
<p>
  For inquiries, please contact
  <strong>Md Youshuf Khan Rakib</strong> —
  <a href="mailto:khanushuf4619@csu.edu.cn">khanushuf4619@csu.edu.cn</a>.
</p>
