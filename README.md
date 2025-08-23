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


<h2 id="step-3-Train the Model">Step 3: Train the Model</h2>
<pre><code>python main.py
</code></pre>

<h2 id="contact">Contact</h2>
<p>
  For inquiries, please contact
  <strong>Md Youshuf Khan Rakib</strong> —
  <a href="mailto:khanushuf4619@csu.edu.cn">khanushuf4619@csu.edu.cn</a>.
</p>





<hr>
<p><strong>Goal.</strong> This step builds all pretrained artifacts needed by the trainer:
  <em>(a)</em> drug embeddings from SMILES using <strong>ChemBERTa</strong>,
  <em>(b)</em> protein residue/contact-map representations from FASTA using <strong>ESM-2</strong>,
  <em>(c)</em> protein sequence embeddings from FASTA using <strong>ESM-Cambrian</strong>.
</p>

<p><strong>Scripts (adjust names to match your repo):</strong></p>
<ul>
  <li>ChemBERTa (SMILES) — <code>pretrained/chemberta.py</code> <em>(or</em> <code>chemberta_pretrained.py</code><em>)</em></li>
  <li>ESM-2 (FASTA → residue reps + contact maps) — <code>pretrained/esm2_map.py</code></li>
  <li>ESM-Cambrian (FASTA → sequence embeddings) — <code>pretrained/esmc_pretrained.py</code> <em>(or</em> <code>esmC_pretrained.py</code><em>)</em></li>
</ul>

<hr>
<h3>Prerequisites</h3>
<ul>
  <li>Install: <code>pip install torch fair-esm transformers numpy pandas scikit-learn rdkit-pypi</code></li>
  <li>Expected data layout (example):<br>
    <code>data/&lt;dataset&gt;/drugs.csv</code> (columns: <code>drug_id,smiles</code>)<br>
    <code>data/&lt;dataset&gt;/proteins.fasta</code> (single multi-FASTA or per-protein files)<br>
    <code>data/&lt;dataset&gt;/splits/*.csv</code> (optional train/val/test splits)</li>
  <li>GPU is recommended; use <code>--device cuda</code>. CPU also works (slower).</li>
</ul>

<hr>
<h3>Common flags</h3>
<pre><code>--seed 42                 # reproducibility
--device cuda             # or "cpu"
--batch_size 64           # tune for your GPU memory
--out_dir pretrained/&lt;dataset&gt;/&lt;modality&gt;
</code></pre>

<hr>
<h3>Per-dataset recipes</h3>

<h4>DAVIS</h4>
<pre><code># 1) SMILES → ChemBERTa (drug embeddings)
python pretrained/chemberta.py \
  --smiles_csv data/davis/drugs.csv \
  --model_name seyonec/ChemBERTa-zinc-base-v1 \
  --batch_size 64 --device cuda --seed 42 \
  --out_dir pretrained/davis/chemberta

# 2) FASTA → ESM-2 (residue reps + contact maps)
python pretrained/esm2_map.py \
  --fasta data/davis/proteins.fasta \
  --esm2_variant esm2_t33_650M_UR50D \
  --contacts yes --device cuda --seed 42 \
  --out_dir pretrained/davis/esm2

# 3) FASTA → ESM-Cambrian (protein sequence embeddings)
python pretrained/esmc_pretrained.py \
  --fasta data/davis/proteins.fasta \
  --hf_model Bytedance/ESM-Cambrian-650M \
  --batch_size 8 --device cuda --seed 42 \
  --out_dir pretrained/davis/esmc
</code></pre>

<h4>KIBA</h4>
<pre><code>python pretrained/chemberta.py \
  --smiles_csv data/kiba/drugs.csv \
  --model_name seyonec/ChemBERTa-zinc-base-v1 \
  --batch_size 64 --device cuda --seed 42 \
  --out_dir pretrained/kiba/chemberta

python pretrained/esm2_map.py \
  --fasta data/kiba/proteins.fasta \
  --esm2_variant esm2_t33_650M_UR50D \
  --contacts yes --device cuda --seed 42 \
  --out_dir pretrained/kiba/esm2

python pretrained/esmc_pretrained.py \
  --fasta data/kiba/proteins.fasta \
  --hf_model Bytedance/ESM-Cambrian-650M \
  --batch_size 8 --device cuda --seed 42 \
  --out_dir pretrained/kiba/esmc
</code></pre>

<h4>METZ</h4>
<pre><code>python pretrained/chemberta.py \
  --smiles_csv data/metz/drugs.csv \
  --model_name seyonec/ChemBERTa-zinc-base-v1 \
  --batch_size 64 --device cuda --seed 42 \
  --out_dir pretrained/metz/chemberta

python pretrained/esm2_map.py \
  --fasta data/metz/proteins.fasta \
  --esm2_variant esm2_t33_650M_UR50D \
  --contacts yes --device cuda --seed 42 \
  --out_dir pretrained/metz/esm2

python pretrained/esmc_pretrained.py \
  --fasta data/metz/proteins.fasta \
  --hf_model Bytedance/ESM-Cambrian-650M \
  --batch_size 8 --device cuda --seed 42 \
  --out_dir pretrained/metz/esmc
</code></pre>

<h4>TEST (toy/debug)</h4>
<pre><code>python pretrained/chemberta.py \
  --smiles_csv data/test/drugs.csv \
  --model_name seyonec/ChemBERTa-zinc-base-v1 \
  --batch_size 64 --device cuda --seed 42 \
  --out_dir pretrained/test/chemberta

python pretrained/esm2_map.py \
  --fasta data/test/proteins.fasta \
  --esm2_variant esm2_t33_650M_UR50D \
  --contacts yes --device cuda --seed 42 \
  --out_dir pretrained/test/esm2

python pretrained/esmc_pretrained.py \
  --fasta data/test/proteins.fasta \
  --hf_model Bytedance/ESM-Cambrian-650M \
  --batch_size 8 --device cuda --seed 42 \
  --out_dir pretrained/test/esmc
</code></pre>

<hr>
<h3>Outputs written</h3>
<table border="1" cellpadding="6" cellspacing="0">
  <thead>
    <tr>
      <th>Folder</th>
      <th>Files</th>
      <th>Meaning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>pretrained/&lt;ds&gt;/chemberta</code></td>
      <td><code>drug_emb.pt</code>, <code>drug_ids.json</code></td>
      <td>Drug-level embeddings from SMILES</td>
    </tr>
    <tr>
      <td><code>pretrained/&lt;ds&gt;/esm2</code></td>
      <td><code>protein_repr.pt</code>, <code>contact_maps.pt</code>, <code>protein_ids.json</code></td>
      <td>Residue/sequence reps and contact maps (ESM-2)</td>
    </tr>
    <tr>
      <td><code>pretrained/&lt;ds&gt;/esmc</code></td>
      <td><code>protein_seq_emb.pt</code>, <code>protein_ids.json</code></td>
      <td>Protein sequence embeddings (ESM-Cambrian)</td>
    </tr>
  </tbody>
</table>

<hr>
<h3>Verify artifacts (quick check)</h3>
<pre><code>python - &lt;&lt;'PY'
import torch, json
drug = torch.load('pretrained/davis/chemberta/drug_emb.pt', map_location='cpu')
prot = torch.load('pretrained/davis/esm2/protein_repr.pt', map_location='cpu')
cm   = torch.load('pretrained/davis/esm2/contact_maps.pt', map_location='cpu')
psc  = torch.load('pretrained/davis/esmc/protein_seq_emb.pt', map_location='cpu')
print('drug:', drug.shape)
print('prot_repr:', prot.shape)
print('contact_maps:', cm.shape)
print('prot_seq_emb:', psc.shape)
print('OK')
PY
</code></pre>

<hr>
<h3>Tips</h3>
<ul>
  <li>If you hit CUDA OOM, reduce <code>--batch_size</code> and/or add mixed precision (e.g., <code>--fp16</code> if supported in your scripts).</li>
  <li>Record exact model variants used:
    <ul>
      <li>ChemBERTa: <code>seyonec/ChemBERTa-zinc-base-v1</code></li>
      <li>ESM-2: e.g., <code>esm2_t33_650M_UR50D</code></li>
      <li>ESM-Cambrian: e.g., <code>Bytedance/ESM-Cambrian-650M</code></li>
    </ul>
  </li>
  <li>Ensure <code>drug_id</code>/<code>protein_id</code> in CSV/FASTA align with the generated <code>*_ids.json</code>.</li>
</ul>
