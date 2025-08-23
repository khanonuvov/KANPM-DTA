<h1 id="gsik-dta">GSIK-DTA: Graph-Sequence Integration with Kolmogorov–Arnold Networks for Improving Generalizability of Drug–Target Affinity Prediction</h1>

<p align="center">
  <!-- Adjust width as needed (e.g., 800–1100) -->
  <img src="images/architecture.png" alt="Model Architecture" width="900">
</p>
<p align="center"><em>Figure 1. GSIK-DTA model architecture.</em></p>

<h2 id="requirements">Requirements</h2>
<hr>
<ul>
  <li>Python 3.10+</li>
  <li>numpy==1.26.4</li>
  <li>torch==2.3.1</li>
  <li>transformers==4.42.3</li>
  <li>scikit-learn==1.5.0</li>
  <li>pandas==2.2.2</li>
</ul>

<h2 id="step-1-clone-repository">Step 1: Clone Repository</h2>
<hr>
<pre><code>git clone https://github.com/&lt;username&gt;/GSIK-DTA.git
cd GSIK-DTA
</code></pre>

<h2 id="step-2-install-requirements">Step 2: Install Requirements</h2>
<hr>
<p><strong>Option A: Using <code>requirements.txt</code></strong></p>
<pre><code>pip install -r requirements.txt
</code></pre>

<p><strong>Option B: Install from the list above</strong></p>
<pre><code>pip install numpy==1.26.4 torch==2.3.1 transformers==4.42.3 scikit-learn==1.5.0 pandas==2.2.2
</code></pre>

<p><strong>Conda (alternative)</strong></p>
<pre><code>conda env create -f environment.yml
conda activate gsik-dta
</code></pre>

<h2 id="step-3-run-the-project">Step 3: Run the Project</h2>
<hr>
<pre><code>python main.py
</code></pre>

<h2 id="export-your-environment">Export Your Current Environment (to capture exact versions)</h2>
<hr>
<p><strong>Pip</strong></p>
<pre><code>pip freeze &gt; requirements.txt
</code></pre>

<p><strong>Conda</strong></p>
<pre><code>conda env export &gt; environment.yml
</code></pre>

<p>Commit the generated file(s) to the repository and keep this README updated.</p>
