# YingHub-v3

**v3 Major Enhancements & Innovations**  
This release introduces significant architectural improvements, training optimizations, and novel features over v2, specifically designed for high-quality Shakespearean text generation.

---

## 🚀 Key Advancements (v3 vs v2)

### 1. **Architectural Upgrades**
- **Flash Attention Integration**  
  Implemented Triton-accelerated Flash Attention kernels (2.1x faster than standard PyTorch attention).
- **Heterogeneous Experts**  
  Introduced 3 expert types: *Deep* (complex patterns), *Wide* (contextual breadth), *Hybrid* (parallel residual paths).
- **Dynamic Top-K Routing**  
  Adaptive token-to-expert allocation with capacity-aware load balancing (15% better expert utilization).

### 2. **Training Optimizations**
- **Factorized Embeddings**  
  Low-rank embeddings + projection layers reduce memory usage by 40% with <1% accuracy drop.
- **Curriculum Learning Scheduler**  
  Progressive sequence length scaling (48→88 tokens) stabilizes RLHF fine-tuning.
- **Structured Dropout**  
  Block-wise dropout (20%) + gradient clipping (norm=1.2) prevents overfitting.

### 3. **Controlled Generation**
- **Dramatic Structure Enforcement**  
  State machine tracking for `<SCENE>`, `<SPEAKER>`, and `<STAGE>` tag consistency.
- **Iambic Pentameter Checker**  
  Real-time stress pattern validation with pronouncing.py integration.
- **Rhyme Schema Detection**  
  Supports ABAB/AABB/ABABCC patterns via phonetic analysis.

### 4. **Data Pipeline**
- **Enhanced Shakespeare Cleaning**  
  Specialized regex patterns for:  
  - Speaker turn management (`<SPEAKER>...</SPEAKER>`)  
  - Stage direction isolation (`<STAGE>[...]</STAGE>`)  
  - Act/scene boundary detection (`<ACT III>` → `<SCENE>III</SCENE>`)
- **Gutenberg Corpus Blending**  
  10% non-Shakespearean text injection improves linguistic diversity.

---

## 🛠 Usage Examples
```bash
### Generate a dataset
python data_cleaner.py

### Training
python MoE.py --train

### Generate
python MoE.py --generate

### RLHF
python MoE.py --rlhf

### After Generate fune-tuning
python MoE.py --ftgenerate
