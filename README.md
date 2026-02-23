# 🧠 Deep Neural Network for Non-Linear Regression — 8 Implementations

> **3-Layer DNN across NumPy, PyTorch, PyTorch Lightning, and TensorFlow (4 variants)**
> 
> Every implementation solves the **same problem** using different frameworks and abstraction levels — from pure manual backpropagation to high-level `model.fit()`.

- Colab A: https://colab.research.google.com/drive/1UX6DCdDNB1FzSa-Q0dUVe7e6a4by6Y6y?usp=sharing
- Video A: https://www.loom.com/share/762936546c754ec483198aa9ac3ed6dd

- Colab B: https://colab.research.google.com/drive/1tpNbD7d1lInJSodivLIGoTSxYPI3W3OX?usp=sharing
- Video B: https://www.loom.com/share/664dc44dce3d45238acb8f8651437781

- Colab C: https://colab.research.google.com/drive/11EgTuxYXtMeelUyP-5hs0dXXm1Xpm5W1?usp=sharing
- Video C: https://www.loom.com/share/3a2e4917d66f4ea68cdb3773a234fbbb

- Colab D: https://colab.research.google.com/drive/1XljznxEpP3KS9_w7a7iKAby7KflqruqV?usp=sharing
- Video D: https://www.loom.com/share/d18ee5f3a5e6480cb543534631df708f

- Colab E-1: https://colab.research.google.com/drive/1nqmmkhcwEPkavuZYB8QxEuz7D21UHnY4?usp=sharing
- Video E-1: https://www.loom.com/share/0ee18247579f4682ab36f63cf01ce0e3

- Colab E-2: https://colab.research.google.com/drive/17Zb1eS5llzUC--xmbh1m3-8HD3D0_0wS?usp=sharing
- Video E-2: https://www.loom.com/share/0d4d7238b28c4843b7f8f1e2f4cf6d64

- Colab E-3: https://colab.research.google.com/drive/1Fme6RdIRg0ad02nSj5mUF1d75Fym5y7n?usp=sharing
- Video E-3: https://www.loom.com/share/e3ec44c544e84ebb91a5ae2bc62bedca

- Colab E-4: https://colab.research.google.com/drive/1qljkEX61f1FIiddr_fexlZrBjFvkUxPW?usp=sharing
- Video E-4: https://www.loom.com/share/8d66d1d93d5d4e4da1fc22972b8d24b4
---

## 🎯 Problem Statement

All 8 notebooks solve **the same non-linear regression problem** with a **3-layer deep neural network**.

### Target Equation (3 Variables)

```
y = sin(x₁) · x₂² + cos(x₃) · x₁ + 0.5 · x₃ · x₂
```

- **Inputs**: 3 variables (x₁, x₂, x₃) uniformly sampled from [-2, 2]
- **Output**: 1 continuous value (y)
- **Samples**: 2,000 data points with normalization

### Network Architecture

```
Input(3) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, Linear)
```

| Layer | Neurons | Activation | Parameters |
|-------|---------|------------|------------|
| Input | 3 | — | — |
| Hidden 1 | 64 | ReLU | 256 (W) + 64 (b) = 320 |
| Hidden 2 | 32 | ReLU | 2,048 (W) + 32 (b) = 2,080 |
| Hidden 3 | 16 | ReLU | 512 (W) + 16 (b) = 528 |
| Output | 1 | Linear | 16 (W) + 1 (b) = 17 |
| **Total** | | | **2,945** |

### 4D Visualization

Since we have 3 input features + 1 output (4D), we use **PCA** (scikit-learn) to reduce input dimensionality to 2 components, then visualize with the target as color/z-axis.

---

## 📁 Repository Structure

```
├── README.md                                    ← This file
├── VIDEO_SCRIPTS.md                             ← All 8 video narration scripts
├── Colab_A_NumPy_From_Scratch_3Layer_DNN.ipynb  ← (a) NumPy + tf.einsum manual backprop
├── Colab_B_PyTorch_From_Scratch_3Layer_DNN.ipynb← (b) PyTorch raw tensors, no nn.Module
├── Colab_C_PyTorch_Classes_3Layer_DNN.ipynb     ← (c) PyTorch nn.Module class-based
├── Colab_D_PyTorch_Lightning_3Layer_DNN.ipynb   ← (d) PyTorch Lightning
├── Colab_Ei_TF_From_Scratch_LowLevel.ipynb      ← (e-i) TF low-level, no Keras
├── Colab_Eii_TF_BuiltIn_Layers.ipynb            ← (e-ii) TF Model subclassing + Dense
├── Colab_Eiii_TF_Functional_API.ipynb            ← (e-iii) TF Functional API
└── Colab_Eiv_TF_HighLevel_Sequential.ipynb       ← (e-iv) TF Sequential + model.fit
```

---

## 📓 Colab Notebooks & Videos

### Colab A — NumPy From Scratch (Manual Backprop)

| | |
|---|---|
| **File** | [`Colab_A_NumPy_From_Scratch_3Layer_DNN.ipynb`](Colab_A_NumPy_From_Scratch_3Layer_DNN.ipynb) |
| **Framework** | NumPy + `tf.einsum` (TF used ONLY for einsum) |
| **Abstraction** | ⭐ Lowest — everything manual |
| **Video** | 📹 [Watch Walkthrough](#video-a) |

**What makes this unique:**
- **Manual backpropagation** — chain rule gradient computation coded by hand
- **`tf.einsum('ij,jk->ik', ...)`** replaces all `np.matmul` / `np.dot` calls (assignment requirement)
- Forward pass caches all intermediate Z and A values for backprop
- He initialization (`√(2/fan_in)`) for ReLU compatibility
- Mini-batch gradient descent with shuffling

**Key Cells:**
| Cell | Description |
|------|------------|
| Cell 2 | Generates synthetic data from the 3-variable non-linear equation |
| Cell 3 | 4D visualization using PCA dimensionality reduction |
| Cell 4 | Network architecture definition & He weight initialization |
| Cell 6 | **Forward pass with `tf.einsum`** — the core assignment requirement |
| Cell 8 | **Manual backward pass** — chain rule through all 4 layers |
| Cell 10 | Training loop (200 epochs, batch size 64) |
| Cell 11 | Loss curve (log scale) |
| Cell 12 | Predicted vs actual, residual histogram, PCA prediction plot |

---

### Colab B — PyTorch From Scratch (No nn.Module)

| | |
|---|---|
| **File** | [`Colab_B_PyTorch_From_Scratch_3Layer_DNN.ipynb`](Colab_B_PyTorch_From_Scratch_3Layer_DNN.ipynb) |
| **Framework** | PyTorch (raw tensors only) |
| **Abstraction** | ⭐ Low — no built-in layer classes |
| **Video** | 📹 [Watch Walkthrough](#video-b) |

**What makes this unique:**
- **NO `nn.Module`, NO `nn.Linear`, NO optimizer object**
- Weights are raw `torch.Tensor` with `requires_grad=True`
- Forward pass uses `@` operator for matrix multiply
- PyTorch autograd computes backward pass, but SGD update is manual
- `p -= learning_rate * p.grad` inside `torch.no_grad()` block

**Key Cells:**
| Cell | Description |
|------|------------|
| Cell 4 | Raw tensor weight initialization with `requires_grad_(True)` |
| Cell 5 | Forward pass — pure `@` operator + `torch.relu`, no layer classes |
| Cell 6 | Training with `loss.backward()` + **manual SGD** (no `optim.SGD`) |

---

### Colab C — PyTorch nn.Module (Class-Based)

| | |
|---|---|
| **File** | [`Colab_C_PyTorch_Classes_3Layer_DNN.ipynb`](Colab_C_PyTorch_Classes_3Layer_DNN.ipynb) |
| **Framework** | PyTorch (standard nn.Module) |
| **Abstraction** | ⭐⭐ Medium — standard PyTorch practice |
| **Video** | 📹 [Watch Walkthrough](#video-c) |

**What makes this unique:**
- Standard **`nn.Module` subclassing** — the PyTorch recommended approach
- `nn.Sequential` with `nn.Linear` + `nn.ReLU`
- Kaiming/He initialization via `nn.init.kaiming_normal_`
- `torch.optim.Adam` optimizer with `nn.MSELoss`
- Train/test split with `DataLoader`

**Key Cells:**
| Cell | Description |
|------|------------|
| Cell 4 | `NonLinearRegressionNet(nn.Module)` class definition |
| Cell 5 | `nn.MSELoss()` + `optim.Adam()` |
| Cell 6 | Standard train/eval loop with `model.train()` / `model.eval()` |

---

### Colab D — PyTorch Lightning

| | |
|---|---|
| **File** | [`Colab_D_PyTorch_Lightning_3Layer_DNN.ipynb`](Colab_D_PyTorch_Lightning_3Layer_DNN.ipynb) |
| **Framework** | PyTorch Lightning |
| **Abstraction** | ⭐⭐⭐ High — framework handles boilerplate |
| **Video** | 📹 [Watch Walkthrough](#video-d) |

**What makes this unique:**
- **`LightningModule`** — defines model + training/validation steps
- **`LightningDataModule`** — handles data loading pipeline
- **`Trainer`** — single line `trainer.fit(model, dm)` runs entire training
- Automatic device management (CPU/GPU), logging, and progress bars
- `save_hyperparameters()` for reproducibility

**Key Cells:**
| Cell | Description |
|------|------------|
| Cell 4 | `RegressionDataModule(pl.LightningDataModule)` |
| Cell 5 | `LitRegressionNet(pl.LightningModule)` with `training_step`, `configure_optimizers` |
| Cell 6 | `Trainer(max_epochs=200).fit(model, dm)` — one line training |

---

### Colab E(i) — TensorFlow Low-Level (No Keras)

| | |
|---|---|
| **File** | [`Colab_Ei_TF_From_Scratch_LowLevel.ipynb`](Colab_Ei_TF_From_Scratch_LowLevel.ipynb) |
| **Framework** | TensorFlow (raw `tf.Variable` + `tf.GradientTape`) |
| **Abstraction** | ⭐ Lowest TF level — no Keras at all |
| **Video** | 📹 [Watch Walkthrough](#video-ei) |

**What makes this unique:**
- **NO Keras layers, NO Keras Model, NO optimizer object**
- Raw `tf.Variable` for all weights
- **`tf.einsum`** for matrix multiplications
- `tf.GradientTape` for automatic differentiation
- Manual SGD: `w.assign_sub(lr * gradient)`

**Key Cells:**
| Cell | Description |
|------|------------|
| Cell 4 | `tf.Variable(he_init([...]))` — raw weight creation |
| Cell 5 | Forward pass with `tf.einsum('ij,jk->ik', ...)` |
| Cell 6 | `tf.GradientTape` + manual `assign_sub` update |

---

### Colab E(ii) — TensorFlow Built-in Layers (Model Subclassing)

| | |
|---|---|
| **File** | [`Colab_Eii_TF_BuiltIn_Layers.ipynb`](Colab_Eii_TF_BuiltIn_Layers.ipynb) |
| **Framework** | TensorFlow / Keras (Model subclassing) |
| **Abstraction** | ⭐⭐ Medium — built-in layers + custom loop |
| **Video** | 📹 [Watch Walkthrough](#video-eii) |

**What makes this unique:**
- **`keras.Model` subclassing** — define architecture in `__init__`, logic in `call`
- `layers.Dense` handles weight creation, initialization, and forward math
- Custom training loop with `tf.GradientTape` (NOT `model.fit`)
- `@tf.function` decorator compiles to static graph for speed
- `keras.optimizers.Adam` for parameter updates

**Key Cells:**
| Cell | Description |
|------|------------|
| Cell 4 | `RegressionDNN(keras.Model)` with `layers.Dense` |
| Cell 5 | `@tf.function` decorated `train_step` with `GradientTape` |

---

### Colab E(iii) — TensorFlow Functional API

| | |
|---|---|
| **File** | [`Colab_Eiii_TF_Functional_API.ipynb`](Colab_Eiii_TF_Functional_API.ipynb) |
| **Framework** | TensorFlow / Keras (Functional API) |
| **Abstraction** | ⭐⭐⭐ High — declarative graph construction |
| **Video** | 📹 [Watch Walkthrough](#video-eiii) |

**What makes this unique:**
- **`keras.Input(shape=(3,))`** declares input type
- Layers chained **functionally**: `x = Dense(64, 'relu')(inputs)` → `x = Dense(32, 'relu')(x)` → ...
- **`keras.Model(inputs, outputs)`** builds the model from the DAG
- Supports `plot_model()` for architecture visualization
- Enables multi-input / multi-output architectures (not possible with Sequential)

**Key Cells:**
| Cell | Description |
|------|------------|
| Cell 4 | Functional model: `keras.Input` → chained `Dense` calls → `keras.Model` |
| Cell 5 | `keras.utils.plot_model` — visual DAG of the network |

---

### Colab E(iv) — TensorFlow High-Level Sequential + `model.fit`

| | |
|---|---|
| **File** | [`Colab_Eiv_TF_HighLevel_Sequential.ipynb`](Colab_Eiv_TF_HighLevel_Sequential.ipynb) |
| **Framework** | TensorFlow / Keras (Sequential + fit) |
| **Abstraction** | ⭐⭐⭐⭐ Highest — maximum automation |
| **Video** | 📹 [Watch Walkthrough](#video-eiv) |

**What makes this unique:**
- **`keras.Sequential([...])` — model in ~6 lines**
- **`model.compile(optimizer, loss, metrics)`** — configure everything
- **`model.fit(X, Y, validation_data, callbacks)`** — one line training
- **`model.evaluate(X_test, Y_test)`** — one line testing
- **Callbacks**: `EarlyStopping` (patience=20), `ReduceLROnPlateau` (factor=0.5)
- Train/Val/Test split (70/15/15)

**Key Cells:**
| Cell | Description |
|------|------------|
| Cell 4 | `keras.Sequential` model definition |
| Cell 5 | `model.compile()` with optimizer, loss, metrics |
| Cell 6 | EarlyStopping + ReduceLROnPlateau callbacks |
| Cell 7 | `model.fit()` with full history tracking |

---

## 📊 Framework Comparison

| Aspect | A (NumPy) | B (PyTorch Raw) | C (PyTorch Module) | D (Lightning) | E-i (TF Low) | E-ii (TF Layers) | E-iii (TF Func) | E-iv (TF Seq) |
|--------|-----------|-----------------|--------------------|----|---|---|---|---|
| **Backprop** | Manual chain rule | Autograd | Autograd | Autograd | GradientTape | GradientTape | GradientTape | Automatic |
| **Weights** | NumPy arrays | Raw tensors | nn.Linear | nn.Linear | tf.Variable | Dense layers | Dense layers | Dense layers |
| **Optimizer** | Manual SGD | Manual SGD | optim.Adam | configure_optimizers | Manual SGD | Adam | Adam | Adam |
| **Training Loop** | Manual | Manual | Manual | Trainer | Manual | Custom | Custom | model.fit |
| **Matrix Multiply** | tf.einsum | @ operator | nn.Linear | nn.Linear | tf.einsum | Dense | Dense | Dense |
| **Lines of Code** | ~120 | ~80 | ~60 | ~50 | ~80 | ~60 | ~55 | ~30 |
| **Abstraction** | ⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

### Abstraction Spectrum

```
LOWEST                                                              HIGHEST
   │                                                                    │
   ▼                                                                    ▼
Colab A ──► Colab B ──► Colab E(i) ──► Colab C ──► Colab E(ii) ──► Colab E(iii) ──► Colab D ──► Colab E(iv)
NumPy       PyTorch     TF Low         PyTorch     TF Layers       TF Functional    Lightning   TF Sequential
Manual BP   Raw Tensors Raw Variables  nn.Module   Subclassing     Functional API   auto loop   model.fit()
```

---

## 🚀 How to Run

### Option 1: Google Colab (Recommended)
1. Open any `.ipynb` file in this repository
2. Click "Open in Colab" or upload to [colab.research.google.com](https://colab.research.google.com)
3. Run all cells top to bottom (`Runtime → Run all`)
4. No additional dependencies needed — all notebooks are self-contained

### Option 2: Local Jupyter
```bash
pip install numpy tensorflow torch pytorch-lightning scikit-learn matplotlib
jupyter notebook
```

### Dependencies
| Package | Version | Used In |
|---------|---------|---------|
| NumPy | ≥1.21 | All |
| TensorFlow | ≥2.10 | A, E(i-iv) |
| PyTorch | ≥1.12 | B, C, D |
| PyTorch Lightning | ≥1.9 | D |
| scikit-learn | ≥1.0 | All (PCA) |
| matplotlib | ≥3.5 | All |

---

## 🎬 Video Walkthroughs

> Each video is a 2-3 minute screen recording walkthrough covering every cell in the notebook with code explanation and output demonstration.

| # | Notebook | Video Link | Duration |
|---|----------|-----------|----------|
| 1 | Colab A — NumPy From Scratch | <a name="video-a"></a> 📹 [YouTube Link - REPLACE WITH YOUR URL] | ~2.5 min |
| 2 | Colab B — PyTorch From Scratch | <a name="video-b"></a> 📹 [YouTube Link - REPLACE WITH YOUR URL] | ~2 min |
| 3 | Colab C — PyTorch nn.Module | <a name="video-c"></a> 📹 [YouTube Link - REPLACE WITH YOUR URL] | ~2 min |
| 4 | Colab D — PyTorch Lightning | <a name="video-d"></a> 📹 [YouTube Link - REPLACE WITH YOUR URL] | ~2 min |
| 5 | Colab E(i) — TF Low-Level | <a name="video-ei"></a> 📹 [YouTube Link - REPLACE WITH YOUR URL] | ~2.5 min |
| 6 | Colab E(ii) — TF Built-in Layers | <a name="video-eii"></a> 📹 [YouTube Link - REPLACE WITH YOUR URL] | ~2 min |
| 7 | Colab E(iii) — TF Functional API | <a name="video-eiii"></a> 📹 [YouTube Link - REPLACE WITH YOUR URL] | ~2 min |
| 8 | Colab E(iv) — TF High-Level | <a name="video-eiv"></a> 📹 [YouTube Link - REPLACE WITH YOUR URL] | ~2.5 min |

> **⚠️ Replace the video links above with your actual YouTube/Google Drive URLs after recording!**

### What Each Video Covers
- ✅ GitHub repository shown with all files checked in
- ✅ Cell-by-cell walkthrough of the executed Colab
- ✅ Explanation of code logic in each section
- ✅ Training output and loss curves
- ✅ Final predictions, R² scores, and visualizations

---

## 📚 References

- [TensorFlow 2.0 + Keras Crash Course (François Chollet)](https://colab.research.google.com/drive/1UCJt8EYjlzCs1H1d1X0iDGYJsHKwu-NO)
- [Intro to Keras for Researchers](https://keras.io/getting_started/intro_to_keras_for_researchers/)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [4D Plotting with Matplotlib](https://www.tutorialspoint.com/how-to-make-a-4d-plot-with-matplotlib-using-arbitrary-data)

---

*Built for CMPE/DATA 255 — Deep Learning Assignment*
