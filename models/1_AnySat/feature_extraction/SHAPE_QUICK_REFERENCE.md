# Quick Reference: S1 Shape Transformation

## 🎯 Die Essenz in 30 Sekunden

```
Input:  [256, 256, 3]      (Bild mit VV, VH, Ratio Kanälen)
         ↓
      [AnySat Model]
         ↓
Output: [256, 256, 1536]   (1536-dim Feature pro Pixel)
```

---

## 💡 Was passiert dazwischen?

| Stufe | Input Shape | Prozess | Output Shape |
|-------|------------|---------|-------------|
| 1 | `[B,3,256,256]` | Batch-Vorbereitung | `[B,1,3,256,256]` |
| 2 | `[B,1,3,256,256]` | S1 Projektor | `[B*N, N, 768]` |
| 3 | `[B*N, N, 768]` | Spatial Encoder | `[B*N, N, 768]` |
| 4 | `[B*N, N+1, 768]` | Transformer ×12 | `[B*N, N+1, 768]` |
| 5 | `[B*N, N+1, 768]` | Sub-Patch Fusion | `[B*N, H, W, 1536]` |
| 6 | `[B, H, W, 1536]` | Speicherung | `[256, 256, 1536]` |

**N** = Anzahl Patches (abhängig von patch_size)

---

## 🔑 Wichtige Parameter

```python
patch_size = 40        # Meter (bestimmt Patch-Auflösung)
scale = 4             # Intern: patch_size // 10
output = 'dense'      # Pro-Pixel Output (nicht Pro-Patch!)
embed_dim = 768       # Basis-Feature-Dimension
output_features = 1536 # = 2 × embed_dim
```

---

## 📊 Feature-Dimension Erklärung

```
1536-dim Features = 768 + 768
                  = Coarse Patches + Fine Sub-Patches
                  = Global Context + Local Details
```

**Coarse (768-dim):**
- Aus Transformer (12 Blöcke)
- Globale Kontexte, Patch-übergreifende Patterns

**Fine (768-dim):**  
- Sub-Patch Level Details
- 1×1 Pixel Granularität
- Räumliche Feinheiten für Segmentation

---

## ✅ Shape Validation (aus deinem Code)

```python
expected_dims_sorted = [256, 256, 1536]
if sorted(list(single_feature.shape)) != expected_dims_sorted:
    raise ValueError(...)
```

Diese Check stellt sicher, dass die Output-Shape korrekt ist!

---

## 🚀 Verwendung in deinem Workflow

```python
# Input vorbereiten
batch = batch.unsqueeze(1).to(device)  # [B, 1, 3, 256, 256]

data = {
    "s1": batch,
    "s1_dates": torch.tensor([196] * batch.shape[0]).to(device)
}

# Model aufrufen
batch_features = model(data, patch_size=40, output='dense', output_modality='s1')
# Output: [B, 256, 256, 1536]

# Pro Element speichern
for i in range(len(batch)):
    single_feature = batch_features[i].cpu()  # [256, 256, 1536]
    
    # Optional: float16 für Speicher sparen
    if PRECISION_HALVING:
        single_feature = single_feature.to(torch.float16)
    
    torch.save(single_feature, output_path)
```

---

## 📚 Weitere Infos

Siehe auch:
- `SHAPE_TRANSFORMATION_EXPLANATION.md` - Detaillierte Erklärung
- `SHAPE_TRANSFORMATION_VISUAL.md` - Visuelle Diagramme
- `README.md` - Offizielle AnySat Dokumentation

---

## 🤔 FAQ

**F: Warum 1536 und nicht 768?**
A: Weil Sub-Patches konkateniert werden (768 + 768 = 1536)

**F: Gilt das auch für andere Output-Typen?**
A: Nein! 
- `output='patch'` → `[num_patches, num_patches, 768]`
- `output='tile'` → `[768]`
- `output='dense'` → `[256, 256, 1536]` ✓

**F: Beeinflusst patch_size die 256×256 Dimension?**
A: Nein! Bei `output='dense'` wird immer zur Original-Größe rekonstruiert.

**F: Kann ich andere Batch Sizes verwenden?**
A: Ja, aber jedes Batch-Element wird einzeln zu `[256, 256, 1536]` verarbeitet.
