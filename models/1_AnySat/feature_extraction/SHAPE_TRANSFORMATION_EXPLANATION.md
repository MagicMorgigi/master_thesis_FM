# S1 Tensor Shape-Transformation durch AnySat Model

## 📊 Zusammenfassung der Shape-Transformation

```
Input:  [256, 256, 3]  →  Model  →  Output: [256, 256, 1536]
```

---

## 🔍 Detaillierte Schritt-für-Schritt Transformation

### **SCHRITT 1: Input-Vorbereitung in deinem Code**

```python
# Original tensor aus dem DataLoader
tensor = torch.load(path).float()  # Shape: [3, 256, 256]
# (Channels: VV, VH, VV/VH Ratio; Height, Width)

# Im Batch processing
batch = batch.unsqueeze(1).to(device)  
# Shape wird zu: [B=4, T=1, C=3, H=256, W=256]
```

**Format für AnySat:**
- `B` (Batch Size) = 4
- `T` (Time steps) = 1 (nur ein Zeitpunkt)
- `C` (Channels) = 3 (VV, VH, Ratio)
- `H x W` (Spatial) = 256 × 256 Pixel

---

### **SCHRITT 2: Modell-Konfiguration**

Deine Parameter:
```python
patch_size = 40  # Meter
output = 'dense'
output_modality = 's1'
```

Interne Umwandlung in `hubconf.py`:
```python
scale = patch_size // 10 = 40 // 10 = 4  # Interne Skalierung
```

---

### **SCHRITT 3: S1 Projektor & Spatial Encoder**

Der S1 **Projektor** verarbeitet die Zeitreihe und projiziert sie in einen **höherdimensionalen Embedding-Raum**.

**Nach dem Projektor:**
- Input: `[B=4, T=1, C=3, 256, 256]`
- Output: `[B*N_patches, N_patches, embed_dim]`
  - Beispiel bei patch_size=4: `[4*64, 64, 768]` = `[256, 64, 768]`
  - `N_patches` = `(256/4)² = 64² = 4096` ...Moment, das passt nicht. Schauen wir genauer hin.

**Korrekt:** Der Projektor teilt die 256×256 Bilder in Patches auf:
- Bei `patch_size=4m` (scale=4 intern): Patches der Größe `patch_size × patch_size` Pixel
- Mit S1 Resolution von 10m und scale=4: Effektive patch_größe = 40m
- Anzahl Patches pro Dimension: `256 / patch_pixel_size = 6 × 6 = 36` oder ähnlich
  
*(Die genaue Patch-Größe hängt von der Modell-Konfiguration ab)*

**Nach Spatial Encoder:**
- Output: `[B, N, embed_dim]` wobei N = Anzahl Patches
- Shape: `[4, num_patches, 768]`

---

### **SCHRITT 4: Transformer-Blöcke**

Die Patch-Tokens durchlaufen 12 Transformer-Blöcke (für base model):
- **Eingabe:** `[B, num_patches+1, 768]` (mit CLS Token)
- **Ausgabe:** `[B, num_patches+1, 768]` (gleiche Dimension, aber transformierte Features)

---

### **SCHRITT 5: Subpatch-Rekonstruktion (Dense Output)**

Dies ist der **Schlüssel** für `output='dense'`:

```python
# 1️⃣ Tokens aus Transformer extrahieren (ohne CLS Token)
tokens = tokens[:, 1:, :]  # Shape: [B, num_patches, 768]

# 2️⃣ Mit Subpatch-Features kombinieren
# Subpatches enthalten Details auf Sub-Patch-Ebene (1×1 Pixel für Zeit-Serien)
# tokens wird erweitert: [B, num_patches, 768, subpatch_count]
tokens = tokens.unsqueeze(2).repeat(1, 1, out['subpatches'].shape[2], 1)

# 3️⃣ Konkatenation mit Subpatch-Features
dense_tokens = torch.cat([tokens, out['subpatches']], dim=3)
# Nach Concat: [B, num_patches, subpatch_count, 1536]
# ↑ 1536 = 768 + 768 (zwei 768-dim Features kombiniert)

B, N, P, D = dense_tokens.shape
# B=4, N=num_patches, P=subpatch_pixel_size, D=1536

# 4️⃣ Umformen zu räumlicher Darstellung
patch_size = int(P**(1/2))  # Z.B. patch_size=256 wenn P=256²
size = num_patches * patch_size  # Finale Bildgröße

# 5️⃣ Komplexe Umformungen zur räumlichen Rekonstruktion
dense_tokens = dense_tokens.view(B, 1, D, N, patch_size, patch_size)
dense_tokens = dense_tokens.view(B, 1, D, num_patches, num_patches, patch_size, patch_size)
dense_tokens = dense_tokens.permute(0, 1, 2, 3, 5, 4, 6)
# Rekonstruktion: Patches werden wieder in räumliche Anordnung gebracht

# 6️⃣ Finale Umformung
dense_tokens = dense_tokens.reshape(B, 1, D, size, size)
dense_tokens = dense_tokens.flatten(0, 1)  # Vereinige Batch und "1"
dense_tokens = dense_tokens.permute(0, 2, 3, 1)  # Permutation!
# Shape: [B, size, size, D] = [B, 256, 256, 1536]
```

**Finale Shape:** `[B=4, H=256, W=256, D=1536]`

---

### **SCHRITT 6: Speicherung (in deinem Code)**

```python
for i in range(len(filenames)):
    single_feature = batch_features[i].cpu()  
    # Shape: [256, 256, 1536]
    # ✅ Passt zu expected_dims_sorted = [256, 256, 1536]!
    
    # Optional: Konvertierung zu float16
    if PRECISION_HALVING:
        single_feature = single_feature.to(torch.float16)
    
    torch.save(single_feature, output_path)
```

---

## 📐 Dimension-Erklärung

| Dimension | Bedeutung | Größe |
|-----------|-----------|-------|
| **256** | Bildbreite (Pixel) | 256 |
| **256** | Bildhöhe (Pixel) | 256 |
| **1536** | Feature-Dimension | 2 × 768 |

Die 1536-dimensionalen Features für jeden Pixel bestehen aus:
- **768 dims** von den Patch-Level Features (Transformer-Output)
- **768 dims** von den Sub-Patch-Level Details (räumliche Feinheiten)

---

## 🎯 Warum 1536 und nicht 768?

Das AnySat-Modell nutzt eine **Hierarchische Feature-Extraction**:

1. **Coarse Features (768):** Patch-basierte Merkmale (größere räumliche Kontexte)
2. **Fine Features (768):** Sub-Patch Features (1×1 Pixel Details für S1)

Diese werden **konkateniert**, um ein reichhaltiges Feature-Beschreibung pro Pixel zu erhalten.

---

## 💡 Zusammenfassung für dich

```python
# Was passiert intern:
model(data, patch_size=40, output='dense', output_modality='s1')

# 1. S1 Input [B, 1, 3, 256, 256] wird in Patches zerlegt
# 2. Patches werden durch Transformer verarbeitet → 768-dim Features
# 3. Für jeden Pixel werden Sub-Patch Details extrahiert → 768-dim Features  
# 4. Beide kombiniert → 1536-dim Features pro Pixel
# 5. Räumlich rekonstruiert → [B, 256, 256, 1536]
# 6. Pro Bild gespeichert → [256, 256, 1536]
```

---

## 🔧 Konfigurationsparameter

- **patch_size**: 40m (bestimmt Patch-Auflösung)
- **output**: 'dense' (Pro-Pixel Output statt Pro-Patch)
- **output_modality**: 's1' (Welche Modalität für Sub-Patches nutzen)
- **embed_dim**: 768 (für base model)
- **Finale Features**: 2×embed_dim = 1536 (Patch + Sub-Patch kombiniert)

