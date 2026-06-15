# AnySat S1 Shape-Transformation - Visuelles Diagramm

## 🎨 Grafische Darstellung der Shape-Transformation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   S1 TENSOR SHAPE TRANSFORMATION                            │
└─────────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 1: INPUT PREPARATION
━━━━━━━━━━━━━━━━━━━━━━━━━

   🔹 Original File
   ┌─────────────────────┐
   │   3×256×256         │
   │ (VV, VH, Ratio)     │
   │ Einzelner Tensor    │
   └─────────────────────┘
                ↓
   🔹 Batching mit unsqueeze(1)
   ┌──────────────────────────────┐
   │   4×1×3×256×256              │
   │   B  T  C  H    W            │
   │   Batch of 4 timestamps      │
   └──────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 2: S1 PROJEKTOR
━━━━━━━━━━━━━━━━━━━━

   🔹 Input (Batch)
   ┌──────────────────────────────┐
   │   [4×1×3×256×256]            │
   │   Rohes S1 Signal            │
   └──────────────────────────────┘
                ↓
   ┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
   │  S1 PROJECTOR               │
   │  (Zeitreihen-Embedding)      │
   └─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
                ↓
   🔹 Output (Patches)
   ┌──────────────────────────────┐
   │   [batch*num_patches×        │
   │    num_patches×768]          │
   │                              │
   │   Beispiel:                  │
   │   [256 × 64 × 768]           │
   │                              │
   │   Jeder Patch:               │
   │   768-dimensional Vector     │
   └──────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 3: SPATIAL ENCODER + TRANSFORMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   🔹 Input
   ┌──────────────────────────────┐
   │   [B, num_patches, 768]      │
   │   Patch-level Embeddings     │
   └──────────────────────────────┘
                ↓
   ┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
   │  12× TRANSFORMER BLOCKS      │
   │  (Selbst-Aufmerksamkeit)     │
   └─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
                ↓
   🔹 Output (Mit CLS Token)
   ┌──────────────────────────────┐
   │   [B, num_patches+1, 768]    │
   │                              │
   │   [CLS-Token | Patch-Tokens] │
   │        1        64 Patches    │
   │                              │
   │   Transformierte Features    │
   └──────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 4: DENSE OUTPUT RECONSTRUCTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   🔹 Schritt 4a: Tokens ohne CLS
   ┌──────────────────────────────┐
   │   [B, num_patches, 768]      │
   │   (CLS Token entfernt)       │
   └──────────────────────────────┘
                ↓
   🔹 Schritt 4b: Mit Subpatches kombinieren
   ┌──────────────────────────────┐
   │   [B, num_patches×          │
   │       subpatch_H×            │
   │       subpatch_W×            │
   │       1536]                  │
   │                              │
   │   1536 = 768 + 768           │
   │   (Patch-Features +          │
   │    Sub-Patch-Details)        │
   └──────────────────────────────┘
                ↓
   🔹 Schritt 4c: Komplexe Umformung
   ┌──────────────────────────────┐
   │   [B, 1, D=1536,             │
   │    num_patches,              │
   │    patch_size, patch_size]   │
   │                              │
   │   Räumliche Rekonstruktion   │
   │   Patches werden wieder      │
   │   zusammengefügt             │
   └──────────────────────────────┘
                ↓
   🔹 Schritt 4d: Finale Permutation
   ┌──────────────────────────────┐
   │   [B, 256, 256, 1536]        │
   │                              │
   │   Permute(0,2,3,1)           │
   │   → [256, 256, 1536] pro     │
   │      Batch-Element           │
   └──────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PHASE 5: OUTPUT
━━━━━━━━━━━━━━

   ✅ FINAL OUTPUT
   ┌──────────────────────────────┐
   │   [256, 256, 1536]           │
   │                              │
   │   Räumliche Koordinaten:     │
   │   • x = 0 bis 255 (Breite)   │
   │   • y = 0 bis 255 (Höhe)     │
   │   • Features: 1536 Dims      │
   │                              │
   │   Jeder Pixel hat einen      │
   │   1536-dimensionalen         │
   │   Feature-Vektor!            │
   └──────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🔬 Featur-Dimensionen Breakdown

```
EMBEDDED FEATURES PRO PIXEL
┌──────────────────────────────────────────────────┐
│                                                  │
│  768-dim: Coarse Patch Features                 │
│  ┌────────────────────────────────────────────┐ │
│  │ Transformer-Output für einen Patch         │ │
│  │ (Kontextuelle Information aus 12 Blöcken) │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  ⊕  (Concatenation)                            │
│                                                  │
│  768-dim: Fine Sub-Patch Features              │
│  ┌────────────────────────────────────────────┐ │
│  │ Sub-Patch Details (1×1 Pixel Details)     │ │
│  │ (Räumliche Feinheit, lokale Muster)      │ │
│  └────────────────────────────────────────────┘ │
│                                                  │
│  ═══════════════════════════════════════════    │
│  = 1536-dim FEATURE VECTOR PRO PIXEL            │
│  ═══════════════════════════════════════════    │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## 📋 Parameter Impact auf Output-Shape

```
patch_size Parameter:
┌─────────────────┬─────────────────┬──────────────────┐
│ patch_size (m)  │ scale (intern)  │ Num Patches      │
├─────────────────┼─────────────────┼──────────────────┤
│ 10              │ 1               │ 256 (256×256)    │
│ 20              │ 2               │ 64 (64×64)       │
│ 40              │ 4               │ 16 (16×16)       │
│ 80              │ 8               │ 4 (4×4)          │
└─────────────────┴─────────────────┴──────────────────┘

Aber output='dense' ändert das!
→ Patches werden zu Sub-Pixel-Ebene rekonstruiert
→ Daher immer [256, 256, 1536] für 256×256 Input!
```

---

## 🎓 Warum diese Architektur?

```
┌─────────────────────────────────────────────────┐
│  HIERARCHICAL FEATURE EXTRACTION                │
├─────────────────────────────────────────────────┤
│                                                 │
│  Coarse Level (768-dim):                       │
│  ✓ Globale Kontexte                           │
│  ✓ Langfristige Muster                        │
│  ✓ Patch-übergreifende Informationen          │
│                                                 │
│  Fine Level (768-dim):                        │
│  ✓ Lokale Details (1×1 Pixel)                 │
│  ✓ Sub-Patch Strukturen                       │
│  ✓ Räumliche Genauigkeit                      │
│                                                 │
│  Kombination (1536-dim):                       │
│  ✓ Multi-skalige Feature-Darstellung          │
│  ✓ Ideal für Segmentation Tasks               │
│  ✓ Balanciert zwischen Kontext und Detail     │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 📌 Wichtige Konzepte

### Patches vs. Sub-Patches

**Patches:** 
- Größere räumliche Einheiten (z.B. 40m × 40m)
- Werden vom Transformer verarbeitet
- Liefern kontextuelle Features (768-dim)

**Sub-Patches:** (für S1 Zeitreihen)
- 1×1 Pixel Granularität
- Räumliche Rekonstruktion ermöglicht
- Liefern Fine-Detail Features (768-dim)

### Output Typen im Vergleich

```
output='tile'  → [768]        (eine Vektor pro Bild)
output='patch' → [6,6,768]    (Vektor pro Patch)
output='dense' → [256,256,1536] (Vektor pro Pixel!)
output='all'   → [65,768]     (Patch + CLS Token)
```

Dein Code nutzt `output='dense'` → **Pro-Pixel Features für Segmentation!**
