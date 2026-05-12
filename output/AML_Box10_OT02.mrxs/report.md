# WSI Agent Report (20260512_100134_74d30c)

## Prompt

```text
You are performing morphology-only triage on May-Grünwald-Giemsa stained bone marrow ROI images.

INPUT:
You are given ROI images selected by a separate ROI collection agent.
Do NOT assume these ROIs prove AML.
Do NOT assume accepted ROIs are diagnostic.
First re-evaluate each ROI for field type and suitability for blast estimation.

PRIMARY GOAL:
Estimate blast percentage among interpretable nucleated hematopoietic cells only.

SECONDARY GOAL:
Only if morphology supports Acute leukemia, provide a morphology-based NPM1 prediction.

This is ROI-based morphology triage. First reassess ROI suitability, then estimate blasts among interpretable nucleated hematopoietic cells only. Exclude RBCs, fat, empty/hemodilute areas, crush artifact, debris, and uninterpretable overlap.

Base the global blast range on diagnostically informative ROIs, not a simple average of all submitted fields. Do not count scattered mimics or artifacts as blasts without immature nuclear features; residual maturation does not exclude AML if a coherent expanded blast/blast-equivalent population is present.

--------------------------------
ROI SUITABILITY
--------------------------------
Accept an ROI for blast estimation only if it contains interpretable nucleated hematopoietic cells.

Ignore for blast estimation:
- RBCs
- fat spaces
- empty areas
- hemodilute areas
- crushed/streaked material
- necrotic debris
- stain precipitate
- stromal fragments
- platelet/fibrin aggregates
- thick overlapped regions where individual nuclei cannot be resolved

Discard or down-weight ROIs dominated by:
- RBC-only/hemodilute material
- fat/empty areas
- crush artifact
- debris
- stromal fragments
- macrophage-only clusters
- uninterpretable dark overlapped marrow particles

Poor-quality or hemodilute ROIs should not lower the global blast range if other accepted ROIs are blast-rich.

--------------------------------
BLAST / BLAST-EQUIVALENT IDENTIFICATION
--------------------------------
Count as blast-like if classic blast morphology OR immature monocytic/blast-equivalent morphology is present.

Classic blast morphology:
- high N:C ratio
- round/oval nucleus
- fine/open chromatin
- visible nucleolus or nucleoli in at least a subset
- scant to moderate agranular cytoplasm
- immature nuclear appearance

Immature monocytic / blast-equivalent morphology:
- immature chromatin compared with mature monocytes
- folded, indented, lobulated, or irregular nuclei
- relatively abundant gray-blue cytoplasm
- nucleoli may be subtle or absent in some cells
- coherent expanded population, not isolated cells

Auer rod:
- Definite Auer rod in an immature myeloid cell supports AML strongly.
- Do not require high blast percentage if definite Auer rods are present in immature cells.

Do NOT require every blast-like cell to show all criteria.
Evaluate whether there is a coherent expanded immature population.

--------------------------------
DO NOT OVERCALL THESE MIMICS AS BLASTS
--------------------------------
Do NOT count the following as blasts unless immature nuclear features are clear:

1. Maturing granulocytes:
- segmented neutrophils
- bands
- metamyelocytes
- myelocytes
- promyelocytes with obvious cytoplasmic granulation

2. Erythroid precursors:
- round nuclei with dense/coarse chromatin
- deeply basophilic cytoplasm
- small-to-medium size
- clustered erythroid islands

3. Monocytes/macrophages/histiocytes:
- abundant gray/blue or vacuolated cytoplasm
- mature chromatin
- foamy cytoplasm
- phagocytosed debris, pigment, or hemosiderin
- isolated large macrophages

4. Megakaryocytes / bare megakaryocyte nuclei:
- very large cells or nuclei
- multilobated, smudged, or hyperchromatic nuclei

5. Artifacts:
- dark crushed streaks
- stain precipitate
- ruptured cells
- naked nuclei
- apoptotic bodies
- thick overlapped particles

Color intensity alone is never sufficient.
Large size alone is never sufficient.
A dark purple cell is not a blast unless nuclear immaturity is visible.

--------------------------------
BLAST ESTIMATION
--------------------------------
Estimate blasts only among interpretable nucleated hematopoietic cells.
Ignore RBCs, fat, empty spaces, artifacts, and uninterpretable material.

Use EXACT tiers:
<5%, 5-9%, 10-19%, 20-50%, >50%

Per ROI:
- Estimate the blast-like/blast-equivalent fraction among interpretable nucleated hematopoietic cells.
- Do not upgrade to >=20% unless approximately at least 1 in 5 interpretable nucleated cells are convincingly blast-like/blast-equivalent.
- Do not upgrade to >50% unless blasts/blast-equivalents are the dominant nucleated population.

Global blast range:
- Use the best diagnostically informative accepted ROI pattern, not a simple average across all ROIs.
- If one or more high-quality accepted ROIs are blast-rich, global_blast_range may be 20-50% or >50% even if other ROIs are hemodilute, fatty, or blast-poor.
- If all accepted ROIs show only scattered blast-like cells with mixed maturation, global_blast_range should remain <5%, 5-9%, or 10-19% as appropriate.

--------------------------------
PRIMARY DECISION: BALANCED BINARY RULE
--------------------------------
Choose ONE final_decision only:
- "Acute leukemia"
- "Normal marrow"

Call "Acute leukemia" if ANY of the following are true:
1. Any accepted diagnostically informative ROI shows >=20% convincing blasts/blast-equivalents among interpretable nucleated hematopoietic cells.
2. The best interpretable marrow areas across ROIs show a coherent expanded immature/blast-equivalent population estimated at >=20%.
3. Several accepted ROIs show 10-19% blast-like/blast-equivalent cells AND the overall pattern is monotonous, immature, or blast-rich rather than normally maturing.
4. Definite Auer rods are present in immature myeloid cells.
5. There is a dominant population of immature monocytic/blast-equivalent cells with immature chromatin, folded/irregular nuclei, and abundant gray-blue cytoplasm.

Call "Normal marrow" if ANY of the following best describes the case:
1. Blasts are clearly <5%.
2. Blast-like cells are scattered only and do not form a coherent expanded immature population.
3. Apparent blasts are better explained by maturing granulocytes, erythroid precursors, promyelocytes, monocytes/macrophages, megakaryocytes, crushed cells, or artifact.
4. ROIs are largely hemodilute, fatty, or non-diagnostic without convincing blast-rich areas.
5. Mixed heterogeneous hematopoietic maturation predominates and blast-like cells are not clearly >=20%.

For 5-19% estimated blasts:
- Choose "Acute leukemia" if the field pattern is closer to diffuse immature/blast-rich disease than reactive or normally maturing marrow.
- Choose "Normal marrow" if maturation is heterogeneous and blast-like cells are only scattered.
- Explain the uncertainty in limitations_confidence.

Important:
Residual granulocytic or erythroid maturation does NOT exclude AML.
A single high-quality blast-rich ROI can support AML.
Do not require every ROI to be blast-rich.
Do not diagnose AML from ROI selection alone.

--------------------------------
SECONDARY TASK: NPM1 PREDICTION ONLY IF AML
--------------------------------
Execute ONLY if final_decision = "Acute leukemia".

Assess whether morphology is suggestive of NPM1 mutation using ONLY morphology.

Features supporting NPM1_mutated:
- Blasts with relatively abundant cytoplasm
- Folded, indented, cup-like, or irregular nuclei
- Cup-like nuclear invaginations
- Less primitive appearance than classic blasts
- Monocytic differentiation / immature monocytic morphology
- Auer rods do NOT exclude NPM1 mutation

Features supporting NPM1_wildtype:
- Predominantly classic primitive blasts without NPM1-like nuclear features
- Very high N:C ratio with scant cytoplasm and no monocytic/cup-like features
- Features insufficient or conflicting for NPM1-mutated morphology

Constraints:
- Morphology-only prediction is probabilistic.
- If ROI quality is insufficient, lower confidence.
- Do not perform NPM1 prediction unless AML is established morphologically.

Classification REQUIRED only if applicable:
- NPM1_mutated
- NPM1_wildtype

Confidence levels:
- high = multiple concordant features
- moderate = partial features
- low = weak, conflicting, or poor-quality features

--------------------------------
OUTPUT: STRICT JSON ONLY
--------------------------------
{
  "morphology_summary": "string (1-4 sentences)",
  "accepted_rois": [
    {
      "roi_id": "string",
      "quality_reason": "string",
      "blast_range": "<5% | 5-9% | 10-19% | 20-50% | >50%",
      "key_features": ["string"]
    }
  ],
  "discard_summary": ["string"],
  "global_blast_range": "<5% | 5-9% | 10-19% | 20-50% | >50%",
  "triage_zone": "normal_like | borderline_suspicious | aml_like",
  "final_decision": "Normal marrow | Acute leukemia",
  "limitations_confidence": {
    "limitations": "string",
    "confidence": "low | medium | high"
  },
  "npm1_prediction": {
    "applicable": true,
    "classification": "NPM1_mutated | NPM1_wildtype",
    "confidence_level": "low | moderate | high",
    "supporting_features": ["string"],
    "comment": "string"
  }
}

--------------------------------
CONDITIONAL RULE
--------------------------------
If final_decision != "Acute leukemia":
"npm1_prediction": {
  "applicable": false,
  "classification": null,
  "confidence_level": null,
  "supporting_features": [],
  "comment": "NPM1 prediction not performed because AML is not established morphologically"
}

--------------------------------
FORMAT RULES
--------------------------------
- JSON ONLY.
- No hallucinated findings.
- Keep concise.
- Do not mention features that are not visible.
- Do not diagnose AML from ROI selection alone.

Return one strict JSON object only. Do not return markdown, prose, or code fences.

Return one strict JSON object only. Do not return markdown, prose, or code fences.
```

## Final Report

{
  "morphology_summary": "The marrow is hypercellular with a predominant population of cells showing high N:C ratios, fine chromatin, and visible nucleoli. There is a marked lack of normal granulocytic maturation, with the nucleated cells appearing monotonous and blast-like.",
  "accepted_rois": [
    {
      "roi_id": "1",
      "quality_reason": "Contains interpretable nucleated hematopoietic cells despite some fat spaces.",
      "blast_range": ">50%",
      "key_features": [
        "High N:C ratio",
        "Fine open chromatin",
        "Visible nucleoli",
        "Monotonous population"
      ]
    },
    {
      "roi_id": "2",
      "quality_reason": "High density of interpretable nucleated hematopoietic cells.",
      "blast_range": ">50%",
      "key_features": [
        "Diffuse infiltration of blasts",
        "Lack of maturation",
        "Immature nuclear morphology"
      ]
    }
  ],
  "discard_summary": [],
  "global_blast_range": ">50%",
  "triage_zone": "aml_like",
  "final_decision": "Acute leukemia",
  "limitations_confidence": {
    "limitations": "None; morphology is highly characteristic of acute leukemia.",
    "confidence": "high"
  },
  "npm1_prediction": {
    "applicable": true,
    "classification": "NPM1_wildtype",
    "confidence_level": "moderate",
    "supporting_features": [
      "Predominantly classic primitive blasts",
      "High N:C ratio",
      "Scant cytoplasm"
    ],
    "comment": "Blasts appear more primitive/classic without prominent cup-like nuclei or abundant monocytic cytoplasm typical of NPM1 mutations."
  }
}

## Regions of Interest (ROIs)

_No ROIs were kept in this run._

## Navigation Steps

_No navigation steps recorded._
