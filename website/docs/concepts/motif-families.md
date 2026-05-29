---
sidebar_position: 3
title: Motif families
---

# Motif families

`leadforge` deliberately avoids a single fixed data-generating process (DGP). Instead, the hidden world is sampled from one of five **motif families**, then stochastically rewired. This ensures that:

1. Different dataset instances have genuinely different causal structures.
2. No single feature engineering recipe is universally optimal.
3. The true DGP is verifiable via the instructor companion.

## The five families

### `fit_dominant`
Account and ICP fit traits are the primary path to conversion. High-fit accounts convert at much higher rates regardless of engagement volume. Feature engineering that captures account-level firmographics will dominate.

### `intent_dominant`
Buying intent signals — session depth, demo requests, content downloads, direct inquiries — are the main driver. Fit alone is insufficient; conversion requires observable interest signals. Engagement-based features and recency weighting matter most.

### `sales_execution_sensitive`
SDR responsiveness, AE follow-through, and meeting-to-proposal timing are the dominant levers. Two otherwise-identical leads have very different outcomes depending on how quickly and consistently they were worked. Activity cadence features are the key signal.

### `demo_trial_mediated`
Conversion is causally gated on a demo or trial event. Leads that never reach a demo rarely convert; leads that do have high conversion probability. Models that can identify "reached demo" or "trial active" as a key pathway will perform well.

### `buying_committee_friction`
Multi-stakeholder dynamics create the primary noise. A lead may have high fit, intent, and SDR attention, but stall because a procurement or finance stakeholder raised objections. Contact-level authority and multi-touch diversity features matter.

## Stochastic rewiring

After sampling a motif family, the graph is subjected to stochastic rewiring: edges are added or removed with small probabilities, and edge weights are perturbed. This means no two generated bundles have exactly the same graph even within the same motif family, while the family-level character is preserved.

## Identifying the motif family

The motif family is **not disclosed** in the student bundle. It is recorded in `metadata/world_spec.json` (instructor mode only). Breaking the dataset — inferring the motif family from the student features — is one of the intended challenges.
