---
sidebar_position: 2
title: World simulation
---

# World simulation

## The fictional world

Every `leadforge` dataset is grounded in a fictional but internally consistent commercial world. For v1, that world is:

> **Veridian Technologies**, a mid-market B2B SaaS company selling procurement and AP workflow automation software ("Veridian Procure") to 200–2,000 employee firms in the US and UK, through a mixed inbound, SDR-assisted, and partner-driven go-to-market motion.

The company narrative, product details, buyer personas, and funnel structure are all declared in a **recipe YAML** and rendered into the dataset card, feature descriptions, and metadata.

## Entities

The simulation tracks 9 entity types, mirroring a real CRM:

| Table | What it represents |
|---|---|
| `accounts` | Companies (the buying org) |
| `contacts` | People at each account |
| `leads` | A contact at a specific account entering the funnel |
| `touches` | Outbound and inbound engagement events |
| `sessions` | Website/product sessions |
| `sales_activities` | SDR/AE-logged activities (calls, emails, meetings) |
| `opportunities` | Formal pipeline records |
| `customers` | Post-conversion account status (instructor only) |
| `subscriptions` | Subscription records (instructor only) |

## The latent trait system

Each entity carries a vector of latent traits that are **not directly observable** in the student dataset. Examples:

- `account_fit_score` — how well the account matches the ICP
- `contact_authority` — decision-making authority of the contact
- `problem_awareness` — how aware the buyer is of the problem being solved
- `urgency_score` — time pressure at the account

These traits modulate the probabilities of events and transitions during simulation. The instructor companion exposes them; the student dataset does not.

## Snapshot safety

The primary task is predicting conversion *within 90 days from a snapshot anchor date*. The anchor date is per-lead. All features in the public dataset are computed from events **on or before** the anchor date — this is enforced at rendering time, not by convention.

Columns and tables that would allow label reconstruction via joins (e.g., `customers`, `subscriptions`, terminal-stage opportunity fields) are excluded from the public bundle entirely.
