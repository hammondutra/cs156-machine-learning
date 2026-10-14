# Brawl Stars — CS156 Pipeline Assignment 1

**Research question:** Can the six brawlers in a 3v3 match predict the result better than a prior-only baseline, ideally learning drafting signal rather than the personal player's win tendency?

## Current data (finalized 2026-10-14)

| File | Content |
| --- | --- |
| [data/raw_matches.csv](data/raw_matches.csv) | **150 independent observed matches**; source of truth for training and real-world evaluation |
| [data/augmented_matches.csv](data/augmented_matches.csv) | **300 rows:** the 150 original matches plus side-swapped versions |
| [data/brawler_names.csv](data/brawler_names.csv) | **107 distinct** brawler labels from the final corrected sheet |
| [RESEARCH_LOG.md](RESEARCH_LOG.md) | Chronological collection decisions, pilot results, motivations, limitations, future directions |

The source of truth is the [Google Sheet](https://docs.google.com/spreadsheets/d/1PT7GDSb56HKSjGbvmj5-bGVWCmbP8Wo7-tSkRxEk7aQ/edit). The observed games have **110 Blue wins / 40 losses**, and the mirrored 300-row table has **150/150**. The six observed modes are Brawl Ball (81), Gem Grab (30), Knockout (25), Hot Zone (8), Heist (4), and Basket Brawl (2). No draws were recorded.

### Mirroring and reproducibility

Each match has one `ORIGINAL` and one `FLIPPED` record with the same `Pair_ID`. For flipped records, brawler teams are swapped and `Blue_Win` is inverted; their trophy change remains empty because it was not observed for the other team. Mirrored records are *not independent new samples*. **Do not use random row-wise splits on augmented_matches.csv**; keep paired rows together with grouped cross-validation, or split original matches first and augment only training folds.

Side flipping balances classes but does **not** remove confounding from individual players, game skill, matchmaking, brawler selection or map effects. The target for future experiments is calibrated out-of-sample win probability on real matches, not an assumption of causal draft strength. Never use match outcomes, trophy change, Pair_ID, source metadata or flip flags as model features.

### Analysis status

The previous Jupyter notebook and scripts under `notebooks/` and `src/` were **pilot experiments** using the first 25 matches, where Dynamike and Mortis were heavily overrepresented. They are retained as historical code and are **not yet adapted or approved for the final augmented dataset**. Their previous metrics must not be confused with final-data performance.

**Next:** regenerate the notebook from scratch using the new CSV schema and leakage-safe validation, compare baselines, explore the Brawl Ball subset and report uncertainty. Do not fit Win/Draw/Loss multinomial regression until genuine draw examples are available.

See [RESEARCH_LOG.md](RESEARCH_LOG.md) for the full history and the [Google Docs project plan](https://docs.google.com/document/d/1IPTG9PpUdT5H7X-I0S7bjCjKbauvxZaRpzkvWXfPYQ0/edit).
