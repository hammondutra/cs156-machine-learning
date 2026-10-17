# Research and Data Collection Log — CS156 Assignment 1

Updated: 2026-10-14. This document records the project's evolution and the decisions and limitations that motivated it. It is a research log, not a claim that composition effects are causal.

## Research objective

Predict the outcome of a 3v3 Brawl Stars match from both teams' three-brawler compositions. The long-term aim is to estimate drafting strength independent of the particular player's skill. Current matches contain only win/loss outcomes; a win/draw/loss softmax requires genuine draw observations.

## Stage 1 — Personal battle-history pilot (25 real matches)

- Started from the last 25 battles available through the player's battle history, manually extracting six brawlers, team outcome, trophies, mode and map.
- These were all Brawl Ball on Pinhole Punt, and were heavily concentrated on Dynamike and Mortis: across the **50 team lineups** (150 brawler slots), Dynamike appeared 26 times and Mortis 21 times. The first 25 matches contained 15 wins and 10 losses.
- This was a serious generalizability problem. A classifier could largely memorize a narrow part of the brawler pool and one map; it could not reliably estimate a general drafting meta.
- The first four out-of-fold binary baseline experiments (25 matches, stratified five-fold CV) were almost tied:
  - class-prior log loss 0.6730;
  - single Blue-slot-1 proxy 0.6766;
  - signed full composition 0.6773;
  - separate-side composition 0.6727.
  - All achieved 60% accuracy and 50% balanced accuracy at the 0.5 threshold. This was not convincing evidence of meaningful composition learning. These are preliminary results **only for the 25-match pilot**, not results from the final 150-match dataset.

## Stage 2 — Diversity-oriented collection

- The player deliberately played approximately 108 additional games, aiming to use **one different playable brawler per game**, with random teammates/opponents ('randoms') and a broad, largely randomized selection of common modes.
- This strategy was intended to spread the player's own brawler choices more widely, instead of repeatedly favoring Dynamike/Mortis, and to sample varied teams.
- Game-mode coverage includes Brawl Ball, Knockout, Gem Grab, Hot Zone, Heist and Basket Brawl. The player recalls occasional Brawl Hockey; the current final CSV's mode field contains **no row explicitly labeled Brawl Hockey**, so it is not counted as a separately observed mode here.
- Even in an ideal randomized sample, 108 games produce 648 total brawler slots (108 × 6), or about six appearances per brawler if the 108-character pool were sampled uniformly across all slots. In practice the **player's own** choices may be evenly spread while teammates and opponents are not. Six observations per character are generally too few to robustly estimate its conditional contribution, let alone multi-brawler synergy.
- The actual corrected final 150-match dataset contains **107 distinct brawler labels**; this is the number observed, not a verified count of all playable brawlers at collection time.
- Data entry evolved through several screenshots and staged additions: 25 → 47 → 72 → 97 → 122 → 147 matches, followed by three manually entered games, for **150 genuine matches**. Records were checked against prior observations for duplicates when added, although absence of a reliable unique battle ID limits certainty. Names were manually corrected and standardized in Google Sheets.

## Stage 3 — Narrower single-mode experiment

- Recognizing that the broad, sparse sample might provide little mode-specific predictive power, the player later concentrated on approximately 20–30 more **Brawl Ball** games to seek a more coherent mode-specific composition signal.
- The working assumption is to pool different Brawl Ball maps as though their drafting effects were broadly comparable. This may be too strong: map geometry, objectives, terrain, and balance patches can change the effective meta. The final dataset retains exact map names so future models can test or relax this assumption.
- Final observed mode breakdown: **Brawl Ball 81, Gem Grab 30, Knockout 25, Hot Zone 8, Heist 4, Basket Brawl 2** (sum = 150). For future analyses, compare an all-modes model, a Brawl Ball subset, and ideally features conditioned on mode/map. Note the Brawl Ball subset may remain small for 107 observed characters.

## Stage 4 — Player-dependent side imbalance and mirroring

- Across collection sessions the player's observed win rate often ranged roughly **64–80%**, consistent with their report of previously playing semi-competitively against largely casual lobbies while queueing with randoms.
- This causes concern that a model may mainly learn to predict that the player's side wins, not how strong each draft is. In the final **150 original matches** the side recorded as Blue won **110** games and lost **40**, a 73.3% win rate.
- Decision: for every observed match, generate a **side-swapped synthetic row** with the Blue and Red compositions exchanged and outcome flipped (Blue_Win' = 1 - Blue_Win). 150 original matches + 150 flipped rows = 300 tabular observations with precisely 150 Blue wins and 150 Blue losses.
- Important correction: This is **side-symmetry augmentation and class balancing**, not removal of player-skill bias or creation of 150 new independent match outcomes. Player ability, team skill, matchmaking difficulty, preferred picks and participation remain potential confounders.
- The original rows retain observed trophy changes. Synthetic flipped rows intentionally have blank Trophy_Change because the opponent's trophy change was not observed.
- Pair_ID connects an original with its mirror; Is_Flipped (0/1) and Record_Type (ORIGINAL/FLIPPED) explain provenance. Group the two rows together in train/test or CV to prevent leakage. Better: split the 150 actual matches first, then derive flips **only within each training fold**, evaluating on held-out original matches as a real-world check.
- Signed representation, x_b = 1(b on Blue) − 1(b on Red), is naturally anti-symmetric under side swaps. An appropriately constrained zero-intercept binary logistic model also yields P(Blue wins | B,R)=1−P(Blue wins | R,B). This mathematical symmetry can reduce dependence on artificial augmentation.

## Stage 5 — External validity and possible future datasets

- To separate composition quality from the player's own ability, collect battles from **many different players**, ideally with unique match identifiers, timestamps, patch versions and known player/brawler roles.
- A more controlled sample could consider high-ranked or competitive matches, analogous to examining grandmaster chess games, e.g. a defined Masters-level cutoff or serious competitive play.
- High-ranked data may improve relevance to competitive drafting, but changes the target population, introduces its own selection biases, and does not automatically establish causal composition quality. Compare performance on different skill brackets and balance patches.
- Later tests: (1) class prior; (2) single-brawler features; (3) signed full composition; (4) separate-side features; optionally brawler-pair synergies when sample size permits. Evaluate on unseen real matches with log loss, Brier score, balanced accuracy and calibration, and report sample uncertainty.

## October 14 dataset snapshot and integrity

Source of truth: [Google Sheets Battles tab](https://docs.google.com/spreadsheets/d/1PT7GDSb56HKSjGbvmj5-bGVWCmbP8Wo7-tSkRxEk7aQ/edit).

- `data/raw_matches.csv`: **150 real observations**, including the three manually added matches.
- `data/augmented_matches.csv`: **300 rows**, all 150 originals followed by 150 flipped records.
- `data/brawler_names.csv`: **107 distinct canonical labels** as currently entered.
- All 150 mirrored pairs have been verified for team swap, inverted outcome, matching map/mode/source and matched Pair_ID.
- There are 110/40 wins/losses in observed originals and 150/150 in the augmented set.
- The original dataset has no draws. There are no remaining `REVIEW_` placeholder names.
- Treat unique battle count as 150, not 300. Do not train on Result, Trophy_Change, Pair_ID, Record_Type, Is_Flipped, Source_Image or Source_Row as input features.
- Data is manually transcribed and is not a formally audited feed. Some match metadata remains limited by screenshots; preserve unknown values rather than infer them.

## Next step

Regenerate the assignment Jupyter notebook from scratch, using these final dataset files. The old prototype notebook was built around an earlier 25-match pilot and must not be presented as final analysis or used with naive row-wise CV on mirrored data.

## October 17 — Additional Brawl Ball screenshots (deduplicated)

- The player supplied three additional battle-history screenshots plus a re-upload containing previously omitted matches. Several screenshot rows overlapped with one another and with already-recorded Sunny Soccer and Super Beach matches, so a full screenshot count would exaggerate the number of independent observations.
- We compared the six-character compositions (treating each team's portrait order as arbitrary), outcome, mode and map to the 150 original matches and deduplicated the overlapping screenshots before import.
- **12 previously unrecorded Brawl Ball matches** remained: 7 on Triple Dribble and 5 on Pump It Up, of which **11 were victories and 1 was a defeat**.
- These real matches were added with `Match_ID`/`Pair_ID` **151–162**; their synthetic mirrors have `Match_ID` **313–324** after arranging all 162 originals first and all 162 mirrored rows afterward. Mirror `Pair_ID` is identical to its original's ID.
- One new original (match 159) has a blank trophy change because the value was not legible; no trophy change was inferred for any mirrored row.
- **The working dataset now contains 162 actual matches (121 wins, 41 losses), with 107 distinct brawlers.** It also contains 162 synthetic side-swapped matches, for **324 total rows (162 wins, 162 losses)**.
- The Brawl Ball subset increased from **81 to 93 independent matches**. Other recorded modes retained their original counts.
- Mirror invariants were re-validated for all 162 pairs: composition side-swap, inverted outcome and matching `Pair_ID`. The final data are published in `data/raw_matches.csv` and `data/augmented_matches.csv`.
- This extension does not create new independent outcomes through mirroring. Future validation must still keep matched mirrors within the same training fold (or add mirrors only to the training set after splitting the original matches).
- The previous sections record the project as it evolved; superseded sample-size figures there are **historical**, not the current dataset totals.
