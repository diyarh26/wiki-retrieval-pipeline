# Full Query Diagnostic Report - Current Code

Branch: `exp/improve-reranker-rebalance-clean`
Commit: `a8bc2890f4c3f413fa110c51c34fb6e395688895`
Public score: `mean_ndcg@10=0.4501`
Diagnostic run time: `27.08s` using `retrieve.debug_search_batch`
Verification note: official eval on this branch also reproduced `mean_ndcg@10=0.4501`.

This report uses public labels for diagnostics only. It is not retrieval logic.

## Summary

| Category | Count | Queries |
| --- | ---: | --- |
| Perfect (1.0) | 7 | Q0, Q1, Q2, Q8, Q9, Q12, Q15 |
| Good (>=0.5) | 5 | Q4, Q11, Q21, Q25, Q26 |
| Medium (>=0.3) | 5 | Q3, Q13, Q14, Q16, Q23 |
| Low (>0, <0.3) | 7 | Q6, Q7, Q17, Q18, Q19, Q20, Q24 |
| Zero (0.0) | 5 | Q5, Q10, Q22, Q27, Q28 |

## Key Opportunities

Relevant pages currently ranked 11-30 are the most plausible reranking-rescue targets:

| Query | Current NDCG | Close relevant pages |
| --- | ---: | --- |
| Q3 | 0.4945 | 26566@25 |
| Q5 | 0.0000 | 4065@16 |
| Q7 | 0.1900 | 33775@16, 34690@21 |
| Q17 | 0.0634 | 47196@20, 10052@22, 36551@25, 19684@28, 32656@30 |
| Q18 | 0.2487 | 13762@13, 18759@17, 15413@25 |
| Q19 | 0.0608 | 11105@13, 44582@25, 40903@27 |
| Q20 | 0.1771 | 3511@14 |
| Q23 | 0.3890 | 37066@11, 7026@19, 2522@22, 31510@26, 36881@27, 8722@30 |
| Q24 | 0.1771 | 31602@16, 40201@17, 35560@21, 12699@27 |
| Q26 | 0.5701 | 36021@19 |
| Q27 | 0.0000 | 34448@13, 4114@18, 45012@23 |
| Q28 | 0.0000 | 45655@26 |

Largest recall gaps inside the inspected rerank pool:

| Query | Missing relevant pages | Missing page IDs |
| --- | ---: | --- |
| Q27 | 3/7 | 7041, 17966, 39400 |
| Q22 | 3/6 | 27139, 33570, 48438 |
| Q20 | 2/5 | 3327, 3481 |
| Q19 | 2/9 | 41307, 50630 |
| Q18 | 1/9 | 11065 |
| Q17 | 1/10 | 26564 |
| Q6 | 1/2 | 50923 |

## Perfect (1.0) Queries - 7

### Q0 (q_public_0)

Who was the point guard that won a seven-game finals series in the 1820s?

NDCG@10 = `1.0000` | relevant pages = `1` | top-10 found = `1/1` | found in rerank pool = `1/1`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | yes | 20263 | Ulric Isenmar |
| 2 | no | 36551 | 1905 Lunaris Comets season |
| 3 | no | 16735 | Levi Northcott |
| 4 | no | 4418 | Xander Lannick |
| 5 | no | 51300 | Greta Aldridge |
| 6 | no | 30087 | 1893 Dorsaly Raptors season |
| 7 | no | 55029 | Larry Bird |
| 8 | no | 22668 | Tallmere Memorial Arena |
| 9 | no | 44461 | Rhea Mercer |
| 10 | no | 36753 | Magic Johnson |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 1 | 20263 | Ulric Isenmar |

Diagnosis: All relevant pages are already in the returned top 10.

### Q1 (q_public_1)

Who captained the Los Angeles basketball franchise when they won the 1987 championship?

NDCG@10 = `1.0000` | relevant pages = `1` | top-10 found = `1/1` | found in rerank pool = `1/1`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | yes | 9112 | Tim Jordan |
| 2 | no | 3305 | Los Angeles Lkers |
| 3 | no | 31400 | 1999 Los Angeles Lkers season |
| 4 | no | 16899 | Kareem Abdul-Jabbar |
| 5 | no | 6983 | Rosa Northcott |
| 6 | no | 18213 | Los Angeles Dodgers |
| 7 | no | 36753 | Magic Johnson |
| 8 | no | 27170 | Los Angeles Chargers |
| 9 | no | 20263 | Ulric Isenmar |
| 10 | no | 22093 | National Basketball Association |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 1 | 9112 | Tim Jordan |

Diagnosis: All relevant pages are already in the returned top 10.

### Q2 (q_public_2)

What river delta municipality has about 1,456,779 residents?

NDCG@10 = `1.0000` | relevant pages = `1` | top-10 found = `1/1` | found in rerank pool = `1/1`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | yes | 25051 | Stoneford |
| 2 | no | 8124 | Delta (letter) |
| 3 | no | 50923 | Westmere |
| 4 | no | 19579 | Mississippi River |
| 5 | no | 47988 | Godavari River |
| 6 | no | 37392 | Sacramento River |
| 7 | no | 46571 | Batavia (region) |
| 8 | no | 6613 | Yangtze |
| 9 | no | 55460 | Doñana National Park |
| 10 | no | 3454 | Bangladesh |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 1 | 25051 | Stoneford |

Diagnosis: All relevant pages are already in the returned top 10.

### Q8 (q_public_8)

Who chaired preliminary peace talks the year before a September 1958 signing?

NDCG@10 = `1.0000` | relevant pages = `1` | top-10 found = `1/1` | found in rerank pool = `1/1`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 24319 | Peace Now |
| 2 | yes | 37638 | Ironquay Armistice |
| 3 | no | 21400 | Meridian Accord of Ironquay |
| 4 | no | 9172 | Treaty of Brunevale |
| 5 | no | 42062 | Meridian Accord of Ironquay |
| 6 | no | 26610 | Confederation of Frosthaven |
| 7 | no | 3310 | Holloway Settlement |
| 8 | no | 3852 | Dorsaly Concord |
| 9 | no | 11196 | Elmswick |
| 10 | no | 18097 | Negotiations for the Rivermark Compromise |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 2 | 37638 | Ironquay Armistice |

Diagnosis: All relevant pages are already in the returned top 10.

### Q9 (q_public_9)

What agreement established trade corridors and a joint commission in 1965?

NDCG@10 = `1.0000` | relevant pages = `1` | top-10 found = `1/1` | found in rerank pool = `1/1`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | yes | 3267 | Treaty of Brunevale |
| 2 | no | 13762 | Negotiations for the Northaven Charter |
| 3 | no | 11196 | Elmswick |
| 4 | no | 26610 | Confederation of Frosthaven |
| 5 | no | 30815 | Congress of Mossenden |
| 6 | no | 51609 | Negotiations for the Congress of Mossenden |
| 7 | no | 49163 | Congress of Mossenden |
| 8 | no | 7997 | Rivermark Compromise |
| 9 | no | 42062 | Meridian Accord of Ironquay |
| 10 | no | 4815 | Negotiations for the Northaven Charter |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 1 | 3267 | Treaty of Brunevale |

Diagnosis: All relevant pages are already in the returned top 10.

### Q12 (q_public_12)

Who led the group that published reproducible laboratory results on stress imaging through harmonic vibration analysis?

NDCG@10 = `1.0000` | relevant pages = `1` | top-10 found = `1/1` | found in rerank pool = `1/1`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 45160 | Harrowgate Institute of Applied Physics |
| 2 | yes | 1904 | Petra Lannick |
| 3 | no | 17672 | Lior Fenridge |
| 4 | no | 13638 | Levi Crandale |
| 5 | no | 8722 | Petra Zeller |
| 6 | no | 42955 | Quinn Jurvan |
| 7 | no | 14267 | Dorsaly Field Research Campus publications |
| 8 | no | 37066 | Rhea Jardine |
| 9 | no | 31510 | Mossenden Technical Institute publications |
| 10 | no | 2728 | Dorsaly Field Research Campus |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 2 | 1904 | Petra Lannick |

Diagnosis: All relevant pages are already in the returned top 10.

### Q15 (q_public_15)

Who led the group that published reproducible laboratory results on phase-sensitive radiometry in orbital configurations?

NDCG@10 = `1.0000` | relevant pages = `1` | top-10 found = `1/1` | found in rerank pool = `1/1`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 33315 | Orbital Phase Radiometry |
| 2 | yes | 14901 | Tamsin Severin |
| 3 | no | 50946 | Soren Crandale |
| 4 | no | 31510 | Mossenden Technical Institute publications |
| 5 | no | 37066 | Rhea Jardine |
| 6 | no | 28339 | Mossenden Technical Institute |
| 7 | no | 18779 | Mossenden Technical Institute |
| 8 | no | 43574 | Juno Kingsley |
| 9 | no | 16195 | Ewan Oakes |
| 10 | no | 13638 | Levi Crandale |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 2 | 14901 | Tamsin Severin |

Diagnosis: All relevant pages are already in the returned top 10.

## Good (>=0.5) Queries - 5

### Q4 (q_public_4)

Which franchise player averaged 24 points in the last two games of a title series?

NDCG@10 = `0.7172` | relevant pages = `3` | top-10 found = `3/3` | found in rerank pool = `3/3`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | yes | 2614 | Helena Hawthorne |
| 2 | no | 22668 | Tallmere Memorial Arena |
| 3 | no | 28865 | Port Helix Navigators |
| 4 | yes | 3905 | Juno Bellweather |
| 5 | no | 19684 | Noam Gallant |
| 6 | yes | 22890 | Felix Oakes |
| 7 | no | 16899 | Kareem Abdul-Jabbar |
| 8 | no | 49556 | Stellan Yarrow |
| 9 | no | 31400 | 1999 Los Angeles Lkers season |
| 10 | no | 36753 | Magic Johnson |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 1 | 2614 | Helena Hawthorne |
| rank 4 | 3905 | Juno Bellweather |
| rank 6 | 22890 | Felix Oakes |

Diagnosis: All relevant pages are already in the returned top 10.

### Q11 (q_public_11)

Which city hosts light commuter rail and a small regional airport on a fjord-lined coast?

NDCG@10 = `0.5000` | relevant pages = `1` | top-10 found = `1/1` | found in rerank pool = `1/1`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 54228 | Quayville |
| 2 | no | 4646 | Geography of Quayville |
| 3 | no | 6286 | Commuter rail |
| 4 | yes | 13249 | Brunevale |
| 5 | no | 31606 | Economy of Quayville |
| 6 | no | 38226 | Transportation in Boston |
| 7 | no | 56494 | Bergen |
| 8 | no | 40903 | Economy of Westmere |
| 9 | no | 11105 | Geography of Mossenden |
| 10 | no | 38426 | Kestrel Bay |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 4 | 13249 | Brunevale |

Diagnosis: All relevant pages are already in the returned top 10.

### Q21 (q_public_21)

Which population center combines cold-water fisheries exports with sister-city training exchanges?

NDCG@10 = `0.6934` | relevant pages = `2` | top-10 found = `2/2` | found in rerank pool = `2/2`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 24372 | Dorsaly |
| 2 | yes | 37855 | Economy of Glimmerford |
| 3 | no | 11105 | Geography of Mossenden |
| 4 | no | 40903 | Economy of Westmere |
| 5 | no | 55142 | Economy of Dorsaly |
| 6 | yes | 37226 | Glimmerford |
| 7 | no | 44025 | Economy of Mossenden |
| 8 | no | 19661 | Harrowgate |
| 9 | no | 41307 | Westmere |
| 10 | no | 50630 | Mossenden |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 2 | 37855 | Economy of Glimmerford |
| rank 6 | 37226 | Glimmerford |

Diagnosis: All relevant pages are already in the returned top 10.

### Q25 (q_public_25)

How did negotiations, a signed treaty, and post-war demobilization reports fit together in 1836?

NDCG@10 = `0.5701` | relevant pages = `3` | top-10 found = `2/3` | found in rerank pool = `3/3`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | yes | 18097 | Negotiations for the Rivermark Compromise |
| 2 | no | 9172 | Treaty of Brunevale |
| 3 | no | 4815 | Negotiations for the Northaven Charter |
| 4 | yes | 26610 | Confederation of Frosthaven |
| 5 | no | 3393 | Negotiations for the Rivermark Compromise |
| 6 | no | 51609 | Negotiations for the Congress of Mossenden |
| 7 | no | 13762 | Negotiations for the Northaven Charter |
| 8 | no | 3594 | Rivermark Compromise |
| 9 | no | 30030 | Treaty of Versailles |
| 10 | no | 36992 | Northaven Charter |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 1 | 18097 | Negotiations for the Rivermark Compromise |
| rank 4 | 26610 | Confederation of Frosthaven |
| rank 38 (TOO LOW) | 7997 | Rivermark Compromise |

Diagnosis: Ranking gap: relevant pages are present but currently too low.

### Q26 (q_public_26)

Which population center combines shipbuilding exports with sister-city training exchanges?

NDCG@10 = `0.5701` | relevant pages = `3` | top-10 found = `2/3` | found in rerank pool = `3/3`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 936 | Economy of Pinebarrow |
| 2 | yes | 19401 | Economy of Juniper Reach |
| 3 | no | 21801 | Pinebarrow |
| 4 | yes | 36861 | Juniper Reach |
| 5 | no | 29447 | Geography of Pinebarrow |
| 6 | no | 54228 | Quayville |
| 7 | no | 24372 | Dorsaly |
| 8 | no | 26708 | Quayville |
| 9 | no | 4646 | Geography of Quayville |
| 10 | no | 17015 | Kaohsiung |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 2 | 19401 | Economy of Juniper Reach |
| rank 4 | 36861 | Juniper Reach |
| rank 19 (CLOSE) | 36021 | Geography of Juniper Reach |

Diagnosis: Reranking opportunity: at least one relevant page is close at rank 11-30.

## Medium (>=0.3) Queries - 5

### Q3 (q_public_3)

Which physicist's team improved stability over thermal imaging pipelines two years before publication?

NDCG@10 = `0.4945` | relevant pages = `3` | top-10 found = `2/3` | found in rerank pool = `3/3`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | yes | 5508 | Noam Fenridge |
| 2 | no | 44664 | Ironquay Laboratory Consortium |
| 3 | no | 45724 | Juno Thorne |
| 4 | no | 44518 | Mira Fenridge |
| 5 | no | 8722 | Petra Zeller |
| 6 | no | 45160 | Harrowgate Institute of Applied Physics |
| 7 | no | 22006 | Ivan Crandale |
| 8 | no | 50794 | Lior Northcott |
| 9 | no | 2728 | Dorsaly Field Research Campus |
| 10 | yes | 42955 | Quinn Jurvan |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 1 | 5508 | Noam Fenridge |
| rank 10 | 42955 | Quinn Jurvan |
| rank 25 (CLOSE) | 26566 | Petra Noxley |

Diagnosis: Reranking opportunity: at least one relevant page is close at rank 11-30.

### Q13 (q_public_13)

Which executive introduced cooperative profit-sharing at a maritime logistics firm?

NDCG@10 = `0.3155` | relevant pages = `1` | top-10 found = `1/1` | found in rerank pool = `1/1`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 4065 | Valecross Maritime Group |
| 2 | no | 53419 | History of North Arc Systems |
| 3 | no | 35560 | North Arc Systems |
| 4 | no | 2046 | Valecross Maritime Group |
| 5 | no | 34690 | Stoneford Precision Works |
| 6 | no | 10249 | History of Rivermark Controls AG |
| 7 | no | 40201 | Rivermark Controls AG |
| 8 | no | 5444 | North Arc Systems |
| 9 | yes | 5045 | Uplands Turbine Partners |
| 10 | no | 31602 | Alaric Rourke |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 9 | 5045 | Uplands Turbine Partners |

Diagnosis: All relevant pages are already in the returned top 10.

### Q14 (q_public_14)

Where was the championship clinched in a Memorial Arena game?

NDCG@10 = `0.3869` | relevant pages = `1` | top-10 found = `1/1` | found in rerank pool = `1/1`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 427 | Glimmerford Memorial Arena |
| 2 | no | 10052 | Cindermark Storks |
| 3 | no | 3905 | Juno Bellweather |
| 4 | no | 22508 | Glimmerford Memorial Arena |
| 5 | no | 32656 | Cillian Quillen |
| 6 | yes | 27627 | Cleo Kestrel |
| 7 | no | 36551 | 1905 Lunaris Comets season |
| 8 | no | 44000 | 1826 Harrowgate Cyclones season |
| 9 | no | 47196 | Northaven Memorial Arena |
| 10 | no | 44461 | Rhea Mercer |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 6 | 27627 | Cleo Kestrel |

Diagnosis: All relevant pages are already in the returned top 10.

### Q16 (q_public_16)

Who served as negotiator for a republic that reopened overland routes after 1995?

NDCG@10 = `0.3333` | relevant pages = `1` | top-10 found = `1/1` | found in rerank pool = `1/1`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 18097 | Negotiations for the Rivermark Compromise |
| 2 | no | 36992 | Northaven Charter |
| 3 | no | 26610 | Confederation of Frosthaven |
| 4 | no | 49441 | Free Province of Elmswick |
| 5 | no | 13762 | Negotiations for the Northaven Charter |
| 6 | no | 7393 | Northaven Charter |
| 7 | no | 18759 | Dorsaly Concord |
| 8 | yes | 3310 | Holloway Settlement |
| 9 | no | 11196 | Elmswick |
| 10 | no | 52154 | Holloway Settlement |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 8 | 3310 | Holloway Settlement |

Diagnosis: All relevant pages are already in the returned top 10.

### Q23 (q_public_23)

What links humidity-controlled experiments, bridge monitoring applications, and a patent pool?

NDCG@10 = `0.3890` | relevant pages = `12` | top-10 found = `4/12` | found in rerank pool = `12/12`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | yes | 13638 | Levi Crandale |
| 2 | no | 42955 | Quinn Jurvan |
| 3 | no | 14267 | Dorsaly Field Research Campus publications |
| 4 | no | 149 | Quantized Lattice Annealing |
| 5 | no | 26566 | Petra Noxley |
| 6 | yes | 6098 | Juno Langford |
| 7 | yes | 34028 | Mossenden Technical Institute |
| 8 | no | 2728 | Dorsaly Field Research Campus |
| 9 | no | 45724 | Juno Thorne |
| 10 | yes | 18779 | Mossenden Technical Institute |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 1 | 13638 | Levi Crandale |
| rank 6 | 6098 | Juno Langford |
| rank 7 | 34028 | Mossenden Technical Institute |
| rank 10 | 18779 | Mossenden Technical Institute |
| rank 11 (CLOSE) | 37066 | Rhea Jardine |
| rank 19 (CLOSE) | 7026 | Coherent Beam Deflectometry |
| rank 22 (CLOSE) | 2522 | Mossenden Technical Institute |
| rank 26 (CLOSE) | 31510 | Mossenden Technical Institute publications |
| rank 27 (CLOSE) | 36881 | Resonant Cavity Interferometry |
| rank 30 (CLOSE) | 8722 | Petra Zeller |
| rank 51 (TOO LOW) | 37542 | Ironquay Laboratory Consortium |
| rank 179 (TOO LOW) | 33315 | Orbital Phase Radiometry |

Diagnosis: Reranking opportunity: at least one relevant page is close at rank 11-30.

## Low (>0, <0.3) Queries - 7

### Q6 (q_public_6)

Where did urban planners redesign a riverfront for festivals in 1972?

NDCG@10 = `0.1667` | relevant pages = `2` | top-10 found = `1/2` | found in rerank pool = `1/2`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 4740 | Transportation in Yewton |
| 2 | no | 37855 | Economy of Glimmerford |
| 3 | no | 44582 | Yewton |
| 4 | no | 48603 | Economy of Yewton |
| 5 | no | 54228 | Quayville |
| 6 | no | 19661 | Harrowgate |
| 7 | no | 37226 | Glimmerford |
| 8 | yes | 5251 | Tallmere |
| 9 | no | 24372 | Dorsaly |
| 10 | no | 4646 | Geography of Quayville |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 8 | 5251 | Tallmere |
| NOT FOUND in rerank pool | 50923 | Westmere |

Diagnosis: Recall gap: at least one relevant page is missing from the inspected rerank pool.

### Q7 (q_public_7)

When did automated assembly lines modernize a factory decades after the company was founded?

NDCG@10 = `0.1900` | relevant pages = `3` | top-10 found = `1/3` | found in rerank pool = `3/3`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 1146 | Assembly line |
| 2 | no | 10249 | History of Rivermark Controls AG |
| 3 | no | 4114 | Alaric Upton |
| 4 | yes | 56456 | Rivermark Controls AG |
| 5 | no | 15367 | Niko Kingsley |
| 6 | no | 7041 | Rivermark Controls AG international expansion |
| 7 | no | 40201 | Rivermark Controls AG |
| 8 | no | 24786 | History of Uplands Turbine Partners |
| 9 | no | 53419 | History of North Arc Systems |
| 10 | no | 31602 | Alaric Rourke |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 4 | 56456 | Rivermark Controls AG |
| rank 16 (CLOSE) | 33775 | Quartz Meridian Holdings |
| rank 21 (CLOSE) | 34690 | Stoneford Precision Works |

Diagnosis: Reranking opportunity: at least one relevant page is close at rank 11-30.

### Q17 (q_public_17)

What links a captain's finals performance, his club's rebuild, and a named home arena?

NDCG@10 = `0.0634` | relevant pages = `10` | top-10 found = `1/10` | found in rerank pool = `9/10`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 21584 | 1980 Port Helix Navigators season |
| 2 | no | 8261 | Harrowgate Memorial Arena |
| 3 | no | 34549 | Alaric Drummond |
| 4 | no | 28542 | Juniper Reach Memorial Arena |
| 5 | no | 427 | Glimmerford Memorial Arena |
| 6 | no | 54465 | Hugo Gallant |
| 7 | no | 54907 | Kestrel Bay Admirals |
| 8 | yes | 22668 | Tallmere Memorial Arena |
| 9 | no | 17309 | 2005 Kestrel Bay Admirals season |
| 10 | no | 45655 | Port Helix Navigators |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 8 | 22668 | Tallmere Memorial Arena |
| rank 20 (CLOSE) | 47196 | Northaven Memorial Arena |
| rank 22 (CLOSE) | 10052 | Cindermark Storks |
| rank 25 (CLOSE) | 36551 | 1905 Lunaris Comets season |
| rank 28 (CLOSE) | 19684 | Noam Gallant |
| rank 30 (CLOSE) | 32656 | Cillian Quillen |
| rank 37 (TOO LOW) | 28865 | Port Helix Navigators |
| rank 41 (TOO LOW) | 29682 | 1877 Port Helix Navigators season |
| rank 199 (TOO LOW) | 51300 | Greta Aldridge |
| NOT FOUND in rerank pool | 26564 | 1921 Cindermark Storks season |

Diagnosis: Reranking opportunity: at least one relevant page is close at rank 11-30.

### Q18 (q_public_18)

Which diplomatic settlement involved neutral observers and a joint commission chair?

NDCG@10 = `0.2487` | relevant pages = `9` | top-10 found = `3/9` | found in rerank pool = `8/9`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 3310 | Holloway Settlement |
| 2 | no | 9172 | Treaty of Brunevale |
| 3 | no | 3267 | Treaty of Brunevale |
| 4 | yes | 49441 | Free Province of Elmswick |
| 5 | yes | 7393 | Northaven Charter |
| 6 | no | 52154 | Holloway Settlement |
| 7 | no | 18097 | Negotiations for the Rivermark Compromise |
| 8 | no | 37638 | Ironquay Armistice |
| 9 | no | 26610 | Confederation of Frosthaven |
| 10 | yes | 42062 | Meridian Accord of Ironquay |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 4 | 49441 | Free Province of Elmswick |
| rank 5 | 7393 | Northaven Charter |
| rank 10 | 42062 | Meridian Accord of Ironquay |
| rank 13 (CLOSE) | 13762 | Negotiations for the Northaven Charter |
| rank 17 (CLOSE) | 18759 | Dorsaly Concord |
| rank 25 (CLOSE) | 15413 | Negotiations for the Dorsaly Concord |
| rank 34 (TOO LOW) | 11549 | Negotiations for the Meridian Accord of Ironquay |
| rank 52 (TOO LOW) | 23248 | Commonwealth of Tallmere |
| NOT FOUND in rerank pool | 11065 | Elmswick |

Diagnosis: Reranking opportunity: at least one relevant page is close at rank 11-30.

### Q19 (q_public_19)

What can be learned about a city's economy, geography, and transport network together?

NDCG@10 = `0.0608` | relevant pages = `9` | top-10 found = `1/9` | found in rerank pool = `7/9`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 51114 | Transport economics |
| 2 | no | 21161 | Transport in the Netherlands |
| 3 | no | 23242 | Transport in China |
| 4 | no | 23022 | Transport in Poland |
| 5 | no | 27275 | Transport in Saudi Arabia |
| 6 | no | 31729 | Transport in the United Kingdom |
| 7 | no | 12115 | Transport in Greece |
| 8 | no | 11932 | Transport in Germany |
| 9 | no | 25642 | Transport in Romania |
| 10 | yes | 15135 | Geography of Yewton |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 10 | 15135 | Geography of Yewton |
| rank 13 (CLOSE) | 11105 | Geography of Mossenden |
| rank 25 (CLOSE) | 44582 | Yewton |
| rank 27 (CLOSE) | 40903 | Economy of Westmere |
| rank 33 (TOO LOW) | 4740 | Transportation in Yewton |
| rank 36 (TOO LOW) | 48603 | Economy of Yewton |
| rank 183 (TOO LOW) | 44025 | Economy of Mossenden |
| NOT FOUND in rerank pool | 41307 | Westmere |
| NOT FOUND in rerank pool | 50630 | Mossenden |

Diagnosis: Reranking opportunity: at least one relevant page is close at rank 11-30.

### Q20 (q_public_20)

What links profit-sharing labor policy, alloy research partnerships, and spin-off software products?

NDCG@10 = `0.1771` | relevant pages = `5` | top-10 found = `1/5` | found in rerank pool = `3/5`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 31602 | Alaric Rourke |
| 2 | no | 24786 | History of Uplands Turbine Partners |
| 3 | yes | 37655 | Copper Fenridge Industries |
| 4 | no | 53419 | History of North Arc Systems |
| 5 | no | 35948 | Uplands Turbine Partners |
| 6 | no | 33775 | Quartz Meridian Holdings |
| 7 | no | 10249 | History of Rivermark Controls AG |
| 8 | no | 45012 | Kael Dunmar |
| 9 | no | 56456 | Rivermark Controls AG |
| 10 | no | 34690 | Stoneford Precision Works |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 3 | 37655 | Copper Fenridge Industries |
| rank 14 (CLOSE) | 3511 | Helix Drummond Ltd |
| rank 128 (TOO LOW) | 36503 | History of Helix Drummond Ltd |
| NOT FOUND in rerank pool | 3327 | Dane Severin |
| NOT FOUND in rerank pool | 3481 | History of Copper Fenridge Industries |

Diagnosis: Reranking opportunity: at least one relevant page is close at rank 11-30.

### Q24 (q_public_24)

Which firm expanded from harbor crane components to international service contracts?

NDCG@10 = `0.1771` | relevant pages = `5` | top-10 found = `1/5` | found in rerank pool = `5/5`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 34448 | History of Helix Drummond Ltd |
| 2 | no | 45012 | Kael Dunmar |
| 3 | yes | 53419 | History of North Arc Systems |
| 4 | no | 17930 | Vera Noxley |
| 5 | no | 4114 | Alaric Upton |
| 6 | no | 46162 | History of Helix Drummond Ltd |
| 7 | no | 34690 | Stoneford Precision Works |
| 8 | no | 10249 | History of Rivermark Controls AG |
| 9 | no | 24786 | History of Uplands Turbine Partners |
| 10 | no | 56456 | Rivermark Controls AG |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 3 | 53419 | History of North Arc Systems |
| rank 16 (CLOSE) | 31602 | Alaric Rourke |
| rank 17 (CLOSE) | 40201 | Rivermark Controls AG |
| rank 21 (CLOSE) | 35560 | North Arc Systems |
| rank 27 (CLOSE) | 12699 | History of Rivermark Controls AG |

Diagnosis: Reranking opportunity: at least one relevant page is close at rank 11-30.

## Zero (0.0) Queries - 5

### Q5 (q_public_5)

Who negotiated overseas distribution deals during a company's international expansion?

NDCG@10 = `0.0000` | relevant pages = `1` | top-10 found = `0/1` | found in rerank pool = `1/1`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 56829 | Willa Morven |
| 2 | no | 5444 | North Arc Systems |
| 3 | no | 7041 | Rivermark Controls AG international expansion |
| 4 | no | 31602 | Alaric Rourke |
| 5 | no | 53419 | History of North Arc Systems |
| 6 | no | 4114 | Alaric Upton |
| 7 | no | 45012 | Kael Dunmar |
| 8 | no | 34448 | History of Helix Drummond Ltd |
| 9 | no | 46162 | History of Helix Drummond Ltd |
| 10 | no | 24786 | History of Uplands Turbine Partners |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 16 (CLOSE) | 4065 | Valecross Maritime Group |

Diagnosis: Reranking opportunity: at least one relevant page is close at rank 11-30.

### Q10 (q_public_10)

Who founded a youth basketball foundation after retiring to a hometown arena city?

NDCG@10 = `0.0000` | relevant pages = `2` | top-10 found = `0/2` | found in rerank pool = `2/2`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 14443 | Hoosier Hysteria |
| 2 | no | 10052 | Cindermark Storks |
| 3 | no | 8261 | Harrowgate Memorial Arena |
| 4 | no | 22093 | National Basketball Association |
| 5 | no | 22668 | Tallmere Memorial Arena |
| 6 | no | 55029 | Larry Bird |
| 7 | no | 27627 | Cleo Kestrel |
| 8 | no | 3305 | Los Angeles Lkers |
| 9 | no | 47196 | Northaven Memorial Arena |
| 10 | no | 2614 | Helena Hawthorne |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 70 (TOO LOW) | 32322 | Ingrid Hawthorne |
| rank 217 (TOO LOW) | 26879 | Helena Whitcomb |

Diagnosis: Ranking gap: relevant pages are present but currently too low.

### Q22 (q_public_22)

How do a lead researcher's career, an institute's field trials, and graduate teaching of a method connect?

NDCG@10 = `0.0000` | relevant pages = `6` | top-10 found = `0/6` | found in rerank pool = `3/6`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 25524 | Research |
| 2 | no | 26833 | Scientific method |
| 3 | no | 19552 | Media studies |
| 4 | no | 1449 | Alan Kay |
| 5 | no | 49892 | Socratic method |
| 6 | no | 9252 | Education |
| 7 | no | 27802 | Seymour Papert |
| 8 | no | 10332 | Educational psychology |
| 9 | no | 15014 | Instructional theory |
| 10 | no | 32127 | University of Chicago |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 169 (TOO LOW) | 2728 | Dorsaly Field Research Campus |
| rank 196 (TOO LOW) | 25172 | Tim Corvin |
| rank 205 (TOO LOW) | 43973 | Petra Severin |
| NOT FOUND in rerank pool | 27139 | Quantized Lattice Annealing |
| NOT FOUND in rerank pool | 33570 | Northaven Center for Instrumentation |
| NOT FOUND in rerank pool | 48438 | Northaven Center for Instrumentation publications |

Diagnosis: Recall gap: at least one relevant page is missing from the inspected rerank pool.

### Q27 (q_public_27)

How do a CEO's agreements, a company's research division, and overseas revenue growth connect?

NDCG@10 = `0.0000` | relevant pages = `7` | top-10 found = `0/7` | found in rerank pool = `4/7`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 52234 | Chief executive officer |
| 2 | no | 56829 | Willa Morven |
| 3 | no | 5681 | Corporate title |
| 4 | no | 37398 | The Walt Disney Company |
| 5 | no | 38798 | Big Four accounting firms |
| 6 | no | 22297 | Rivermark Controls AG |
| 7 | no | 29656 | Steve Ballmer |
| 8 | no | 12102 | General Motors |
| 9 | no | 17930 | Vera Noxley |
| 10 | no | 52108 | General Dynamics |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 13 (CLOSE) | 34448 | History of Helix Drummond Ltd |
| rank 18 (CLOSE) | 4114 | Alaric Upton |
| rank 23 (CLOSE) | 45012 | Kael Dunmar |
| rank 126 (TOO LOW) | 5731 | Helix Drummond Ltd |
| NOT FOUND in rerank pool | 7041 | Rivermark Controls AG international expansion |
| NOT FOUND in rerank pool | 17966 | Rivermark Controls AG |
| NOT FOUND in rerank pool | 39400 | History of Rivermark Controls AG |

Diagnosis: Reranking opportunity: at least one relevant page is close at rank 11-30.

### Q28 (q_public_28)

Who was the on-court leader during a championship year that also saw a commemorative banner raised?

NDCG@10 = `0.0000` | relevant pages = `4` | top-10 found = `0/4` | found in rerank pool = `4/4`

Returned top 10:

| Rank | Relevant? | Page ID | Title |
| ---: | :---: | ---: | --- |
| 1 | no | 27627 | Cleo Kestrel |
| 2 | no | 36551 | 1905 Lunaris Comets season |
| 3 | no | 8261 | Harrowgate Memorial Arena |
| 4 | no | 32656 | Cillian Quillen |
| 5 | no | 3905 | Juno Bellweather |
| 6 | no | 9112 | Tim Jordan |
| 7 | no | 10052 | Cindermark Storks |
| 8 | no | 3305 | Los Angeles Lkers |
| 9 | no | 44825 | 1862 Mossenden Waves season |
| 10 | no | 34549 | Alaric Drummond |

Relevant-page rank diagnostics:

| Status | Page ID | Title |
| --- | ---: | --- |
| rank 26 (CLOSE) | 45655 | Port Helix Navigators |
| rank 45 (TOO LOW) | 28542 | Juniper Reach Memorial Arena |
| rank 67 (TOO LOW) | 54465 | Hugo Gallant |
| rank 117 (TOO LOW) | 21584 | 1980 Port Helix Navigators season |

Diagnosis: Reranking opportunity: at least one relevant page is close at rank 11-30.
