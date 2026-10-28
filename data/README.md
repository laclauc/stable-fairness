# Data

| Dataset | Expected file | G = 1 | y = 1 | Source |
|---|---|---|---|---|
| Adult Income | `adult/adult_norm.csv` | male (`gender:is_Male`) | income > 50K (`income:is_>50K`) | UCI Machine Learning Repository (CC BY 4.0) |
| COMPAS | `compas/Processed_Compas.csv` | African-American | recidivism (`is_recid`) | ProPublica |
| ACS Income | downloaded to `acs/` on first use | White (`RAC1P` = 1) | income > 50K | US Census, via folktables |

**Adult.** All columns other than the two above are used as features, as they
are in the file (encoded and normalised). TODO: how `adult_norm.csv` is built.

**COMPAS.** Only African-American and Caucasian defendants are kept. Features:
`age`, `priors_count`, `days_b_screening_arrest`, `duration` (numerical) and
one-hot encodings of `c_charge_degree`, `age_cat`, `sex`.
TODO: how `Processed_Compas.csv` is built from the ProPublica data.
