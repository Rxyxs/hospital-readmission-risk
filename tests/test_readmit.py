import numpy as np
import pandas as pd
import pytest

from readmit.data import icd9_group
from readmit.model import net_benefit


@pytest.mark.parametrize("code,group", [("428", "circulatory"), ("250.83", "diabetes"), ("486", "respiratory"),
                                        ("V58", "other"), (np.nan, "missing"), ("785", "circulatory"),
                                        ("174", "neoplasms"), ("820", "injury")])
def test_icd9_groups(code, group):
    assert icd9_group(code) == group


def test_net_benefit_matches_definition():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([.9, .8, .1, .1, .1, .1, .1, .1, .1, .1])
    # at t=0.5 treat two patients: one true positive, one false positive
    assert net_benefit(y, p, [0.5])[0] == pytest.approx(1 / 10 - 1 / 10 * 0.5 / 0.5)
    # perfect model at any threshold: net benefit = prevalence
    assert net_benefit(y, y.astype(float), [0.2])[0] == pytest.approx(0.2)


def test_prior_encounters_never_count_the_future():
    d = pd.DataFrame({"encounter_id": [3, 1, 2, 4], "patient_nbr": [7, 7, 8, 7]}).sort_values("encounter_id")
    d["prior"] = d.groupby("patient_nbr").cumcount()
    d["total"] = d.groupby("patient_nbr").encounter_id.transform("size")
    first = d[d.encounter_id == 1].iloc[0]
    assert first.prior == 0 and first.total == 3   # the leaky count already knows about visits 3 and 4
