import numpy as np
from bgremover.config import QCConfig
from bgremover.qc import analyze_mask, mask_similarity
from bgremover.edge import EdgeMetrics
from bgremover.verification import HumanVerification

def test_center_mask_is_ready():
    a=np.zeros((100,100),np.uint8); a[20:80,25:75]=255
    result=analyze_mask(a,QCConfig())
    assert not result.needs_review and result.components == 1

def test_sustained_side_contact_marks_cropped_source():
    a=np.zeros((100,100),np.uint8); a[20:80,:40]=255
    result=analyze_mask(a,QCConfig())
    assert result.touch_left and "cropped_source" in result.review_reasons

def test_wide_bottom_contact_is_not_automatically_cropped():
    a=np.zeros((100,100),np.uint8); a[25:,20:80]=255
    result=analyze_mask(a,QCConfig())
    assert "cropped_source" not in result.review_reasons

def test_missing_detected_person_is_never_ready():
    a=np.zeros((100,100),np.uint8); a[20:80,20:60]=255
    human=HumanVerification(person_count=2,missing_count=1,coverages=[.5,.02],center_coverages=[.7,.01])
    result=analyze_mask(a,QCConfig(),human=human)
    assert "missing_body_part" in result.review_reasons

def test_multiple_people_uncertain_reason():
    a=np.zeros((100,100),np.uint8); a[10:90,10:90]=255
    human=HumanVerification(person_count=3,uncertain_count=1)
    result=analyze_mask(a,QCConfig(),human=human)
    assert "multiple_people_uncertain" in result.review_reasons

def test_strong_edge_contamination_is_review():
    a=np.zeros((100,100),np.uint8); a[20:80,20:80]=255
    edge=EdgeMetrics(.2,.5,.3,500)
    assert "edge_halo" in analyze_mask(a,QCConfig(),edge=edge).review_reasons

def test_similarity():
    a=np.zeros((10,10),np.uint8); b=a.copy()
    assert mask_similarity(a,b)==(0.0,0.0)
