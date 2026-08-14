import numpy as np
from bgremover.config import QCConfig
from bgremover.qc import analyze_mask, mask_similarity
from bgremover.edge import EdgeMetrics
from bgremover.verification import HumanVerification

def test_center_mask_is_ready():
    a=np.zeros((100,100),np.uint8); a[20:80,25:75]=255
    result=analyze_mask(a,QCConfig())
    assert not result.needs_review and result.components == 1

def test_sustained_side_contact_is_telemetry_only():
    a=np.zeros((100,100),np.uint8); a[20:80,:40]=255
    result=analyze_mask(a,QCConfig())
    assert result.touch_left and result.cropped_source_signal
    assert not result.needs_review

def test_wide_bottom_contact_is_not_automatically_cropped():
    a=np.zeros((100,100),np.uint8); a[25:,20:80]=255
    result=analyze_mask(a,QCConfig())
    assert not result.cropped_source_signal

def test_ssdlite_signal_does_not_directly_change_fast_qc_status():
    a=np.zeros((100,100),np.uint8); a[20:80,20:60]=255
    human=HumanVerification(person_count=2,missing_count=1,coverages=[.5,.02],center_coverages=[.7,.01])
    result=analyze_mask(a,QCConfig(),human=human)
    assert "missing_body_part" not in result.review_reasons

def test_multiple_people_signal_waits_for_semantic_verifier():
    a=np.zeros((100,100),np.uint8); a[10:90,10:90]=255
    human=HumanVerification(person_count=3,uncertain_count=1)
    result=analyze_mask(a,QCConfig(),human=human)
    assert "multiple_people_uncertain" not in result.review_reasons

def test_rgb_correction_magnitude_is_telemetry_not_review():
    a=np.zeros((100,100),np.uint8); a[20:80,20:80]=255
    edge=EdgeMetrics(.2,.5,.3,500)
    assert "edge_halo" not in analyze_mask(a,QCConfig(),edge=edge).review_reasons

def test_similarity():
    a=np.zeros((10,10),np.uint8); b=a.copy()
    assert mask_similarity(a,b)==(0.0,0.0)

def test_review_details_are_exported():
    result=analyze_mask(np.zeros((20,20),np.uint8),QCConfig())
    assert result.as_dict()["review_details"]


def test_large_hole_is_verification_trigger_not_hard_review():
    alpha = np.zeros((100, 100), np.uint8)
    alpha[10:90, 10:90] = 255
    alpha[30:70, 30:70] = 0
    result = analyze_mask(alpha, QCConfig())
    assert "large_internal_holes" in result.verification_triggers
    assert not result.hard_reasons


def test_almost_empty_mask_remains_hard_review():
    alpha = np.zeros((100, 100), np.uint8)
    alpha[45:50, 45:50] = 255
    result = analyze_mask(alpha, QCConfig())
    assert "mask_issue" in result.hard_reasons
