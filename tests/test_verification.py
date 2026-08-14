from bgremover.config import VerificationConfig
from bgremover.verification import (
    HumanVerification,
    select_sam_prompt_indices,
    should_run_sam,
)


def test_sam_not_requested_for_clean_result():
    human = HumanVerification(person_count=1, coverages=[0.75], center_coverages=[0.9])
    assert not should_run_sam([], human, VerificationConfig())


def test_sam_requested_for_strong_semantic_signal():
    human = HumanVerification(person_count=1, coverages=[0.1], center_coverages=[0.2])
    assert should_run_sam([], human, VerificationConfig())


def test_multiple_people_with_marginal_coverage_requests_verifier():
    human = HumanVerification(person_count=2, coverages=[0.58, 0.8], center_coverages=[0.8, 0.9])
    assert should_run_sam([], human, VerificationConfig())


def test_zero_person_detections_never_request_sam():
    assert not should_run_sam(["mask_issue"], HumanVerification(), VerificationConfig())


def test_missing_center_signal_requests_sam_even_with_high_box_coverage():
    human = HumanVerification(
        person_count=1, missing_count=1, coverages=[0.8], center_coverages=[0.1]
    )
    assert should_run_sam([], human, VerificationConfig())


def test_small_background_person_is_not_a_sam_prompt():
    human = HumanVerification(
        person_count=2,
        boxes=[(10, 10, 90, 190), (140, 20, 160, 60)],
        scores=[0.99, 0.91],
        coverages=[0.8, 0.0],
        center_coverages=[0.9, 0.0],
    )
    indices, details = select_sam_prompt_indices(human, VerificationConfig())
    assert indices == [0]
    assert details


def test_significant_partially_missing_person_remains_a_sam_prompt():
    human = HumanVerification(
        person_count=2,
        boxes=[(0, 0, 100, 200), (120, 20, 180, 120)],
        scores=[0.99, 0.95],
        coverages=[0.8, 0.10],
        center_coverages=[0.9, 0.02],
    )
    indices, _ = select_sam_prompt_indices(human, VerificationConfig())
    assert indices == [0, 1]


def test_box_spanning_supported_people_is_not_a_sam_prompt():
    human = HumanVerification(
        person_count=3,
        boxes=[(0, 0, 60, 180), (65, 0, 125, 180), (0, 0, 125, 180)],
        scores=[0.99, 0.97, 0.91],
        coverages=[0.8, 0.8, 0.20],
        center_coverages=[0.9, 0.9, 0.10],
    )
    indices, _ = select_sam_prompt_indices(human, VerificationConfig())
    assert indices == [0, 1]
