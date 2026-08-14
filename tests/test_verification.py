from bgremover.config import VerificationConfig
from bgremover.verification import HumanVerification, should_run_sam


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
