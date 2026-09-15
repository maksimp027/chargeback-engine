package rules

import (
	"testing"

	riskv1 "github.com/maksimp027/chargeback_predict/api/proto/v1"
	"github.com/maksimp027/chargeback_predict/internal/features"
)

func TestRuleEngine_Evaluate(t *testing.T) {
	re := DefaultRuleEngine()

	req := &riskv1.RiskEvaluationRequest{
		TransactionId: "tx-100",
		CvvMatch:      true,
		Is_3DsPassed:  true,
	}

	vf := features.VelocityFeatures{}
	decision, reasons := re.Evaluate(req, 0.15, vf)

	if decision != riskv1.Decision_APPROVE {
		t.Errorf("Expected APPROVE, got %v", decision)
	}
	if len(reasons) != 0 {
		t.Errorf("Expected 0 reasons, got %v", reasons)
	}

	decision, reasons = re.Evaluate(req, 0.75, vf)
	if decision != riskv1.Decision_DECLINE {
		t.Errorf("Expected DECLINE, got %v", decision)
	}
	if len(reasons) == 0 {
		t.Errorf("Expected reason codes, got empty")
	}
}
