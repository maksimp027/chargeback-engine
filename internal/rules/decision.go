package rules

import (
	riskv1 "github.com/maksimp027/chargeback_predict/api/proto/v1"
	"github.com/maksimp027/chargeback_predict/internal/features"
)

type RuleEngine struct {
	DeclineThreshold   float32
	Review3DSThreshold float32
}

func NewRuleEngine(declineThreshold, review3DSThreshold float32) *RuleEngine {
	return &RuleEngine{
		DeclineThreshold:   declineThreshold,
		Review3DSThreshold: review3DSThreshold,
	}
}

func DefaultRuleEngine() *RuleEngine {
	return NewRuleEngine(0.70, 0.30)
}

func (r *RuleEngine) Evaluate(req *riskv1.RiskEvaluationRequest, score float32, vf features.VelocityFeatures) (riskv1.Decision, []string) {
	var reasons []string

	if vf.CardTxCnt5m >= 3 {
		reasons = append(reasons, "HIGH_VELOCITY_5M")
	}
	if vf.CardTxCnt60m >= 10 {
		reasons = append(reasons, "HIGH_VELOCITY_60M")
	}
	if !req.CvvMatch {
		reasons = append(reasons, "CVV_MISMATCH")
	}
	if !req.Is_3DsPassed {
		reasons = append(reasons, "NO_3DS_VERIFICATION")
	}
	if score >= 0.50 {
		reasons = append(reasons, "HIGH_ML_SCORE")
	}

	var decision riskv1.Decision
	switch {
	case score >= r.DeclineThreshold:
		decision = riskv1.Decision_DECLINE
	case score >= r.Review3DSThreshold || (!req.CvvMatch && !req.Is_3DsPassed):
		decision = riskv1.Decision_REVIEW_3DS
	default:
		decision = riskv1.Decision_APPROVE
	}

	return decision, reasons
}
