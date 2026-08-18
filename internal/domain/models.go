package domain

import (
	riskv1 "github.com/maksimp027/chargeback_predict/api/proto/v1"
)

type FeaturesSchema struct {
	Features   []string           `json:"features"`
	BinRiskMap map[string]float64 `json:"bin_risk_map"`
}

type RiskEvaluationInput struct {
	Request *riskv1.RiskEvaluationRequest
}

type EvaluationResult struct {
	Decision        riskv1.Decision
	RiskScore       float32
	ModelVersion    string
	ReasonCodes     []string
	ExecutionTimeUs int64
}
