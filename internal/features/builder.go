package features

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"time"

	riskv1 "github.com/maksimp027/chargeback_predict/api/proto/v1"
)

type Schema struct {
	Features   []string           `json:"features"`
	BinRiskMap map[string]float64 `json:"bin_risk_map"`
}

type FeatureBuilder struct {
	schema Schema
}

func NewFeatureBuilder(schemaPath string) (*FeatureBuilder, error) {
	data, err := os.ReadFile(schemaPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read schema file: %w", err)
	}

	var schema Schema
	if err := json.Unmarshal(data, &schema); err != nil {
		return nil, fmt.Errorf("failed to parse schema json: %w", err)
	}

	return &FeatureBuilder{schema: schema}, nil
}

// Build constructs the 19-element feature vector in the exact order defined by features_schema.json.
func (fb *FeatureBuilder) Build(req *riskv1.RiskEvaluationRequest, vf VelocityFeatures, reqTime time.Time) []float32 {
	feats := make([]float32, 19)

	feats[0] = float32(math.Log1p(float64(req.AmountCents) / 100.0))

	if req.AmountCents%1000 == 0 {
		feats[1] = 1.0
	}

	feats[2] = float32(req.AmountCents % 100)

	// Cyclical encoding of UTC hour-of-day to preserve continuity across midnight.
	hour := float64(reqTime.Hour())
	feats[3] = float32(math.Sin(2 * math.Pi * hour / 24.0))
	feats[4] = float32(math.Cos(2 * math.Pi * hour / 24.0))

	// Shift Go's Weekday (Sunday=0..Saturday=6) to match pandas dayofweek (Monday=0..Sunday=6).
	dow := (int(reqTime.Weekday()) + 6) % 7
	feats[5] = float32(dow)

	if dow >= 5 {
		feats[6] = 1.0
	}

	if req.CvvMatch {
		feats[7] = 1.0
	}

	if req.Is_3DsPassed {
		feats[8] = 1.0
	}

	if !req.CvvMatch && !req.Is_3DsPassed {
		feats[9] = 1.0
	}

	if req.Currency == "USD" {
		feats[10] = 1.0
	}

	if req.BillingCountry == "US" {
		feats[11] = 1.0
	}

	var ageDays float64
	if req.UserCreatedAt > 0 {
		// Clamp negative clock skew to zero before log1p to prevent NaN/domain errors.
		ageSec := float64(reqTime.Unix() - req.UserCreatedAt)
		if ageSec < 0 {
			ageSec = 0
		}
		ageDays = ageSec / 86400.0
	}
	feats[12] = float32(ageDays)
	feats[13] = float32(math.Log1p(math.Max(0, ageDays)))

	binStr := fmt.Sprintf("%d", req.CardBin)
	// Default to global prior chargeback rate (0.07) for unseen BINs.
	binRisk := 0.07
	if r, ok := fb.schema.BinRiskMap[binStr]; ok {
		binRisk = r
	}
	feats[14] = float32(binRisk)

	feats[15] = vf.CardTxCnt5m
	feats[16] = vf.CardTxCnt15m
	feats[17] = vf.CardTxCnt60m
	feats[18] = vf.CardSumCents60m

	return feats
}
