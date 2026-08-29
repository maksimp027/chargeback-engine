package features

import (
	"context"
	"time"

	riskv1 "github.com/maksimp027/chargeback_predict/api/proto/v1"
)

type Enricher struct {
	builder        *FeatureBuilder
	velocityClient *VelocityClient
}

func NewEnricher(builder *FeatureBuilder, velocityClient *VelocityClient) *Enricher {
	return &Enricher{
		builder:        builder,
		velocityClient: velocityClient,
	}
}

func (e *Enricher) Enrich(ctx context.Context, req *riskv1.RiskEvaluationRequest, now time.Time) ([]float32, VelocityFeatures, error) {
	var vf VelocityFeatures
	var err error

	if e.velocityClient != nil && req.CardFingerprint != "" {
		vf, err = e.velocityClient.GetVelocityFeatures(ctx, req.CardFingerprint, req.AmountCents, req.TransactionId, now.Unix())
		if err != nil {
			vf = VelocityFeatures{}
		}
	}

	feats := e.builder.Build(req, vf, now)
	return feats, vf, nil
}
