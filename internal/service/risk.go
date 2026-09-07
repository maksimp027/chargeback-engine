package service

import (
	"context"
	"time"

	riskv1 "github.com/maksimp027/chargeback_predict/api/proto/v1"
	"github.com/maksimp027/chargeback_predict/internal/engine"
	"github.com/maksimp027/chargeback_predict/internal/features"
	"github.com/maksimp027/chargeback_predict/internal/rules"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type RiskService struct {
	riskv1.UnimplementedRiskServiceServer
	onnxEngine *engine.ONNXEngine
	enricher   *features.Enricher
	ruleEngine *rules.RuleEngine
}

func NewRiskService(onnxEngine *engine.ONNXEngine, enricher *features.Enricher, ruleEngine *rules.RuleEngine) *RiskService {
	return &RiskService{
		onnxEngine: onnxEngine,
		enricher:   enricher,
		ruleEngine: ruleEngine,
	}
}

func (s *RiskService) EvaluateTransaction(ctx context.Context, req *riskv1.RiskEvaluationRequest) (*riskv1.RiskEvaluationResponse, error) {
	startTime := time.Now()

	if req.TransactionId == "" {
		return nil, status.Error(codes.InvalidArgument, "transaction_id is required")
	}

	feats, vf, err := s.enricher.Enrich(ctx, req, startTime)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to enrich features: %v", err)
	}

	score, err := s.onnxEngine.Predict(feats)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to predict risk: %v", err)
	}

	decision, reasons := s.ruleEngine.Evaluate(req, score, vf)
	executionTimeUs := time.Since(startTime).Microseconds()

	return &riskv1.RiskEvaluationResponse{
		TransactionId:   req.TransactionId,
		Decision:        decision,
		RiskScore:       score,
		ModelVersion:    "1.0.0",
		ReasonCodes:     reasons,
		ExecutionTimeUs: executionTimeUs,
	}, nil
}
