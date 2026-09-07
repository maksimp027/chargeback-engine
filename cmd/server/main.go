package main

import (
	"context"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/redis/go-redis/v9"
	"google.golang.org/grpc"

	riskv1 "github.com/maksimp027/chargeback_predict/api/proto/v1"
	"github.com/maksimp027/chargeback_predict/internal/config"
	"github.com/maksimp027/chargeback_predict/internal/engine"
	"github.com/maksimp027/chargeback_predict/internal/features"
	"github.com/maksimp027/chargeback_predict/internal/rules"
	"github.com/maksimp027/chargeback_predict/internal/service"
)

func main() {
	cfg := config.Load()
	log.Printf("Starting Risk Engine service on %s...", cfg.ServerAddr)

	if err := engine.InitRuntime(cfg.ONNXLibPath); err != nil {
		log.Fatalf("Failed to initialize ONNX Runtime: %v", err)
	}
	defer engine.CleanupRuntime()

	onnxEng, err := engine.NewONNXEngine(cfg.ModelPath)
	if err != nil {
		log.Fatalf("Failed to load ONNX model from %s: %v", cfg.ModelPath, err)
	}
	defer onnxEng.Close()
	log.Printf("Loaded ONNX model from %s", cfg.ModelPath)

	featBuilder, err := features.NewFeatureBuilder(cfg.SchemaPath)
	if err != nil {
		log.Fatalf("Failed to load features schema from %s: %v", cfg.SchemaPath, err)
	}
	log.Printf("Loaded features schema from %s", cfg.SchemaPath)

	var velocityClient *features.VelocityClient
	if cfg.RedisAddr != "" {
		rdb := redis.NewClient(&redis.Options{
			Addr: cfg.RedisAddr,
		})
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()

		if err := rdb.Ping(ctx).Err(); err != nil {
			log.Printf("Warning: Redis unavailable at %s: %v (continuing with degraded velocity metrics)", cfg.RedisAddr, err)
		} else {
			log.Printf("Connected to Redis at %s", cfg.RedisAddr)
		}
		velocityClient = features.NewVelocityClient(rdb)
	}

	enricher := features.NewEnricher(featBuilder, velocityClient)
	ruleEngine := rules.DefaultRuleEngine()
	riskSvc := service.NewRiskService(onnxEng, enricher, ruleEngine)

	listener, err := net.Listen("tcp", cfg.ServerAddr)
	if err != nil {
		log.Fatalf("Failed to listen on %s: %v", cfg.ServerAddr, err)
	}

	grpcServer := grpc.NewServer()
	riskv1.RegisterRiskServiceServer(grpcServer, riskSvc)

	stopChan := make(chan os.Signal, 1)
	signal.Notify(stopChan, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Printf("gRPC server listening on %s", cfg.ServerAddr)
		if err := grpcServer.Serve(listener); err != nil {
			log.Fatalf("gRPC server failed: %v", err)
		}
	}()

	<-stopChan
	log.Println("Shutting down gRPC server gracefully...")
	grpcServer.GracefulStop()
	log.Println("Server stopped successfully.")
}
