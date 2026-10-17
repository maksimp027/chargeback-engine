# Chargeback Risk Engine

A pre-authorization transaction risk evaluation service written in Go and backed by an in-process ONNX Runtime inference engine and Redis 7 sliding-window state store. The service exposes a unary gRPC endpoint that enriches incoming payment events with real-time card velocity aggregations, evaluates a 19-feature gradient-boosted decision tree (XGBoost Opset 15), and applies deterministic policy rules to return `APPROVE`, `REVIEW_3DS`, or `DECLINE` decisions. End-to-end evaluation sustains p99 < 8ms latency under concurrent load by binding pre-allocated C heap tensors directly to the ONNX Runtime shared library over CGO.

## Architecture

```text
+--------------------+
|   Payment Gateway  |
+---------+----------+
          |
          | 1. RiskEvaluationRequest (gRPC / Protobuf)
          v
+--------------------+       2. EVALSHA velocityLuaScript        +--------------------+
|    gRPC Handler    +------------------------------------------>+      Redis 7+      |
| (internal/service) |<------------------------------------------+  (ZSET per Card)   |
+---------+----------+       3. 5m/15m/60m counts & 60m sum      +--------------------+
          |
          | 4. Construct 19-float32 vector (cyclical time, log transforms, smoothed BIN risk)
          v
+--------------------+
|  Feature Enricher  |
| (internal/features)|
+---------+----------+
          |
          | 5. In-place copy to pinned C tensor buffer (mutex-serialized)
          v
+--------------------+
|  ONNX CGO Session  |  --> libonnxruntime.so (XGBoost Binary Classifier, Opset 15)
|  (internal/engine) |
+---------+----------+
          |
          | 6. P(chargeback) in [0.0, 1.0]
          v
+--------------------+
| Policy Rule Engine |  --> Score thresholds (0.30 / 0.70) + CVV / 3DS / Velocity checks
|  (internal/rules)  |
+---------+----------+
          |
          | 7. RiskEvaluationResponse (Decision, RiskScore, ReasonCodes, ExecutionTimeUs)
          v
+--------------------+
|   Payment Gateway  |
+--------------------+
```

## Technical Design Decisions

### In-Process CGO Inference vs. External Model Server
Inference runs inside the Go process via `onnxruntime_go` CGO bindings against `libonnxruntime.so` rather than dispatching RPCs to an external Python/Triton model server. For single-row tabular tree ensembles (1x19 `float32` input), network serialization and inter-service hop overhead dominate total execution time. `ONNXEngine` allocates the `[1, 19]` input tensor and `[1, 2]` output tensor once on the C heap during initialization and reuses them across requests under a mutex (`sync.Mutex`), eliminating per-request CGO pointer allocations and keeping pure model evaluation below 0.5ms and end-to-end p99 latency below 8ms.

### Atomic Sliding-Window Aggregations in Redis
Card velocity features (`card_tx_cnt_5m`, `card_tx_cnt_15m`, `card_tx_cnt_60m`, `card_sum_cents_60m`) are maintained in Redis sorted sets (`ZSET`) keyed by `velocity:card:<card_fingerprint>`, where scores represent Unix epoch timestamps in seconds and members encode `<amount_cents>:<transaction_id>` to guarantee uniqueness across identical amounts. A single server-side Lua script executes atomically on each request:
1. Evicts entries older than 3600 seconds via `ZREMRANGEBYSCORE`.
2. Computes 5-minute, 15-minute, and 60-minute window counts (`ZCOUNT`) and sums 60-minute transaction amounts (`ZRANGEBYSCORE`) strictly prior to recording the current transaction, matching the left-closed rolling windows used during offline training and preventing target leakage.
3. Appends the current transaction (`ZADD`) and refreshes key TTL to 3600 seconds (`EXPIRE`).

### Degraded-Mode Fallback Handling
The service prioritizes payment path availability when Redis is unreachable or times out:
- **Startup**: If the initial Redis `PING` fails (3-second timeout), the server logs a degraded-state warning and continues serving traffic.
- **Runtime**: If `EVALSHA` fails during `Enrich`, the enricher suppresses the storage error and substitutes a zero-initialized `VelocityFeatures` struct (`0` counts and `0` rolling sum). Inference proceeds using static transaction attributes, historical smoothed BIN risk priors (`features_schema.json`), and hard verification rules (CVV/3DS).

## Interface Contract

Defined in `api/proto/risk_service.proto`:

```protobuf
syntax = "proto3";

package risk.v1;

option go_package = "github.com/maksimp027/chargeback_predict/api/proto/v1;riskv1";

service RiskService {
  rpc EvaluateTransaction (RiskEvaluationRequest) returns (RiskEvaluationResponse);
}

message RiskEvaluationRequest {
  string transaction_id = 1;
  string merchant_id = 2;
  string account_id = 3;
  int64 amount_cents = 4;
  string currency = 5;
  int32 card_bin = 6;
  string card_fingerprint = 7;
  string client_ip = 8;
  bool cvv_match = 9;
  bool is_3ds_passed = 10;
  string billing_country = 11;
  int64 user_created_at = 12;
}

enum Decision {
  DECISION_UNSPECIFIED = 0;
  APPROVE = 1;
  REVIEW_3DS = 2;
  DECLINE = 3;
}

message RiskEvaluationResponse {
  string transaction_id = 1;
  Decision decision = 2;
  float risk_score = 3;
  string model_version = 4;
  repeated string reason_codes = 5;
  int64 execution_time_us = 6;
}
```

### Decision Policy & Reason Codes
- `DECLINE`: `risk_score >= 0.70`
- `REVIEW_3DS`: `0.30 <= risk_score < 0.70` or (`!cvv_match && !is_3ds_passed`)
- `APPROVE`: `risk_score < 0.30` with at least one of CVV or 3DS verified
- Emitted `reason_codes`: `HIGH_VELOCITY_5M` (>= 3 tx), `HIGH_VELOCITY_60M` (>= 10 tx), `CVV_MISMATCH`, `NO_3DS_VERIFICATION`, `HIGH_ML_SCORE` (>= 0.50).

## Quickstart

### Prerequisites
- Go 1.22+ (CGO enabled, `gcc`/`build-essential`)
- ONNX Runtime C shared library (`libonnxruntime.so` 1.20+)
- Redis 7+
- Python 3.11+ (offline model training and ONNX export only)
- Docker & Docker Compose v2

### Container Deployment

Start the gRPC service, Redis 7 instance, and Prometheus scraper:

```bash
docker compose -f deploy/docker-compose.yml up --build -d
```

Verify container status and inspect logs:

```bash
docker compose -f deploy/docker-compose.yml ps
docker logs chargeback-risk-engine
```

### Local Build & Execution

```bash
export ONNX_LIB_PATH=/usr/lib/libonnxruntime.so
make build
./bin/chargeback-risk-engine
```

Verify the gRPC endpoint using `grpcurl` (or TCP port probe):

```bash
nc -zv localhost 50051
```

## Configuration

All runtime parameters are configured via environment variables:

| Name | Type | Default Value | Purpose |
| :--- | :--- | :--- | :--- |
| `SERVER_ADDR` | `string` | `:50051` | TCP bind address and port for the gRPC server |
| `REDIS_ADDR` | `string` | `localhost:6379` | Redis `host:port` endpoint for sliding-window velocity state |
| `MODEL_PATH` | `string` | `model_artifacts/chargeback_model.onnx` | Filesystem path to the serialized ONNX model graph |
| `SCHEMA_PATH` | `string` | `model_artifacts/features_schema.json` | Filesystem path to feature ordering and smoothed BIN risk map |
| `ONNX_LIB_PATH` | `string` | `/usr/lib/libonnxruntime.so` | Filesystem path to the `libonnxruntime` shared C library |

## Testing & Benchmarks

Run all unit tests (rule engine policy logic and CGO ONNX session execution):

```bash
ONNX_LIB_PATH=/usr/lib/libonnxruntime.so make test
```

Run in-process ONNX inference latency and allocation benchmarks:

```bash
ONNX_LIB_PATH=/usr/lib/libonnxruntime.so go test -bench=. -benchmem ./internal/...
```

Retrain the XGBoost classifier and validate numerical parity (`max_diff < 1e-4`) between native XGBoost and exported ONNX Runtime outputs:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r ml/requirements.txt
python ml/train.py
python ml/export_onnx.py
```
