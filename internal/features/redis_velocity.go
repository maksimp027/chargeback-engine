package features

import (
	"context"
	"fmt"

	"github.com/redis/go-redis/v9"
)

type VelocityClient struct {
	rdb *redis.Client
}

func NewVelocityClient(rdb *redis.Client) *VelocityClient {
	return &VelocityClient{rdb: rdb}
}

type VelocityFeatures struct {
	CardTxCnt5m     float32
	CardTxCnt15m    float32
	CardTxCnt60m    float32
	CardSumCents60m float32
}

var velocityLuaScript = redis.NewScript(`
local key = KEYS[1]
local now = tonumber(ARGV[1])
local amount = tonumber(ARGV[2])
local tx_id = ARGV[3]

redis.call('ZREMRANGEBYSCORE', key, '-inf', '(' .. (now - 3600))

-- Evaluate left-closed sliding windows prior to inserting current tx to prevent target leakage.
local cnt_5m = redis.call('ZCOUNT', key, now - 300, '+inf')
local cnt_15m = redis.call('ZCOUNT', key, now - 900, '+inf')
local cnt_60m = redis.call('ZCOUNT', key, now - 3600, '+inf')

local members = redis.call('ZRANGEBYSCORE', key, now - 3600, '+inf')
local sum_60m = 0
for _, val in ipairs(members) do
    local colon_idx = string.find(val, ":")
    if colon_idx then
        local val_amount = tonumber(string.sub(val, 1, colon_idx - 1))
        if val_amount then
            sum_60m = sum_60m + val_amount
        end
    end
end

local member = amount .. ':' .. tx_id
redis.call('ZADD', key, now, member)
redis.call('EXPIRE', key, 3600)

return {cnt_5m, cnt_15m, cnt_60m, sum_60m}
`)

func (vc *VelocityClient) GetVelocityFeatures(ctx context.Context, cardFingerprint string, amountCents int64, txID string, timestamp int64) (VelocityFeatures, error) {
	key := fmt.Sprintf("velocity:card:%s", cardFingerprint)

	res, err := velocityLuaScript.Run(ctx, vc.rdb, []string{key}, timestamp, amountCents, txID).Result()
	if err != nil {
		return VelocityFeatures{}, err
	}

	slice, ok := res.([]interface{})
	if !ok || len(slice) < 4 {
		return VelocityFeatures{}, fmt.Errorf("unexpected lua script response format")
	}

	cnt5m := float32(slice[0].(int64))
	cnt15m := float32(slice[1].(int64))
	cnt60m := float32(slice[2].(int64))
	sum60m := float32(slice[3].(int64))

	return VelocityFeatures{
		CardTxCnt5m:     cnt5m,
		CardTxCnt15m:    cnt15m,
		CardTxCnt60m:    cnt60m,
		CardSumCents60m: sum60m,
	}, nil
}
