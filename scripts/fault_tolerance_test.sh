#!/bin/bash
# ==============================================================================
# ESM-2 Multi-GPU Inference Service - Fault Tolerance Tests
# ==============================================================================
# Tests API resilience against invalid inputs and edge cases.
# These are NOT unit tests - see tests/ directory for pytest unit tests.
#
# Usage: ./scripts/fault_tolerance_test.sh [BASE_URL]
# Default: http://localhost:8000
# ==============================================================================

BASE_URL="${1:-http://localhost:8000}"

echo "=============================================="
echo "ESM-2 Stress Tests"
echo "Target: $BASE_URL"
echo "=============================================="

# ------------------------------------------------------------------------------
# Test 1: Exceed max batch size (should return 422)
# ------------------------------------------------------------------------------
echo ""
echo "🧪 Test 1: Exceed max batch size (65 sequences, limit is 64)"
echo "Expected: 422 Validation Error"
echo "----------------------------------------------"

curl -s -X POST "$BASE_URL/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"sequences": ["MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK","MKTVRQERLK"]}' | python3 -m json.tool

# ------------------------------------------------------------------------------
# Test 2: Invalid amino acid characters (should return 422)
# ------------------------------------------------------------------------------
echo ""
echo "🧪 Test 2: Invalid amino acid characters"
echo "Expected: 422 Validation Error"
echo "----------------------------------------------"

curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{"sequence": "INVALID123!@#$%"}' | python3 -m json.tool

# ------------------------------------------------------------------------------
# Test 3: Empty request (should return 422)
# ------------------------------------------------------------------------------
echo ""
echo "🧪 Test 3: Empty request body"
echo "Expected: 422 Validation Error"
echo "----------------------------------------------"

curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{}' | python3 -m json.tool

# ------------------------------------------------------------------------------
# Test 4: Empty batch (should return 422)
# ------------------------------------------------------------------------------
echo ""
echo "🧪 Test 4: Empty batch array"
echo "Expected: 422 Validation Error"
echo "----------------------------------------------"

curl -s -X POST "$BASE_URL/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"sequences": []}' | python3 -m json.tool

# ------------------------------------------------------------------------------
# Test 5: Sequence exceeding max length (should return 400 or 422)
# ------------------------------------------------------------------------------
echo ""
echo "🧪 Test 5: Sequence exceeding max length (2000 chars, limit is 1024)"
echo "Expected: 422 Validation Error"
echo "----------------------------------------------"

LONG_SEQ=$(python3 -c "print('M' + 'A'*1999)")
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d "{\"sequence\": \"$LONG_SEQ\"}" | python3 -m json.tool

# ------------------------------------------------------------------------------
# Test 6: Large batch with embeddings and attention (memory stress)
# ------------------------------------------------------------------------------
echo ""
echo "🧪 Test 6: Large batch with embeddings + attention (memory stress)"
echo "Expected: 200 OK (may be slow or OOM on limited hardware)"
echo "----------------------------------------------"

curl -s -X POST "$BASE_URL/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "sequences": [
      "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
      "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
      "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
      "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
      "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
      "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
      "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
      "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
    ],
    "include_embeddings": true,
    "include_attention": true
  }' | python3 -m json.tool

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "✅ Stress tests completed!"
echo "=============================================="
echo ""
echo "Tests 1-5 should return validation errors (422)"
echo "Test 6 should succeed on GPU hardware"
echo ""
