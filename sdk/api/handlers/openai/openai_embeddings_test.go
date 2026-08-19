package openai

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/api/handlers"
)

func TestOpenAIEmbeddingsHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Mock upstream server
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("expected path /embeddings, got %s", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer test-api-key" {
			t.Errorf("unexpected auth header: %s", r.Header.Get("Authorization"))
		}
		if r.Header.Get("X-Custom-Header") != "CustomValue" {
			t.Errorf("expected X-Custom-Header: CustomValue, got %s", r.Header.Get("X-Custom-Header"))
		}

		body, _ := io.ReadAll(r.Body)
		var reqMap map[string]interface{}
		_ = json.Unmarshal(body, &reqMap)
		if reqMap["model"] != "nvidia/nemotron-3-embed-1b" {
			t.Errorf("expected rewritten model nvidia/nemotron-3-embed-1b, got %v", reqMap["model"])
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"object": "list",
			"data": [
				{
					"object": "embedding",
					"embedding": [0.1, 0.2, 0.3],
					"index": 0
				}
			],
			"model": "nvidia/nemotron-3-embed-1b",
			"usage": {
				"prompt_tokens": 5,
				"total_tokens": 5
			}
		}`))
	}))
	defer upstreamServer.Close()

	cfg := &config.Config{
		SDKConfig: config.SDKConfig{
			OpenAICompatibility: []config.OpenAICompatibility{
				{
					Name:    "nvidia",
					BaseURL: upstreamServer.URL,
					Headers: map[string]string{
						"X-Custom-Header": "CustomValue",
					},
					APIKeyEntries: []config.OpenAICompatibilityAPIKey{
						{APIKey: "test-api-key"},
					},
					Models: []config.OpenAICompatibilityModel{
						{
							Name:  "nvidia/nemotron-3-embed-1b",
							Alias: "embed-nemotron",
						},
					},
				},
			},
		},
	}

	baseHandler := handlers.NewBaseAPIHandlers(&cfg.SDKConfig, nil)
	handler := NewOpenAIAPIHandler(baseHandler)

	router := gin.New()
	router.POST("/v1/embeddings", handler.Embeddings)

	// Test 1: Successful request using model alias
	reqBody := []byte(`{"model":"embed-nemotron","input":"test embedding"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d: %s", w.Code, w.Body.String())
	}

	var respMap map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &respMap); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	if respMap["object"] != "list" {
		t.Errorf("expected object list, got %v", respMap["object"])
	}

	// Test 2: Missing model
	reqBodyMissing := []byte(`{"input":"test"}`)
	req2 := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(reqBodyMissing))
	req2.Header.Set("Content-Type", "application/json")
	w2 := httptest.NewRecorder()
	router.ServeHTTP(w2, req2)

	if w2.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400 for missing model, got %d", w2.Code)
	}

	// Test 3: Unknown model
	reqBodyUnknown := []byte(`{"model":"unknown-model","input":"test"}`)
	req3 := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(reqBodyUnknown))
	req3.Header.Set("Content-Type", "application/json")
	w3 := httptest.NewRecorder()
	router.ServeHTTP(w3, req3)

	if w3.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400 for unknown model, got %d", w3.Code)
	}
}
