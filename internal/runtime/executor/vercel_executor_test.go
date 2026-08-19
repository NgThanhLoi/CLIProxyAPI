package executor

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
)

func TestNewVercelExecutor(t *testing.T) {
	cfg := &config.Config{}
	executor := NewVercelExecutor(cfg)
	if executor.Identifier() != "vercel" {
		t.Errorf("expected vercel, got %s", executor.Identifier())
	}
}

func TestVercelExecutorExecute(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-access-token" {
			t.Errorf("expected Bearer test-access-token, got %s", r.Header.Get("Authorization"))
		}
		if r.Header.Get("ai-language-model-id") != "zai/glm-5.2" {
			t.Errorf("expected zai/glm-5.2, got %s", r.Header.Get("ai-language-model-id"))
		}
		if r.Header.Get("X-Title") != "fx" {
			t.Errorf("expected X-Title fx, got %s", r.Header.Get("X-Title"))
		}

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"text-delta\",\"delta\":\"Xin chao tu GLM 5.2\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"finish-step\",\"finishReason\":\"stop\"}\n\n"))
	}))
	defer server.Close()

	cfg := &config.Config{}
	executor := NewVercelExecutor(cfg)

	auth := &cliproxyauth.Auth{
		Provider: "vercel",
		Metadata: map[string]interface{}{
			"access_token": "test-access-token",
		},
		Attributes: map[string]string{
			"base_url": server.URL,
		},
	}

	payload := []byte(`{"model":"zai/glm-5.2","messages":[{"role":"user","content":"Xin chao"}]}`)
	req := cliproxyexecutor.Request{
		Model:   "zai/glm-5.2",
		Payload: payload,
	}
	opts := cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FormatOpenAI,
	}

	resp, err := executor.Execute(context.Background(), auth, req, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(string(resp.Payload), "Xin chao tu GLM 5.2") {
		t.Errorf("expected response to contain Xin chao tu GLM 5.2, got %s", string(resp.Payload))
	}
}

func TestVercelExecutorExecuteStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"text-delta\",\"delta\":\"chunk1\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"finish\",\"finishReason\":\"stop\"}\n\n"))
	}))
	defer server.Close()

	cfg := &config.Config{}
	executor := NewVercelExecutor(cfg)

	auth := &cliproxyauth.Auth{
		Provider: "vercel",
		Metadata: map[string]interface{}{
			"access_token": "test-access-token",
		},
		Attributes: map[string]string{
			"base_url": server.URL,
		},
	}

	payload := []byte(`{"model":"zai/glm-5.2","messages":[{"role":"user","content":"Hi"}]}`)
	req := cliproxyexecutor.Request{
		Model:   "zai/glm-5.2",
		Payload: payload,
	}
	opts := cliproxyexecutor.Options{
		SourceFormat: sdktranslator.FormatOpenAI,
	}

	streamRes, err := executor.ExecuteStream(context.Background(), auth, req, opts)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var chunks []string
	for chunk := range streamRes.Chunks {
		if len(chunk.Payload) > 0 {
			chunks = append(chunks, string(chunk.Payload))
		}
	}

	full := strings.Join(chunks, "")
	if !strings.Contains(full, "chunk1") {
		t.Errorf("expected stream to contain chunk1, got %s", full)
	}
}
