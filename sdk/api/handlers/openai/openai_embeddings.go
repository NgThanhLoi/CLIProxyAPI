// Package openai — Embeddings endpoint handler (custom, feat/embeddings).
//
// Implements POST /v1/embeddings as a pass-through proxy to the upstream
// provider declared in config.yaml for the requested model. Unlike chat
// completions, this bypasses the translator pipeline (which would forward to
// /v1/chat/completions) and calls the upstream {base-url}/embeddings endpoint
// directly, resolving the API key from the provider's api-key-entries.
//
// Embedding models are declared in config.yaml like any other model:
//
//	- name: "nvidia/nemotron-3-embed-1b"
//	  alias: "embed-nemotron"
//	  display-name: "Nemotron 3 Embed 1B"
package openai

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/api/handlers"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"gopkg.in/yaml.v3"
)

// ---- minimal config view (only what embeddings needs) ----------------------

type embedProviderEntry struct {
	Name         string `yaml:"name"`
	BaseURL      string `yaml:"base-url"`
	APIKeyEntries []struct {
		APIKey string `yaml:"api-key"`
	} `yaml:"api-key-entries"`
	ModelEntries []struct {
		Name  string `yaml:"name"`
		Alias string `yaml:"alias"`
	} `yaml:"models"`
}

type embedConfig struct {
	OpenAICompat []embedProviderEntry `yaml:"openai-compatibility"`
}

// ---- provider resolution ---------------------------------------------------

// resolveEmbedProvider maps a requested model name (alias or upstream name)
// to (baseURL, apiKeys, upstreamModelName) using config.yaml at the mounted
// path. The upstream name is used to rewrite the body's "model" field before
// forwarding (NIM rejects aliases).
func resolveEmbedProvider(modelName string) (string, []string, string, bool) {
	const configPath = "/CLIProxyAPI/config.yaml"
	raw, err := os.ReadFile(configPath)
	if err != nil {
		return "", nil, "", false
	}
	var cfg embedConfig
	if err := yaml.Unmarshal(raw, &cfg); err != nil {
		return "", nil, "", false
	}
	for _, p := range cfg.OpenAICompat {
		upstreamName := ""
		for _, m := range p.ModelEntries {
			if m.Name == modelName || m.Alias == modelName {
				upstreamName = m.Name
				break
			}
		}
		if upstreamName == "" {
			continue
		}
		keys := make([]string, 0, len(p.APIKeyEntries))
		for _, k := range p.APIKeyEntries {
			if k.APIKey != "" {
				keys = append(keys, k.APIKey)
			}
		}
		if len(keys) == 0 {
			return "", nil, "", false
		}
		return p.BaseURL, keys, upstreamName, true
	}
	return "", nil, "", false
}

var embedRRMu sync.Mutex
var embedRRIdx = 0

// pickEmbedKey round-robins across the provider's api keys.
func pickEmbedKey(keys []string) string {
	embedRRMu.Lock()
	defer embedRRMu.Unlock()
	idx := embedRRIdx % len(keys)
	embedRRIdx++
	return keys[idx]
}

// Embeddings handles POST /v1/embeddings (OpenAI-compatible).
func (h *OpenAIAPIHandler) Embeddings(c *gin.Context) {
	c.Header("Content-Type", "application/json")

	rawJSON, err := handlers.ReadRequestBody(c)
	if err != nil {
		c.JSON(http.StatusBadRequest, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: "Invalid request: " + err.Error(),
				Type:    "invalid_request_error",
			},
		})
		return
	}

	modelName := gjson.GetBytes(rawJSON, "model").String()
	if modelName == "" {
		c.JSON(http.StatusBadRequest, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: "Missing required field 'model'",
				Type:    "invalid_request_error",
			},
		})
		return
	}

	baseURL, keys, upstreamModel, ok := resolveEmbedProvider(modelName)
	if !ok || baseURL == "" {
		c.JSON(http.StatusBadRequest, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: fmt.Sprintf("unknown provider for model %s (embedding model must be declared in config.yaml)", modelName),
				Type:    "invalid_request_error",
				Code:    "model_not_found",
			},
		})
		return
	}

	// Rewrite the body's "model" to the upstream name (NIM rejects aliases).
	payload := rawJSON
	if upstreamModel != modelName {
		rewritten, errSet := sjson.SetBytes(rawJSON, "model", upstreamModel)
		if errSet == nil {
			payload = rewritten
		}
	}

	upstreamURL := baseURL
	if upstreamURL[len(upstreamURL)-1] != '/' {
		upstreamURL += "/"
	}
	upstreamURL += "embeddings"

	ctx, cancel := context.WithTimeout(c.Request.Context(), 180*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, upstreamURL, bytes.NewReader(payload))
	if err != nil {
		c.JSON(http.StatusInternalServerError, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{Message: err.Error(), Type: "server_error"},
		})
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+pickEmbedKey(keys))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		c.JSON(http.StatusBadGateway, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{Message: err.Error(), Type: "upstream_error"},
		})
		return
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		c.JSON(http.StatusBadGateway, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{Message: err.Error(), Type: "upstream_error"},
		})
		return
	}
	c.Status(resp.StatusCode)
	_, _ = c.Writer.Write(body)
}
