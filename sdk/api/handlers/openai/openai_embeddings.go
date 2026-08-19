// Package openai provides HTTP handlers for OpenAI API endpoints.
//
// Implements POST /v1/embeddings as a pass-through proxy to the upstream
// provider declared in configuration for the requested model. Unlike chat
// completions, this bypasses the translator pipeline (which would forward to
// /v1/chat/completions) and calls the upstream {base-url}/embeddings endpoint
// directly, resolving the API key from the provider's api-key-entries.
package openai

import (
	"bytes"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"

	"github.com/gin-gonic/gin"
	"github.com/router-for-me/CLIProxyAPI/v7/sdk/api/handlers"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

var embedRRIdx atomic.Uint64

type embedResolvedTarget struct {
	baseURL       string
	apiKey        string
	proxyURL      string
	upstreamModel string
	headers       map[string]string
}

// resolveEmbedProvider maps a requested model name (alias or upstream name)
// to a resolved target using the in-memory configuration from h.Cfg.
func (h *OpenAIAPIHandler) resolveEmbedProvider(modelName string) (*embedResolvedTarget, bool) {
	if h == nil || h.Cfg == nil {
		return nil, false
	}
	for _, p := range h.Cfg.OpenAICompatibility {
		if p.Disabled {
			continue
		}
		upstreamName := ""
		for _, m := range p.Models {
			if m.Name == modelName || m.Alias == modelName {
				upstreamName = m.Name
				break
			}
		}
		if upstreamName == "" {
			continue
		}

		type keyEntry struct {
			apiKey   string
			proxyURL string
		}
		var validKeys []keyEntry
		for _, k := range p.APIKeyEntries {
			if k.APIKey != "" {
				validKeys = append(validKeys, keyEntry{
					apiKey:   k.APIKey,
					proxyURL: k.ProxyURL,
				})
			}
		}
		if len(validKeys) == 0 {
			continue
		}

		idx := embedRRIdx.Add(1) - 1
		chosen := validKeys[idx%uint64(len(validKeys))]

		proxy := chosen.proxyURL
		if proxy == "" {
			proxy = h.Cfg.ProxyURL
		}

		return &embedResolvedTarget{
			baseURL:       p.BaseURL,
			apiKey:        chosen.apiKey,
			proxyURL:      proxy,
			upstreamModel: upstreamName,
			headers:       p.Headers,
		}, true
	}
	return nil, false
}

func createEmbedHTTPClient(proxyURL string) *http.Client {
	if proxyURL == "" {
		return http.DefaultClient
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	if u, err := url.Parse(proxyURL); err == nil {
		transport.Proxy = http.ProxyURL(u)
	}
	return &http.Client{
		Transport: transport,
	}
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

	modelName := strings.TrimSpace(gjson.GetBytes(rawJSON, "model").String())
	if modelName == "" {
		c.JSON(http.StatusBadRequest, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: "Missing required field 'model'",
				Type:    "invalid_request_error",
			},
		})
		return
	}

	target, ok := h.resolveEmbedProvider(modelName)
	if !ok || target == nil || target.baseURL == "" {
		c.JSON(http.StatusBadRequest, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{
				Message: fmt.Sprintf("unknown provider for model %s (embedding model must be declared in config.yaml)", modelName),
				Type:    "invalid_request_error",
				Code:    "model_not_found",
			},
		})
		return
	}

	// Rewrite the body's "model" to the upstream name (some providers reject aliases).
	payload := rawJSON
	if target.upstreamModel != "" && target.upstreamModel != modelName {
		rewritten, errSet := sjson.SetBytes(rawJSON, "model", target.upstreamModel)
		if errSet == nil {
			payload = rewritten
		}
	}

	upstreamURL := strings.TrimSuffix(target.baseURL, "/") + "/embeddings"

	req, err := http.NewRequestWithContext(c.Request.Context(), http.MethodPost, upstreamURL, bytes.NewReader(payload))
	if err != nil {
		c.JSON(http.StatusInternalServerError, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{Message: err.Error(), Type: "server_error"},
		})
		return
	}
	req.Header.Set("Content-Type", "application/json")
	if target.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+target.apiKey)
	}
	for k, v := range target.headers {
		if k != "" && v != "" {
			req.Header.Set(k, v)
		}
	}

	client := createEmbedHTTPClient(target.proxyURL)
	resp, err := client.Do(req)
	if err != nil {
		c.JSON(http.StatusBadGateway, handlers.ErrorResponse{
			Error: handlers.ErrorDetail{Message: err.Error(), Type: "upstream_error"},
		})
		return
	}
	defer func() {
		_ = resp.Body.Close()
	}()

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
