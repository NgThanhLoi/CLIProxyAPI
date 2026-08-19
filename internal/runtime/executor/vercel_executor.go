package executor

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	vercelauth "github.com/router-for-me/CLIProxyAPI/v7/internal/auth/vercel"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/runtime/executor/helps"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/thinking"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/util"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tiktoken-go/tokenizer"
)

// VercelExecutor is a stateless executor for Vercel AI Gateway (FX) using native gateway protocol.
type VercelExecutor struct {
	cfg *config.Config
}

// NewVercelExecutor creates a new Vercel executor.
func NewVercelExecutor(cfg *config.Config) *VercelExecutor {
	return &VercelExecutor{
		cfg: cfg,
	}
}

// Identifier returns the executor identifier.
func (e *VercelExecutor) Identifier() string { return "vercel" }

// RequestToFormat reports the upstream request format used after auth selection.
func (e *VercelExecutor) RequestToFormat(_ cliproxyexecutor.Request, _ cliproxyexecutor.Options) sdktranslator.Format {
	return sdktranslator.FormatOpenAI
}

// PrepareRequest injects Vercel credentials into the outgoing HTTP request.
func (e *VercelExecutor) PrepareRequest(req *http.Request, auth *cliproxyauth.Auth) error {
	if req == nil {
		return nil
	}
	token := vercelCreds(auth)
	if strings.TrimSpace(token) != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(req, attrs)
	return nil
}

// HttpRequest injects Vercel credentials into the request and executes it.
func (e *VercelExecutor) HttpRequest(ctx context.Context, auth *cliproxyauth.Auth, req *http.Request) (*http.Response, error) {
	if req == nil {
		return nil, fmt.Errorf("vercel executor: request is nil")
	}
	if ctx == nil {
		ctx = req.Context()
	}
	httpReq := req.WithContext(ctx)
	if err := e.PrepareRequest(httpReq, auth); err != nil {
		return nil, err
	}
	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	return httpClient.Do(httpReq)
}

// Execute performs a non-streaming chat completion request to Vercel AI Gateway.
func (e *VercelExecutor) Execute(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	responseFormat := cliproxyexecutor.ResponseFormatOrSource(opts)
	baseModel := thinking.ParseSuffix(req.Model).ModelName
	token := vercelCreds(auth)

	reporter := helps.NewExecutorUsageReporter(ctx, e, baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	from := opts.SourceFormat
	to := sdktranslator.FromString("openai")

	body := helps.TranslateRequestWithCodexMultiAgentV2(ctx, opts.Headers, e.cfg, from, to, baseModel, bytes.Clone(req.Payload), false)

	upstreamModel := normalizeVercelUpstreamModel(baseModel)

	// Transform OpenAI payload to Vercel AI Gateway payload
	vercelPayload := helps.TransformToVercel(upstreamModel, body)

	targetURL := vercelauth.VercelAIGatewayChatURL
	if auth != nil && auth.Attributes != nil && auth.Attributes["base_url"] != "" {
		targetURL = auth.Attributes["base_url"]
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(vercelPayload))
	if err != nil {
		return resp, err
	}
	applyVercelHeaders(httpReq, token, upstreamModel)

	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(httpReq, attrs)

	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	helps.RecordAPIRequest(ctx, e.cfg, helps.UpstreamRequestLog{
		URL:       targetURL,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      vercelPayload,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpClient = reporter.TrackHTTPClient(httpClient)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("vercel executor: close response error: %v", errClose)
		}
	}()

	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	bodyBytes, err := io.ReadAll(httpResp.Body)
	if err != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	helps.AppendAPIResponseChunk(ctx, e.cfg, bodyBytes)

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		helps.LogWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), bodyBytes))
		err = statusErr{code: httpResp.StatusCode, msg: string(bodyBytes)}
		return resp, err
	}

	// Transform Vercel SSE output to OpenAI Chat Completion JSON
	openaiJSON, err := helps.TransformVercelResponseToOpenAI(baseModel, bodyBytes)
	if err != nil {
		return resp, err
	}

	reporter.Publish(ctx, helps.ParseOpenAIUsage(openaiJSON))
	var param any
	out := sdktranslator.TranslateNonStream(ctx, to, responseFormat, req.Model, opts.OriginalRequest, body, openaiJSON, &param)
	if responseFormat == sdktranslator.FormatOpenAIResponse {
		out = helps.EnsureResponsesUsageDetails(out)
	}

	return cliproxyexecutor.Response{
		Payload: out,
		Headers: httpResp.Header.Clone(),
	}, nil
}

// ExecuteStream performs a streaming chat completion request to Vercel AI Gateway.
func (e *VercelExecutor) ExecuteStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (_ *cliproxyexecutor.StreamResult, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName
	token := vercelCreds(auth)

	reporter := helps.NewExecutorUsageReporter(ctx, e, baseModel, auth)
	defer reporter.TrackFailure(ctx, &err)

	from := opts.SourceFormat
	to := sdktranslator.FromString("openai")

	body := helps.TranslateRequestWithCodexMultiAgentV2(ctx, opts.Headers, e.cfg, from, to, baseModel, bytes.Clone(req.Payload), false)
	upstreamModel := normalizeVercelUpstreamModel(baseModel)

	vercelPayload := helps.TransformToVercel(upstreamModel, body)

	targetURL := vercelauth.VercelAIGatewayChatURL
	if auth != nil && auth.Attributes != nil && auth.Attributes["base_url"] != "" {
		targetURL = auth.Attributes["base_url"]
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(vercelPayload))
	if err != nil {
		return nil, err
	}
	applyVercelHeaders(httpReq, token, upstreamModel)

	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(httpReq, attrs)

	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	helps.RecordAPIRequest(ctx, e.cfg, helps.UpstreamRequestLog{
		URL:       targetURL,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      vercelPayload,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := helps.NewProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpClient = reporter.TrackHTTPClient(httpClient)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		helps.RecordAPIResponseError(ctx, e.cfg, err)
		return nil, err
	}

	helps.RecordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		helps.AppendAPIResponseChunk(ctx, e.cfg, b)
		helps.LogWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, helps.SummarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("vercel executor: close response body error: %v", errClose)
		}
		err = statusErr{code: httpResp.StatusCode, msg: string(b)}
		return nil, err
	}

	chatID := "chatcmpl-" + helps.GenerateVercelUUID()
	out := make(chan cliproxyexecutor.StreamChunk)

	go func() {
		defer close(out)
		defer func() {
			if errClose := httpResp.Body.Close(); errClose != nil {
				log.Errorf("vercel executor: close response body error: %v", errClose)
			}
		}()

		reader := bufio.NewReader(httpResp.Body)
		for {
			select {
			case <-ctx.Done():
				return
			default:
			}

			line, errRead := reader.ReadString('\n')
			if len(line) > 0 {
				chunk := helps.TransformVercelStreamChunk(baseModel, line, chatID)
				if len(chunk) > 0 {
					helps.AppendAPIResponseChunk(ctx, e.cfg, chunk)
					out <- cliproxyexecutor.StreamChunk{Payload: chunk}
				}
			}
			if errRead != nil {
				if errRead != io.EOF {
					log.Warnf("vercel executor: stream read error: %v", errRead)
				}
				doneChunk := []byte("data: [DONE]\n\n")
				helps.AppendAPIResponseChunk(ctx, e.cfg, doneChunk)
				out <- cliproxyexecutor.StreamChunk{Payload: doneChunk}
				return
			}
		}
	}()

	return &cliproxyexecutor.StreamResult{
		Headers: httpResp.Header.Clone(),
		Chunks:  out,
	}, nil
}

func vercelCreds(auth *cliproxyauth.Auth) string {
	if auth == nil {
		return ""
	}
	if tok, ok := auth.Metadata["access_token"].(string); ok && strings.TrimSpace(tok) != "" {
		return strings.TrimSpace(tok)
	}
	if tok, ok := auth.Attributes["api_key"]; ok && strings.TrimSpace(tok) != "" {
		return strings.TrimSpace(tok)
	}
	return ""
}

func applyVercelHeaders(req *http.Request, token, model string) {
	if req == nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if strings.TrimSpace(token) != "" {
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(token))
	}
	req.Header.Set("ai-gateway-protocol-version", "0.0.1")
	req.Header.Set("ai-language-model-specification-version", "4")
	req.Header.Set("ai-language-model-id", model)
	req.Header.Set("ai-language-model-streaming", "true")
	req.Header.Set("HTTP-Referer", "https://github.com/vercel-labs/fx")
	req.Header.Set("X-Title", "fx")
	req.Header.Set("User-Agent", "fx/0.1.0")
}

// Refresh refreshes Vercel OAuth credentials using the stored refresh token.
func (e *VercelExecutor) Refresh(ctx context.Context, auth *cliproxyauth.Auth) (*cliproxyauth.Auth, error) {
	log.Debugf("vercel executor: refresh called")
	if refreshed, handled, err := helps.RefreshAuthViaHome(ctx, e.cfg, auth); handled {
		return refreshed, err
	}
	if auth == nil {
		return nil, statusErr{code: http.StatusInternalServerError, msg: "vercel executor: auth is nil"}
	}
	var refreshToken string
	if auth.Metadata != nil {
		if rt, ok := auth.Metadata["refresh_token"].(string); ok {
			refreshToken = rt
		}
	}
	if refreshToken == "" {
		return auth, nil
	}

	proxyURL := auth.ProxyURL
	if proxyURL == "" && e.cfg != nil {
		proxyURL = e.cfg.ProxyURL
	}

	td, err := vercelauth.RefreshTokenSingleFlight(ctx, e.cfg, refreshToken, proxyURL)
	if err != nil {
		return nil, err
	}

	if auth.Metadata == nil {
		auth.Metadata = make(map[string]any)
	}
	auth.Metadata["type"] = "vercel"
	auth.Metadata["access_token"] = td.AccessToken
	if td.RefreshToken != "" {
		auth.Metadata["refresh_token"] = td.RefreshToken
	}
	if td.TokenType != "" {
		auth.Metadata["token_type"] = td.TokenType
	}
	if td.ExpiresAt > 0 {
		exp := time.Unix(td.ExpiresAt, 0).UTC().Format(time.RFC3339)
		auth.Metadata["expired"] = exp
	}
	return auth, nil
}

// CountTokens estimates token count for Vercel chat requests.
func (e *VercelExecutor) CountTokens(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	enc, err := tokenizer.Get(tokenizer.O200kBase)
	if err != nil {
		return cliproxyexecutor.Response{}, fmt.Errorf("vercel executor: tokenizer init failed: %w", err)
	}
	countInt, err := enc.Count(string(req.Payload))
	if err != nil {
		countInt = len(req.Payload) / 4
	}
	count := int64(countInt)
	usageJSON := fmt.Sprintf(`{"input_tokens":%d,"total_tokens":%d}`, count, count)
	from := opts.SourceFormat
	to := sdktranslator.FromString("openai")
	translated := sdktranslator.TranslateTokenCount(ctx, to, from, count, []byte(usageJSON))
	return cliproxyexecutor.Response{Payload: translated}, nil
}

func normalizeVercelUpstreamModel(model string) string {
	m := strings.TrimSpace(model)
	m = strings.TrimPrefix(m, "vercel/")
	m = strings.TrimPrefix(m, "fx/")
	return m
}
