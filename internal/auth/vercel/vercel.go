// Package vercel provides authentication and token management for Vercel (FX) API.
// It handles the RFC 8628 OAuth2 Device Authorization Grant flow for secure authentication.
package vercel

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/misc"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/util"
	log "github.com/sirupsen/logrus"
	"golang.org/x/sync/singleflight"
)

const (
	// VercelClientID is Vercel FX's official OAuth client ID.
	VercelClientID = "cl_zzh5hiOZbwJ9bfqEcYqPIJv3TaPaEYL0"
	// VercelOAuthDeviceCodeURL is the endpoint for requesting device authorization codes.
	VercelOAuthDeviceCodeURL = "https://api.vercel.com/login/oauth/device-authorization"
	// VercelOAuthTokenURL is the endpoint for exchanging device codes and refreshing tokens.
	VercelOAuthTokenURL = "https://api.vercel.com/login/oauth/token"
	// VercelAIGatewayChatURL is the base chat URL for Vercel AI Gateway.
	VercelAIGatewayChatURL = "https://ai-gateway.vercel.sh/v3/ai/language-model"
	// defaultScope is the OAuth scope required for identity and refresh token.
	defaultScope = "openid offline_access"
	// defaultPollInterval is the default interval for polling the token endpoint.
	defaultPollInterval = 5 * time.Second
	// maxPollDuration is the maximum time to wait for user authorization (10 minutes).
	maxPollDuration = 10 * time.Minute
)

var vercelRefreshGroup singleflight.Group

// VercelAuth handles Vercel authentication flow.
type VercelAuth struct {
	deviceClient *DeviceFlowClient
	cfg          *config.Config
}

// NewVercelAuth creates a new VercelAuth service instance.
func NewVercelAuth(cfg *config.Config) *VercelAuth {
	return &VercelAuth{
		deviceClient: NewDeviceFlowClient(cfg),
		cfg:          cfg,
	}
}

// StartDeviceFlow initiates the device flow authentication.
func (v *VercelAuth) StartDeviceFlow(ctx context.Context) (*DeviceCodeResponse, error) {
	return v.deviceClient.RequestDeviceCode(ctx)
}

// WaitForAuthorization polls for user authorization and returns the auth bundle.
func (v *VercelAuth) WaitForAuthorization(ctx context.Context, deviceCode *DeviceCodeResponse) (*VercelAuthBundle, error) {
	tokenData, err := v.deviceClient.PollForToken(ctx, deviceCode)
	if err != nil {
		return nil, err
	}

	return &VercelAuthBundle{
		TokenData: tokenData,
	}, nil
}

// CreateTokenStorage creates a new VercelTokenStorage from auth bundle.
func (v *VercelAuth) CreateTokenStorage(bundle *VercelAuthBundle) *VercelTokenStorage {
	expired := ""
	if bundle.TokenData.ExpiresAt > 0 {
		expired = time.Unix(bundle.TokenData.ExpiresAt, 0).UTC().Format(time.RFC3339)
	}
	return &VercelTokenStorage{
		AccessToken:  bundle.TokenData.AccessToken,
		RefreshToken: bundle.TokenData.RefreshToken,
		TokenType:    bundle.TokenData.TokenType,
		Scope:        bundle.TokenData.Scope,
		Expired:      expired,
		Type:         "vercel",
	}
}

// DeviceFlowClient handles the OAuth2 device flow for Vercel.
type DeviceFlowClient struct {
	httpClient *http.Client
	cfg        *config.Config
}

// NewDeviceFlowClient creates a new device flow client.
func NewDeviceFlowClient(cfg *config.Config) *DeviceFlowClient {
	return NewDeviceFlowClientWithProxyURL(cfg, "")
}

// NewDeviceFlowClientWithProxyURL creates a new device flow client with a proxy override.
func NewDeviceFlowClientWithProxyURL(cfg *config.Config, proxyURL string) *DeviceFlowClient {
	client := &http.Client{Timeout: 30 * time.Second}
	effectiveProxyURL := strings.TrimSpace(proxyURL)
	var sdkCfg config.SDKConfig
	if cfg != nil {
		sdkCfg = cfg.SDKConfig
		if effectiveProxyURL == "" {
			effectiveProxyURL = strings.TrimSpace(cfg.ProxyURL)
		}
	}
	sdkCfg.ProxyURL = effectiveProxyURL
	client = util.SetProxy(&sdkCfg, client)

	return &DeviceFlowClient{
		httpClient: client,
		cfg:        cfg,
	}
}

// DeviceCodeResponse represents the response from the device authorization endpoint.
type DeviceCodeResponse struct {
	DeviceCode              string `json:"device_code"`
	UserCode                string `json:"user_code"`
	VerificationURI         string `json:"verification_uri"`
	VerificationURIComplete string `json:"verification_uri_complete"`
	ExpiresIn               int    `json:"expires_in"`
	Interval                int    `json:"interval"`
}

// VercelTokenData represents the token response from Vercel OAuth.
type VercelTokenData struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	TokenType    string `json:"token_type"`
	ExpiresAt    int64  `json:"expires_at"`
	Scope        string `json:"scope"`
}

// VercelAuthBundle contains the complete auth state after successful login.
type VercelAuthBundle struct {
	TokenData *VercelTokenData `json:"token_data"`
}

// VercelTokenStorage is the persisted token structure.
type VercelTokenStorage struct {
	AccessToken  string                 `json:"access_token"`
	RefreshToken string                 `json:"refresh_token"`
	TokenType    string                 `json:"token_type"`
	Scope        string                 `json:"scope"`
	Expired      string                 `json:"expired"`
	Type         string                 `json:"type"`
	Metadata     map[string]interface{} `json:"-"`
}

// SaveTokenToFile serializes the Vercel token storage to a JSON file.
func (ts *VercelTokenStorage) SaveTokenToFile(authFilePath string) error {
	misc.LogSavingCredentials(authFilePath)
	ts.Type = "vercel"

	if err := os.MkdirAll(filepath.Dir(authFilePath), 0700); err != nil {
		return fmt.Errorf("failed to create directory: %v", err)
	}

	data, errMerge := misc.MergeMetadata(ts, ts.Metadata)
	if errMerge != nil {
		return fmt.Errorf("failed to merge metadata: %w", errMerge)
	}

	f, err := os.Create(authFilePath)
	if err != nil {
		return fmt.Errorf("failed to create token file: %w", err)
	}
	defer func() {
		if errClose := f.Close(); errClose != nil {
			log.Errorf("vercel token storage: close token file error: %v", errClose)
		}
	}()

	encoder := json.NewEncoder(f)
	encoder.SetIndent("", "  ")
	if err = encoder.Encode(data); err != nil {
		return fmt.Errorf("failed to write token to file: %w", err)
	}

	return nil
}

// RequestDeviceCode initiates the device flow by requesting a device code from Vercel.
func (c *DeviceFlowClient) RequestDeviceCode(ctx context.Context) (*DeviceCodeResponse, error) {
	data := url.Values{}
	data.Set("client_id", VercelClientID)
	data.Set("scope", defaultScope)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, VercelOAuthDeviceCodeURL, strings.NewReader(data.Encode()))
	if err != nil {
		return nil, fmt.Errorf("vercel: failed to create device code request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("vercel: device code request failed: %w", err)
	}
	defer func() {
		if errClose := resp.Body.Close(); errClose != nil {
			log.Errorf("vercel device code: close body error: %v", errClose)
		}
	}()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("vercel: failed to read device code response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("vercel: device code request failed with status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var deviceCodeResp DeviceCodeResponse
	if err = json.Unmarshal(bodyBytes, &deviceCodeResp); err != nil {
		return nil, fmt.Errorf("vercel: failed to parse device code response: %w", err)
	}

	if deviceCodeResp.Interval <= 0 {
		deviceCodeResp.Interval = int(defaultPollInterval / time.Second)
	}

	return &deviceCodeResp, nil
}

// PollForToken polls the token endpoint until the user authorizes or the flow times out.
func (c *DeviceFlowClient) PollForToken(ctx context.Context, deviceCode *DeviceCodeResponse) (*VercelTokenData, error) {
	interval := time.Duration(deviceCode.Interval) * time.Second
	if interval < defaultPollInterval {
		interval = defaultPollInterval
	}

	timeout := maxPollDuration
	if deviceCode.ExpiresIn > 0 {
		timeout = time.Duration(deviceCode.ExpiresIn) * time.Second
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			if ctx.Err() == context.DeadlineExceeded {
				return nil, fmt.Errorf("vercel: device code expired")
			}
			return nil, ctx.Err()

		case <-ticker.C:
			token, pollErr, shouldContinue := c.exchangeDeviceCode(ctx, deviceCode.DeviceCode)
			if token != nil {
				return token, nil
			}
			if !shouldContinue {
				return nil, pollErr
			}
		}
	}
}

// exchangeDeviceCode attempts to exchange the device code for an access token.
func (c *DeviceFlowClient) exchangeDeviceCode(ctx context.Context, deviceCode string) (*VercelTokenData, error, bool) {
	data := url.Values{}
	data.Set("client_id", VercelClientID)
	data.Set("device_code", deviceCode)
	data.Set("grant_type", "urn:ietf:params:oauth:grant-type:device_code")

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, VercelOAuthTokenURL, strings.NewReader(data.Encode()))
	if err != nil {
		return nil, fmt.Errorf("vercel: failed to create token request: %w", err), false
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("vercel: token request failed: %w", err), false
	}
	defer func() {
		if errClose := resp.Body.Close(); errClose != nil {
			log.Errorf("vercel token exchange: close body error: %v", errClose)
		}
	}()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("vercel: failed to read token response: %w", err), false
	}

	var oauthResp struct {
		Error            string  `json:"error"`
		ErrorDescription string  `json:"error_description"`
		AccessToken      string  `json:"access_token"`
		RefreshToken     string  `json:"refresh_token"`
		TokenType        string  `json:"token_type"`
		ExpiresIn        float64 `json:"expires_in"`
		Scope            string  `json:"scope"`
	}

	if err = json.Unmarshal(bodyBytes, &oauthResp); err != nil {
		return nil, fmt.Errorf("vercel: failed to parse token response: %w", err), false
	}

	if oauthResp.Error != "" {
		switch oauthResp.Error {
		case "authorization_pending":
			return nil, nil, true // Continue polling
		case "slow_down":
			return nil, nil, true // Continue polling
		case "expired_token":
			return nil, fmt.Errorf("vercel: device code expired"), false
		case "access_denied":
			return nil, fmt.Errorf("vercel: access denied by user"), false
		default:
			return nil, fmt.Errorf("vercel: OAuth error: %s - %s", oauthResp.Error, oauthResp.ErrorDescription), false
		}
	}

	if oauthResp.AccessToken == "" {
		return nil, fmt.Errorf("vercel: empty access token in response"), false
	}

	var expiresAt int64
	if oauthResp.ExpiresIn > 0 {
		expiresAt = time.Now().Unix() + int64(oauthResp.ExpiresIn)
	}

	return &VercelTokenData{
		AccessToken:  oauthResp.AccessToken,
		RefreshToken: oauthResp.RefreshToken,
		TokenType:    oauthResp.TokenType,
		ExpiresAt:    expiresAt,
		Scope:        oauthResp.Scope,
	}, nil, false
}

// RefreshToken exchanges a refresh token for a new access token.
func (c *DeviceFlowClient) RefreshToken(ctx context.Context, refreshToken string) (*VercelTokenData, error) {
	if strings.TrimSpace(refreshToken) == "" {
		return nil, fmt.Errorf("vercel: refresh token is required")
	}

	data := url.Values{}
	data.Set("client_id", VercelClientID)
	data.Set("grant_type", "refresh_token")
	data.Set("refresh_token", refreshToken)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, VercelOAuthTokenURL, strings.NewReader(data.Encode()))
	if err != nil {
		return nil, fmt.Errorf("vercel: failed to create refresh request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("vercel: refresh request failed: %w", err)
	}
	defer func() {
		if errClose := resp.Body.Close(); errClose != nil {
			log.Errorf("vercel refresh: close body error: %v", errClose)
		}
	}()

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("vercel: failed to read refresh response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("vercel: refresh failed with status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var oauthResp struct {
		Error            string  `json:"error"`
		ErrorDescription string  `json:"error_description"`
		AccessToken      string  `json:"access_token"`
		RefreshToken     string  `json:"refresh_token"`
		TokenType        string  `json:"token_type"`
		ExpiresIn        float64 `json:"expires_in"`
		Scope            string  `json:"scope"`
	}

	if err = json.Unmarshal(bodyBytes, &oauthResp); err != nil {
		return nil, fmt.Errorf("vercel: failed to parse refresh response: %w", err)
	}

	if oauthResp.Error != "" {
		return nil, fmt.Errorf("vercel: refresh error: %s - %s", oauthResp.Error, oauthResp.ErrorDescription)
	}

	if oauthResp.AccessToken == "" {
		return nil, fmt.Errorf("vercel: empty access token in refresh response")
	}

	newRefreshToken := oauthResp.RefreshToken
	if newRefreshToken == "" {
		newRefreshToken = refreshToken // Keep old refresh token if not rotated
	}

	var expiresAt int64
	if oauthResp.ExpiresIn > 0 {
		expiresAt = time.Now().Unix() + int64(oauthResp.ExpiresIn)
	}

	return &VercelTokenData{
		AccessToken:  oauthResp.AccessToken,
		RefreshToken: newRefreshToken,
		TokenType:    oauthResp.TokenType,
		ExpiresAt:    expiresAt,
		Scope:        oauthResp.Scope,
	}, nil
}

// RefreshTokenSingleFlight performs token refresh with singleflight deduplication.
func RefreshTokenSingleFlight(ctx context.Context, cfg *config.Config, refreshToken string, proxyURL string) (*VercelTokenData, error) {
	key := fmt.Sprintf("vercel_refresh_%s", refreshToken)
	result, err, _ := vercelRefreshGroup.Do(key, func() (interface{}, error) {
		client := NewDeviceFlowClientWithProxyURL(cfg, proxyURL)
		return client.RefreshToken(ctx, refreshToken)
	})
	if err != nil {
		return nil, err
	}
	return result.(*VercelTokenData), nil
}
