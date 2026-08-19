package auth

import (
	"context"
	"fmt"
	"time"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/auth/vercel"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/browser"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	coreauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	log "github.com/sirupsen/logrus"
)

// vercelRefreshLead is the duration before token expiry when refresh should occur.
var vercelRefreshLead = 5 * time.Minute

// VercelAuthenticator implements the OAuth device flow login for Vercel (FX).
type VercelAuthenticator struct{}

// NewVercelAuthenticator constructs a new Vercel authenticator.
func NewVercelAuthenticator() Authenticator {
	return &VercelAuthenticator{}
}

// Provider returns the provider key for vercel.
func (VercelAuthenticator) Provider() string {
	return "vercel"
}

// RefreshLead returns the duration before token expiry when refresh should occur.
func (VercelAuthenticator) RefreshLead() *time.Duration {
	return &vercelRefreshLead
}

// Login initiates the Vercel device flow authentication.
func (a VercelAuthenticator) Login(ctx context.Context, cfg *config.Config, opts *LoginOptions) (*coreauth.Auth, error) {
	if cfg == nil {
		return nil, fmt.Errorf("cliproxy auth: configuration is required")
	}
	if opts == nil {
		opts = &LoginOptions{}
	}

	authSvc := vercel.NewVercelAuth(cfg)

	// Start the device flow
	fmt.Println("Starting Vercel (FX) authentication...")
	deviceCode, err := authSvc.StartDeviceFlow(ctx)
	if err != nil {
		return nil, fmt.Errorf("vercel: failed to start device flow: %w", err)
	}

	// Display the verification URL
	verificationURL := deviceCode.VerificationURIComplete
	if verificationURL == "" {
		verificationURL = deviceCode.VerificationURI
	}

	fmt.Printf("\nTo authenticate, please visit:\n%s\n\n", verificationURL)
	if deviceCode.UserCode != "" {
		fmt.Printf("User code: %s\n\n", deviceCode.UserCode)
	}

	// Try to open the browser automatically
	if !opts.NoBrowser {
		if browser.IsAvailable() {
			if errOpen := browser.OpenURL(verificationURL); errOpen != nil {
				log.Warnf("Failed to open browser automatically: %v", errOpen)
			} else {
				fmt.Println("Browser opened automatically.")
			}
		}
	}

	fmt.Println("Waiting for authorization...")
	if deviceCode.ExpiresIn > 0 {
		fmt.Printf("(This will timeout in %d seconds if not authorized)\n", deviceCode.ExpiresIn)
	}

	// Wait for user authorization
	authBundle, err := authSvc.WaitForAuthorization(ctx, deviceCode)
	if err != nil {
		return nil, fmt.Errorf("vercel: %w", err)
	}

	// Create the token storage
	tokenStorage := authSvc.CreateTokenStorage(authBundle)

	// Build metadata with token information
	metadata := map[string]any{
		"type":          "vercel",
		"access_token":  authBundle.TokenData.AccessToken,
		"refresh_token": authBundle.TokenData.RefreshToken,
		"token_type":    authBundle.TokenData.TokenType,
		"scope":         authBundle.TokenData.Scope,
		"timestamp":     time.Now().UnixMilli(),
	}

	if authBundle.TokenData.ExpiresAt > 0 {
		exp := time.Unix(authBundle.TokenData.ExpiresAt, 0).UTC().Format(time.RFC3339)
		metadata["expired"] = exp
	}

	// Generate a unique filename
	fileName := fmt.Sprintf("vercel-%d.json", time.Now().UnixMilli())

	fmt.Println("\nVercel authentication successful!")

	return &coreauth.Auth{
		ID:       fileName,
		Provider: a.Provider(),
		FileName: fileName,
		Label:    "Vercel User",
		Storage:  tokenStorage,
		Metadata: metadata,
	}, nil
}
