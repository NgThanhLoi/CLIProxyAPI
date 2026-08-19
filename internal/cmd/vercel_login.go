package cmd

import (
	"context"
	"fmt"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	sdkAuth "github.com/router-for-me/CLIProxyAPI/v7/sdk/auth"
	log "github.com/sirupsen/logrus"
)

// DoVercelLogin triggers the OAuth device flow for Vercel (FX) and saves tokens.
// It initiates the device flow authentication, displays the verification URL for the user,
// and waits for authorization before saving the tokens.
//
// Parameters:
//   - cfg: The application configuration containing proxy and auth directory settings
//   - options: Login options including browser behavior settings
func DoVercelLogin(cfg *config.Config, options *LoginOptions) {
	if options == nil {
		options = &LoginOptions{}
	}

	manager := newAuthManager()
	authOpts := &sdkAuth.LoginOptions{
		NoBrowser: options.NoBrowser,
		Metadata:  map[string]string{},
		Prompt:    options.Prompt,
	}

	record, savedPath, err := manager.Login(context.Background(), "vercel", cfg, authOpts)
	if err != nil {
		log.Errorf("Vercel authentication failed: %v", err)
		return
	}

	if savedPath != "" {
		fmt.Printf("Authentication saved to %s\n", savedPath)
	}
	if record != nil && record.Label != "" {
		fmt.Printf("Authenticated as %s\n", record.Label)
	}
	fmt.Println("Vercel (FX) authentication successful!")
}
