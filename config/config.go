package config

import (
	"os"

	"github.com/joho/godotenv"
)

// Config holds all configuration for the application
type Config struct {
	API      APIConfig
	Agent    AgentConfig
	Security SecurityConfig
	UI       UIConfig
}

// APIConfig holds API-related configuration
type APIConfig struct {
	Key string
}

// AgentConfig holds agent behavior configuration
type AgentConfig struct {
	SystemPromptFile string
	WorkingDir       string
	TokenLimits      TokenLimits
}

// TokenLimits holds token management configuration
type TokenLimits struct {
	MaxOutputTokens    int // For API calls (10K)
	MaxInputTokens     int // For conversation management (200K)
	WarningThreshold   int // Show warning at 190K
	RecentMessagesKeep int
	SummaryTokenTarget int
}

// SecurityConfig holds security-related configuration
type SecurityConfig struct {
	AllowDangerousCommands bool
	RequireApproval        bool
}

// UIConfig holds UI-related configuration
type UIConfig struct {
	ShowThinking   bool
	AnimationSpeed int // milliseconds
	ColorOutput    bool
}

// Load loads configuration from environment and defaults
func Load() (*Config, error) {
	// Load environment variables from .env file (if it exists)
	_ = godotenv.Load()

	config := &Config{
		API: APIConfig{
			Key: os.Getenv("ANTHROPIC_API_KEY"),
		},
		Agent: AgentConfig{
			SystemPromptFile: "system_prompt.txt",
			TokenLimits: TokenLimits{
				MaxOutputTokens:    MaxOutputTokens,
				MaxInputTokens:     MaxInputTokens,
				WarningThreshold:   WarningThreshold,
				RecentMessagesKeep: RecentMessagesKeep,
				SummaryTokenTarget: SummaryTokenTarget,
			},
		},
		Security: SecurityConfig{
			AllowDangerousCommands: false,
			RequireApproval:        true,
		},
		UI: UIConfig{
			ShowThinking:   true,
			AnimationSpeed: 500,
			ColorOutput:    true,
		},
	}

	return config, nil
}

// NewConfig creates a new configuration with default values
func NewConfig() *Config {
	config, _ := Load()
	return config
}

// MaxTokens returns the maximum output token limit for API calls
func (c *Config) MaxTokens() int {
	return c.Agent.TokenLimits.MaxOutputTokens
}

// RecentMessagesKeep returns the number of recent messages to keep
func (c *Config) RecentMessagesKeep() int {
	return c.Agent.TokenLimits.RecentMessagesKeep
}

// MaxInputTokens returns the maximum input token limit for conversation management
func (c *Config) MaxInputTokens() int {
	return c.Agent.TokenLimits.MaxInputTokens
}

// WarningThreshold returns the token count at which to show warnings
func (c *Config) WarningThreshold() int {
	return c.Agent.TokenLimits.WarningThreshold
}
