package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"anthropic-chat/config"
	"anthropic-chat/tools"
	"anthropic-chat/tools/file"
	"anthropic-chat/ui"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/joho/godotenv"
)

func main() {
	// Load environment variables
	if err := godotenv.Load(); err != nil {
		log.Printf("Warning: .env file not found or couldn't be loaded: %v", err)
	}

	// Get API key
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		log.Fatal("ANTHROPIC_API_KEY environment variable is required")
	}

	// Create Anthropic client
	client := anthropic.NewClient(option.WithAPIKey(apiKey))

	// Set up user input handler
	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	// Prompt for working directory
	workingDir, err := promptForDirectory(scanner)
	if err != nil {
		log.Fatal("Failed to set working directory:", err)
	}

	fmt.Printf("Working directory set to: %s\n\n", workingDir)

	// Create and configure agent
	agent := NewRefactoredAgent(&client, getUserMessage, workingDir)

	// Register tools using the new system
	agent.RegisterTools()

	// Run the agent
	if err := agent.Run(context.TODO()); err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

// RefactoredAgent represents the improved agent architecture
type RefactoredAgent struct {
	client         *anthropic.Client
	getUserMessage func() (string, bool)
	workingDir     string
	systemPrompt   string
	toolRegistry   *tools.Registry
	config         *config.Config
	uiManager      *ui.Manager
}

// NewRefactoredAgent creates a new agent with the improved architecture
func NewRefactoredAgent(client *anthropic.Client, getUserMessage func() (string, bool), workingDir string) *RefactoredAgent {
	return &RefactoredAgent{
		client:         client,
		getUserMessage: getUserMessage,
		workingDir:     workingDir,
		systemPrompt:   loadSystemPrompt(),
		toolRegistry:   tools.NewRegistry(),
		config:         config.NewConfig(),
		uiManager:      ui.NewManager(),
	}
}

// RegisterTools registers all available tools with the agent
func (a *RefactoredAgent) RegisterTools() {
	// Register file operation tools
	a.toolRegistry.Register(file.NewReadFileTool())
	a.toolRegistry.Register(file.NewListFilesTool())
	// Note: Would register other tools here:
	// a.toolRegistry.Register(file.NewEditFileTool())
	// a.toolRegistry.Register(file.NewDuplicateFileTool())
	// a.toolRegistry.Register(command.NewExecuteCommandTool())
}

// WorkingDir implements the ToolContext interface
func (a *RefactoredAgent) WorkingDir() string {
	return a.workingDir
}

// ResolveFilePath implements the ToolContext interface with security validation
func (a *RefactoredAgent) ResolveFilePath(relativePath string) (string, error) {
	// Clean the path to prevent directory traversal
	cleanPath := filepath.Clean(relativePath)

	// Prevent paths from escaping the working directory
	if strings.Contains(cleanPath, "..") {
		return "", fmt.Errorf("path cannot contain '..' for security reasons")
	}

	// Join with working directory
	fullPath := filepath.Join(a.workingDir, cleanPath)

	// Ensure the resolved path is still within the working directory
	absWorkingDir, err := filepath.Abs(a.workingDir)
	if err != nil {
		return "", fmt.Errorf("failed to get absolute working directory: %w", err)
	}

	absFullPath, err := filepath.Abs(fullPath)
	if err != nil {
		return "", fmt.Errorf("failed to get absolute path: %w", err)
	}

	if !strings.HasPrefix(absFullPath, absWorkingDir) {
		return "", fmt.Errorf("path escapes working directory")
	}

	return fullPath, nil
}

// Run executes the main agent loop
func (a *RefactoredAgent) Run(ctx context.Context) error {
	conversation := []anthropic.MessageParam{}

	// Display welcome message
	a.uiManager.ShowWelcome()
	a.uiManager.ShowCommands()

	for {
		fmt.Print("\u001b[94mYou\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}

		// Handle slash commands
		if handled := a.handleSlashCommand(ctx, userInput, conversation); handled {
			continue
		}

		// Add user message to conversation
		userMessage := anthropic.NewUserMessage(anthropic.NewTextBlock(userInput))
		conversation = append(conversation, userMessage)

		// Manage conversation length
		managedConversation, err := a.manageConversationLength(ctx, conversation)
		if err != nil {
			log.Printf("Warning: failed to manage conversation length: %v", err)
		} else {
			conversation = managedConversation
		}

		// Process conversation with tool execution loop
		for {
			message, err := a.runInference(ctx, conversation)
			if err != nil {
				return err
			}
			conversation = append(conversation, message.ToParam())

			// Process tool use blocks
			toolResults := []anthropic.ContentBlockParamUnion{}
			hasToolUse := false

			for _, content := range message.Content {
				if block, ok := content.AsAny().(anthropic.ToolUseBlock); ok {
					hasToolUse = true

					// Execute tool using the new registry system
					result, err := a.toolRegistry.Execute(ctx, a, block.Name, block.Input)
					if err != nil {
						result = fmt.Sprintf("Error executing tool: %s", err.Error())
					}

					fmt.Printf("\u001b[96m[Tool Result]\u001b[0m: %s\n", result)
					toolResults = append(toolResults, anthropic.NewToolResultBlock(block.ID, result, false))
				}
			}

			if !hasToolUse {
				break
			}

			// Add tool results to conversation and continue
			if len(toolResults) > 0 {
				conversation = append(conversation, anthropic.NewUserMessage(toolResults...))
			}
		}
	}

	return nil
}

// handleSlashCommand processes slash commands and returns true if handled
func (a *RefactoredAgent) handleSlashCommand(ctx context.Context, input string, conversation []anthropic.MessageParam) bool {
	if strings.HasPrefix(input, "/cd") {
		scanner := bufio.NewScanner(os.Stdin)
		fmt.Print("Enter new directory path: ")
		if scanner.Scan() {
			newDir := strings.TrimSpace(scanner.Text())
			if newDir != "" {
				// Expand ~ to home directory
				if strings.HasPrefix(newDir, "~/") {
					home, err := os.UserHomeDir()
					if err != nil {
						fmt.Printf("\u001b[91mError\u001b[0m: Failed to get home directory: %v\n\n", err)
						return true
					}
					newDir = filepath.Join(home, newDir[2:])
				}

				// Clean and validate the path
				newDir = filepath.Clean(newDir)
				if err := validateDirectory(newDir); err != nil {
					fmt.Printf("\u001b[91mError\u001b[0m: %v\n\n", err)
				} else {
					a.workingDir = newDir
					fmt.Printf("\u001b[92mWorking directory changed to:\u001b[0m %s\n\n", newDir)
				}
			}
		}
		return true
	}

	if strings.HasPrefix(input, "/tokens") {
		if len(conversation) == 0 {
			fmt.Printf("\u001b[96mToken Info\u001b[0m: No conversation yet (0 tokens)\n\n")
		} else {
			tokenCount, err := a.countConversationTokens(ctx, conversation)
			if err != nil {
				fmt.Printf("\u001b[91mError\u001b[0m: Failed to count tokens: %v\n\n", err)
			} else {
				percentage := float64(tokenCount) / float64(a.config.MaxInputTokens()) * 100
				fmt.Printf("\u001b[96mToken Info\u001b[0m: Current conversation has %d tokens (%.1f%% of %d input limit)\n", tokenCount, percentage, a.config.MaxInputTokens())
				fmt.Printf("\u001b[96mToken Info\u001b[0m: Max output tokens per response: %d\n", a.config.MaxTokens())
				fmt.Printf("\u001b[96mToken Info\u001b[0m: %d messages in conversation\n\n", len(conversation))

				// Show warning if approaching threshold
				if tokenCount >= a.config.WarningThreshold() {
					fmt.Printf("\u001b[93m⚠️  Warning\u001b[0m: Approaching input token limit (%d/%d tokens)\n", tokenCount, a.config.MaxInputTokens())
					fmt.Printf("\u001b[93m⚠️  Warning\u001b[0m: Conversation will be summarized soon to manage length\n\n")
				}
			}
		}
		return true
	}

	return false
}

// runInference handles the Anthropic API call with streaming
func (a *RefactoredAgent) runInference(ctx context.Context, conversation []anthropic.MessageParam) (*anthropic.Message, error) {
	// Convert tools to Anthropic format
	toolDefs := a.toolRegistry.All()
	toolParams := make([]anthropic.ToolParam, len(toolDefs))
	for i, tool := range toolDefs {
		toolParams[i] = anthropic.ToolParam{
			Name:        tool.Name,
			Description: anthropic.String(tool.Description),
			InputSchema: tool.InputSchema,
		}
	}

	tools := make([]anthropic.ToolUnionParam, len(toolParams))
	for i, toolParam := range toolParams {
		tools[i] = anthropic.ToolUnionParam{OfTool: &toolParam}
	}

	// Start thinking animation
	animation := a.uiManager.NewThinkingAnimation()
	animation.Start()

	// Use streaming API
	stream := a.client.Messages.NewStreaming(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaude3_7SonnetLatest,
		MaxTokens: int64(a.config.MaxTokens()),
		System: []anthropic.TextBlockParam{
			{Text: a.systemPrompt},
		},
		Messages: conversation,
		Tools:    tools,
	})

	message := anthropic.Message{}
	hasStartedTextOutput := false
	animationStopped := false

	for stream.Next() {
		event := stream.Current()
		err := message.Accumulate(event)
		if err != nil {
			animation.Stop()
			return nil, fmt.Errorf("failed to accumulate stream event: %w", err)
		}

		// Process streaming events
		switch eventVariant := event.AsAny().(type) {
		case anthropic.ContentBlockDeltaEvent:
			switch deltaVariant := eventVariant.Delta.AsAny().(type) {
			case anthropic.TextDelta:
				if !hasStartedTextOutput {
					if !animationStopped {
						animation.Stop()
						animationStopped = true
					}
					fmt.Print("\u001b[93mClaude\u001b[0m: ")
					hasStartedTextOutput = true
				}
				print(deltaVariant.Text)
			}
		case anthropic.ContentBlockStartEvent:
			if block, ok := eventVariant.ContentBlock.AsAny().(anthropic.ToolUseBlock); ok {
				if !animationStopped {
					animation.Stop()
					animationStopped = true
				}
				if hasStartedTextOutput {
					fmt.Println()
				}
				inputJSON, _ := json.Marshal(block.Input)
				fmt.Printf("\u001b[92m[Tool: %s]\u001b[0m: %s\n", block.Name, string(inputJSON))
				hasStartedTextOutput = false
			}
		}
	}

	if !animationStopped {
		animation.Stop()
	}

	if stream.Err() != nil {
		return nil, fmt.Errorf("streaming error: %w", stream.Err())
	}

	if hasStartedTextOutput {
		fmt.Println()
	}

	return &message, nil
}

// estimateConversationTokens provides a client-side approximation of token count
func (a *RefactoredAgent) estimateConversationTokens(conversation []anthropic.MessageParam) int {
	if len(conversation) == 0 {
		return 0
	}

	totalChars := 0
	totalChars += len(a.systemPrompt) // System prompt

	// Estimate tokens for messages - simplified approach
	for _, msg := range conversation {
		// Use JSON encoding to get approximate size
		msgBytes, _ := json.Marshal(msg)
		totalChars += len(msgBytes)
	}

	// Add estimated overhead for tools and structure (rough approximation)
	toolDefs := a.toolRegistry.All()
	toolOverhead := len(toolDefs) * 200 // ~200 chars per tool definition
	totalChars += toolOverhead

	// Rough conversion: ~4 characters per token (conservative estimate)
	return totalChars / 4
}

// countConversationTokensAccurate gets precise token count via API (used sparingly)
func (a *RefactoredAgent) countConversationTokensAccurate(ctx context.Context, conversation []anthropic.MessageParam) (int, error) {
	if len(conversation) == 0 {
		return 0, nil
	}

	// Convert tools to the format needed for token counting
	toolDefs := a.toolRegistry.All()
	toolParams := make([]anthropic.MessageCountTokensToolUnionParam, len(toolDefs))
	for i, tool := range toolDefs {
		toolParams[i] = anthropic.MessageCountTokensToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        tool.Name,
				Description: anthropic.String(tool.Description),
				InputSchema: tool.InputSchema,
			},
		}
	}

	// Count tokens for the conversation
	tokenCount, err := a.client.Messages.CountTokens(ctx, anthropic.MessageCountTokensParams{
		Model:    anthropic.ModelClaude3_7SonnetLatest,
		Messages: conversation,
		Tools:    toolParams,
	})
	if err != nil {
		return 0, fmt.Errorf("failed to count tokens: %w", err)
	}

	return int(tokenCount.InputTokens), nil
}

// countConversationTokens provides intelligent token counting - uses estimation for quick checks,
// accurate API counting only when needed
func (a *RefactoredAgent) countConversationTokens(ctx context.Context, conversation []anthropic.MessageParam) (int, error) {
	// Use fast estimation first
	estimated := a.estimateConversationTokens(conversation)

	// If we're well under the limit, use estimation to save API calls
	if estimated < a.config.MaxInputTokens()*3/4 { // 75% threshold
		return estimated, nil
	}

	// If we're close to the limit, use accurate counting
	return a.countConversationTokensAccurate(ctx, conversation)
}

// summarizeConversation creates a summary of older messages in the conversation
func (a *RefactoredAgent) summarizeConversation(ctx context.Context, messagesToSummarize []anthropic.MessageParam) (*anthropic.MessageParam, error) {
	if len(messagesToSummarize) == 0 {
		return nil, fmt.Errorf("no messages to summarize")
	}

	// Create a prompt to summarize the conversation
	summaryPrompt := "Please provide a concise summary of this conversation, preserving key context, decisions made, and important information that might be relevant for future interactions. Focus on factual content and avoid redundant details."

	// Add the messages to summarize as context
	summaryMessages := []anthropic.MessageParam{
		anthropic.NewUserMessage(anthropic.NewTextBlock(summaryPrompt)),
	}
	summaryMessages = append(summaryMessages, messagesToSummarize...)
	summaryMessages = append(summaryMessages, anthropic.NewUserMessage(anthropic.NewTextBlock("Now provide the summary:")))

	// Get the summary from Claude
	message, err := a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaude3_7SonnetLatest,
		MaxTokens: int64(config.SummaryTokenTarget),
		Messages:  summaryMessages,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to generate summary: %w", err)
	}

	// Extract the text content from the response
	var summaryText strings.Builder
	for _, content := range message.Content {
		if textBlock, ok := content.AsAny().(anthropic.TextBlock); ok {
			summaryText.WriteString(textBlock.Text)
		}
	}

	// Create a system-like message with the summary
	summaryMessage := anthropic.NewUserMessage(
		anthropic.NewTextBlock(fmt.Sprintf("[CONVERSATION SUMMARY] %s", summaryText.String())),
	)

	return &summaryMessage, nil
}

// manageConversationLength ensures the conversation stays within token limits
func (a *RefactoredAgent) manageConversationLength(ctx context.Context, conversation []anthropic.MessageParam) ([]anthropic.MessageParam, error) {
	tokenCount, err := a.countConversationTokens(ctx, conversation)
	if err != nil {
		// If we can't count tokens, fall back to message count limit
		log.Printf("Warning: couldn't count tokens, falling back to message limit: %v", err)
		if len(conversation) > a.config.RecentMessagesKeep()*2 { // *2 because we might have tool use messages
			return conversation[len(conversation)-a.config.RecentMessagesKeep():], nil
		}
		return conversation, nil
	}

	// If we're under the limit, no need to manage
	if tokenCount < a.config.MaxInputTokens() {
		return conversation, nil
	}

	fmt.Printf("\u001b[95m[Token Management]\u001b[0m: Conversation has %d tokens, managing length...\n", tokenCount)

	// Keep the most recent messages
	if len(conversation) <= a.config.RecentMessagesKeep() {
		// If we have very few messages but still over limit, something's wrong
		return conversation, nil
	}

	// Split conversation: messages to summarize vs recent messages to keep
	splitPoint := len(conversation) - a.config.RecentMessagesKeep()
	messagesToSummarize := conversation[:splitPoint]
	recentMessages := conversation[splitPoint:]

	// Create summary of older messages
	summaryMessage, err := a.summarizeConversation(ctx, messagesToSummarize)
	if err != nil {
		log.Printf("Warning: failed to create summary, truncating instead: %v", err)
		// Fall back to simple truncation
		return recentMessages, nil
	}

	// Combine summary with recent messages
	managedConversation := []anthropic.MessageParam{*summaryMessage}
	managedConversation = append(managedConversation, recentMessages...)

	// Verify we're now under the limit
	newTokenCount, err := a.countConversationTokens(ctx, managedConversation)
	if err == nil {
		fmt.Printf("\u001b[95m[Token Management]\u001b[0m: Reduced from %d to %d tokens.\n", tokenCount, newTokenCount)
	}

	return managedConversation, nil
}

// Helper functions (kept from original)
func promptForDirectory(scanner *bufio.Scanner) (string, error) {
	fmt.Print("Enter the directory you'd like to work in (or press Enter for current directory): ")
	if !scanner.Scan() {
		return "", fmt.Errorf("failed to read input")
	}

	input := strings.TrimSpace(scanner.Text())
	if input == "" {
		cwd, err := os.Getwd()
		if err != nil {
			return "", fmt.Errorf("failed to get current directory: %w", err)
		}
		return cwd, nil
	}

	if strings.HasPrefix(input, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("failed to get home directory: %w", err)
		}
		input = filepath.Join(home, input[2:])
	}

	dir := filepath.Clean(input)
	if err := validateDirectory(dir); err != nil {
		return "", err
	}

	return dir, nil
}

func validateDirectory(dir string) error {
	info, err := os.Stat(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("directory does not exist: %s", dir)
		}
		return fmt.Errorf("cannot access directory: %w", err)
	}

	if !info.IsDir() {
		return fmt.Errorf("path is not a directory: %s", dir)
	}

	return nil
}

func loadSystemPrompt() string {
	content, err := os.ReadFile("system_prompt.txt")
	if err != nil {
		log.Printf("Warning: Could not load system_prompt.txt: %v. Using default prompt.", err)
		return "You are GooCode, a helpful AI coding assistant with access to file operations within the working directory."
	}
	return string(content)
}
