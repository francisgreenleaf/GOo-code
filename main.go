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

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/invopop/jsonschema"
	"github.com/joho/godotenv"
)

// Constants for conversation management
const (
	MaxTokensLimit     = 190000 // Leave buffer below 200K limit
	RecentMessagesKeep = 6      // Keep last 3 exchanges (6 messages)
	SummaryTokenTarget = 2000   // Target token count for summary
)

// Global variable for current agent (needed for tool functions)
var currentAgent *Agent

// loadSystemPrompt loads the system prompt from file with fallback to default
func loadSystemPrompt() string {
	content, err := os.ReadFile("system_prompt.txt")
	if err != nil {
		log.Printf("Warning: Could not load system_prompt.txt: %v. Using default prompt.", err)
		return "You are GooCode, a helpful AI coding assistant with access to file operations within the working directory. You have three tools: read_file, list_files, and edit_file."
	}
	return string(content)
}

// Helper functions for directory management
func promptForDirectory(scanner *bufio.Scanner) (string, error) {
	fmt.Print("Enter the directory you'd like to work in (or press Enter for current directory): ")
	if !scanner.Scan() {
		return "", fmt.Errorf("failed to read input")
	}

	input := strings.TrimSpace(scanner.Text())
	if input == "" {
		// Default to current directory
		cwd, err := os.Getwd()
		if err != nil {
			return "", fmt.Errorf("failed to get current directory: %w", err)
		}
		return cwd, nil
	}

	// Expand ~ to home directory
	if strings.HasPrefix(input, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("failed to get home directory: %w", err)
		}
		input = filepath.Join(home, input[2:])
	}

	// Clean and validate the path
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

func (a *Agent) setWorkingDirectory(dir string) error {
	if err := validateDirectory(dir); err != nil {
		return err
	}
	a.workingDir = dir
	return nil
}

func (a *Agent) resolveFilePath(relativePath string) (string, error) {
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

func main() {
	// Load environment variables from .env file (if it exists)
	err := godotenv.Load()
	if err != nil {
		// .env file not found or couldn't be loaded - that's okay, we'll try system env vars
		log.Printf("Warning: .env file not found or couldn't be loaded: %v", err)
	}

	// Get API key from environment variable
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		log.Fatal("ANTHROPIC_API_KEY environment variable is required")
	}

	client := anthropic.NewClient(
		option.WithAPIKey(apiKey),
	)

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

	tools := []ToolDefinition{ReadFileDefinition, ListFilesDefinition, EditFileDefinition}
	agent := NewAgent(&client, getUserMessage, tools, workingDir)
	currentAgent = agent // Set global agent for tool functions
	err = agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

func NewAgent(
	client *anthropic.Client,
	getUserMessage func() (string, bool),
	tools []ToolDefinition,
	workingDir string) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
		tools:          tools,
		workingDir:     workingDir,
		systemPrompt:   loadSystemPrompt(),
	}
}

type Agent struct {
	client         *anthropic.Client
	getUserMessage func() (string, bool)
	tools          []ToolDefinition
	workingDir     string
	systemPrompt   string
}

// countConversationTokens counts the total tokens in the current conversation
func (a *Agent) countConversationTokens(ctx context.Context, conversation []anthropic.MessageParam) (int, error) {
	if len(conversation) == 0 {
		return 0, nil
	}

	// Convert tools to the format needed for token counting
	toolParams := make([]anthropic.MessageCountTokensToolUnionParam, len(a.tools))
	for i, tool := range a.tools {
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

// summarizeConversation creates a summary of older messages in the conversation
func (a *Agent) summarizeConversation(ctx context.Context, messagesToSummarize []anthropic.MessageParam) (*anthropic.MessageParam, error) {
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
		MaxTokens: int64(SummaryTokenTarget),
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
func (a *Agent) manageConversationLength(ctx context.Context, conversation []anthropic.MessageParam) ([]anthropic.MessageParam, error) {
	tokenCount, err := a.countConversationTokens(ctx, conversation)
	if err != nil {
		// If we can't count tokens, fall back to message count limit
		log.Printf("Warning: couldn't count tokens, falling back to message limit: %v", err)
		if len(conversation) > RecentMessagesKeep*2 { // *2 because we might have tool use messages
			return conversation[len(conversation)-RecentMessagesKeep:], nil
		}
		return conversation, nil
	}

	// If we're under the limit, no need to manage
	if tokenCount < MaxTokensLimit {
		return conversation, nil
	}

	fmt.Printf("\u001b[95m[Token Management]\u001b[0m: Conversation has %d tokens, managing length...\n", tokenCount)

	// Keep the most recent messages
	if len(conversation) <= RecentMessagesKeep {
		// If we have very few messages but still over limit, something's wrong
		return conversation, nil
	}

	// Split conversation: messages to summarize vs recent messages to keep
	splitPoint := len(conversation) - RecentMessagesKeep
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

func (a *Agent) Run(ctx context.Context) error {
	conversation := []anthropic.MessageParam{}

	fmt.Println("Chat with GooCode (use 'ctrl-c' to quit)")
	fmt.Printf("Type '/cd' to change working directory\n")
	fmt.Printf("Type '/tokens' to see current token count\n\n")

	for {
		fmt.Print("\u001b[94mYou\u001b[0m: ")
		userInput, ok := a.getUserMessage()
		if !ok {
			break
		}

		// Handle slash commands
		if strings.HasPrefix(userInput, "/cd") {
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
							continue
						}
						newDir = filepath.Join(home, newDir[2:])
					}

					// Clean and validate the path
					newDir = filepath.Clean(newDir)
					if err := a.setWorkingDirectory(newDir); err != nil {
						fmt.Printf("\u001b[91mError\u001b[0m: %v\n\n", err)
					} else {
						fmt.Printf("\u001b[92mWorking directory changed to:\u001b[0m %s\n\n", newDir)
					}
				}
			}
			continue
		}

		if strings.HasPrefix(userInput, "/tokens") {
			if len(conversation) == 0 {
				fmt.Printf("\u001b[96mToken Info\u001b[0m: No conversation yet (0 tokens)\n\n")
			} else {
				tokenCount, err := a.countConversationTokens(ctx, conversation)
				if err != nil {
					fmt.Printf("\u001b[91mError\u001b[0m: Failed to count tokens: %v\n\n", err)
				} else {
					percentage := float64(tokenCount) / float64(MaxTokensLimit) * 100
					fmt.Printf("\u001b[96mToken Info\u001b[0m: Current conversation has %d tokens (%.1f%% of %d limit)\n", tokenCount, percentage, MaxTokensLimit)
					fmt.Printf("\u001b[96mToken Info\u001b[0m: %d messages in conversation\n\n", len(conversation))
				}
			}
			continue
		}

		userMessage := anthropic.NewUserMessage(anthropic.NewTextBlock(userInput))
		conversation = append(conversation, userMessage)

		// Manage conversation length before starting tool execution loop
		managedConversation, err := a.manageConversationLength(ctx, conversation)
		if err != nil {
			log.Printf("Warning: failed to manage conversation length: %v", err)
		} else {
			conversation = managedConversation
		}

		for {
			message, err := a.runInference(ctx, conversation)
			if err != nil {
				return err
			}
			conversation = append(conversation, message.ToParam())

			toolResults := []anthropic.ContentBlockParamUnion{}
			hasToolUse := false

			for _, content := range message.Content {
				switch block := content.AsAny().(type) {
				case anthropic.TextBlock:
					fmt.Printf("\u001b[93mClaude\u001b[0m: %s\n", block.Text)
				case anthropic.ToolUseBlock:
					hasToolUse = true
					inputJSON, _ := json.Marshal(block.Input)
					fmt.Printf("\u001b[92m[Tool: %s]\u001b[0m: %s\n", block.Name, string(inputJSON))

					// Execute the tool
					result, err := a.executeTool(block.Name, block.Input)
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

func (a *Agent) runInference(ctx context.Context, conversation []anthropic.MessageParam) (*anthropic.Message, error) {
	// Convert tools to Anthropic SDK format
	toolParams := make([]anthropic.ToolParam, len(a.tools))
	for i, tool := range a.tools {
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

	message, err := a.client.Messages.New(ctx, anthropic.MessageNewParams{
		Model:     anthropic.ModelClaude3_7SonnetLatest,
		MaxTokens: int64(1024),
		System: []anthropic.TextBlockParam{
			{Text: a.systemPrompt},
		},
		Messages: conversation,
		Tools:    tools,
	})
	return message, err
}

func (a *Agent) executeTool(toolName string, input json.RawMessage) (string, error) {
	for _, tool := range a.tools {
		if tool.Name == toolName {
			return tool.Function(input)
		}
	}
	return "", fmt.Errorf("tool %s not found", toolName)
}

type ToolDefinition struct {
	Name        string
	Description string
	InputSchema anthropic.ToolInputSchemaParam
	Function    func(json.RawMessage) (string, error)
}

var ReadFileDefinition = ToolDefinition{
	Name:        "read_file",
	Description: "Read the contents of a given relative file path. Use this when you want to see what's inside a file. Do not use this with directory names.",
	InputSchema: ReadFileInputSchema,
	Function:    func(input json.RawMessage) (string, error) { return currentAgent.ReadFile(input) },
}

type ReadFileInput struct {
	Path string `json:"path" jsonschema_description:"The relative path of a file in the working directory."`
}

var ReadFileInputSchema = GenerateSchema[ReadFileInput]()

func (a *Agent) ReadFile(input json.RawMessage) (string, error) {
	readFileInput := ReadFileInput{}
	err := json.Unmarshal(input, &readFileInput)
	if err != nil {
		return "", fmt.Errorf("failed to parse input: %w", err)
	}

	fullPath, err := a.resolveFilePath(readFileInput.Path)
	if err != nil {
		return "", err
	}

	content, err := os.ReadFile(fullPath)
	if err != nil {
		return "", err
	}
	return string(content), nil
}

var ListFilesDefinition = ToolDefinition{
	Name:        "list_files",
	Description: "List files and directories at a given path. If no path is provided, lists files in the current directory.",
	InputSchema: ListFilesInputSchema,
	Function:    func(input json.RawMessage) (string, error) { return currentAgent.ListFiles(input) },
}

type ListFilesInput struct {
	Path string `json:"path,omitempty" jsonschema_description:"Optional relative path to list files from. Defaults to current directory if not provided."`
}

var ListFilesInputSchema = GenerateSchema[ListFilesInput]()

func (a *Agent) ListFiles(input json.RawMessage) (string, error) {
	listFilesInput := ListFilesInput{}
	err := json.Unmarshal(input, &listFilesInput)
	if err != nil {
		return "", fmt.Errorf("failed to parse input: %w", err)
	}

	dir := a.workingDir
	if listFilesInput.Path != "" {
		dir, err = a.resolveFilePath(listFilesInput.Path)
		if err != nil {
			return "", err
		}
	}

	var files []string
	err = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}

		if relPath != "." {
			if info.IsDir() {
				files = append(files, relPath+"/")
			} else {
				files = append(files, relPath)
			}
		}
		return nil
	})

	if err != nil {
		return "", err
	}

	result, err := json.Marshal(files)
	if err != nil {
		return "", err
	}

	return string(result), nil
}

var EditFileDefinition = ToolDefinition{
	Name:        "edit_file",
	Description: "Make edits to a file. The input should be a JSON object with the file path and the changes to apply. If the file specified does not exist, it will be created. Returns the updated file contents for verification.",
	InputSchema: EditFileInputSchema,
	Function:    func(input json.RawMessage) (string, error) { return currentAgent.EditFile(input) },
}

type EditFileInput struct {
	Path    string `json:"path" jsonschema_description:"The relative path of the file to edit."`
	Changes string `json:"changes" jsonschema_description:"The changes to apply to the file. This should be a string containing the changes to apply, such as 'add line 1', 'remove line 2', etc."`
}

var EditFileInputSchema = GenerateSchema[EditFileInput]()

func (a *Agent) EditFile(input json.RawMessage) (string, error) {
	editFileInput := EditFileInput{}
	err := json.Unmarshal(input, &editFileInput)
	if err != nil {
		return "", fmt.Errorf("failed to parse input: %w", err)
	}

	if editFileInput.Path == "" {
		return "", fmt.Errorf("path is required")
	}

	fullPath, err := a.resolveFilePath(editFileInput.Path)
	if err != nil {
		return "", err
	}

	// For simplicity, we'll just append the changes to the file
	file, err := os.OpenFile(fullPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		return "", fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	if _, err := file.WriteString(editFileInput.Changes + "\n"); err != nil {
		return "", fmt.Errorf("failed to write changes to file: %w", err)
	}

	// Read back the file contents for verification
	updatedContent, err := os.ReadFile(fullPath)
	if err != nil {
		return "", fmt.Errorf("file edited but failed to read back contents: %w", err)
	}

	// Return success message with file contents
	return fmt.Sprintf("File edited successfully. Updated contents:\n\n%s", string(updatedContent)), nil
}

func createNewfile(filePath, content string) (string, error) {
	dir := filepath.Dir(filePath)
	if dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return "", fmt.Errorf("failed to create directory: %w", err)
		}
	}
	if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
		return "", fmt.Errorf("failed to write file: %w", err)
	}
	return fmt.Sprintf("File created at %s", filePath), nil
}

func GenerateSchema[T any]() anthropic.ToolInputSchemaParam {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T

	schema := reflector.Reflect(v)

	return anthropic.ToolInputSchemaParam{
		Properties: schema.Properties,
	}
}
