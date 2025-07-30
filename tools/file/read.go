package file

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"anthropic-chat/tools"
	"anthropic-chat/tools/schemas"

	"github.com/anthropics/anthropic-sdk-go"
)

// ReadFileTool implements the read_file tool
type ReadFileTool struct{}

// NewReadFileTool creates a new ReadFile tool instance
func NewReadFileTool() *ReadFileTool {
	return &ReadFileTool{}
}

// Name returns the tool name
func (t *ReadFileTool) Name() string {
	return "read_file"
}

// Description returns the tool description
func (t *ReadFileTool) Description() string {
	return "Read file contents from relative path within working directory."
}

// InputSchema returns the input schema for this tool
func (t *ReadFileTool) InputSchema() anthropic.ToolInputSchemaParam {
	return schemas.ReadFileInputSchema
}

// Execute performs the read file operation
func (t *ReadFileTool) Execute(ctx context.Context, agent tools.ToolContext, input json.RawMessage) (string, error) {
	var readInput schemas.ReadFileInput
	if err := json.Unmarshal(input, &readInput); err != nil {
		return "", fmt.Errorf("failed to parse input: %w", err)
	}

	// Resolve the file path using the agent's security validation
	fullPath, err := agent.ResolveFilePath(readInput.Path)
	if err != nil {
		return "", err
	}

	// Read the file content
	content, err := os.ReadFile(fullPath)
	if err != nil {
		return "", fmt.Errorf("failed to read file %s: %w", readInput.Path, err)
	}

	return string(content), nil
}
