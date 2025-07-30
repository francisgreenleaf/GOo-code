package file

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"anthropic-chat/tools"
	"anthropic-chat/tools/schemas"

	"github.com/anthropics/anthropic-sdk-go"
)

// ListFilesTool implements the list_files tool
type ListFilesTool struct{}

// NewListFilesTool creates a new ListFiles tool instance
func NewListFilesTool() *ListFilesTool {
	return &ListFilesTool{}
}

// Name returns the tool name
func (t *ListFilesTool) Name() string {
	return "list_files"
}

// Description returns the tool description
func (t *ListFilesTool) Description() string {
	return "List files and directories at specified path (defaults to current directory)."
}

// InputSchema returns the input schema for this tool
func (t *ListFilesTool) InputSchema() anthropic.ToolInputSchemaParam {
	return schemas.ListFilesInputSchema
}

// Execute performs the list files operation
func (t *ListFilesTool) Execute(ctx context.Context, agent tools.ToolContext, input json.RawMessage) (string, error) {
	var listInput schemas.ListFilesInput
	if err := json.Unmarshal(input, &listInput); err != nil {
		return "", fmt.Errorf("failed to parse input: %w", err)
	}

	// Determine directory to list
	dir := agent.WorkingDir()
	if listInput.Path != "" {
		var err error
		dir, err = agent.ResolveFilePath(listInput.Path)
		if err != nil {
			return "", err
		}
	}

	// Collect files and directories
	files := []string{}
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}

		// Skip the current directory entry
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
		return "", fmt.Errorf("failed to list files in %s: %w", listInput.Path, err)
	}

	// Convert to JSON for output
	result, err := json.Marshal(files)
	if err != nil {
		return "", fmt.Errorf("failed to marshal file list: %w", err)
	}

	return string(result), nil
}
