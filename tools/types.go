package tools

import (
	"context"
	"encoding/json"

	"github.com/anthropics/anthropic-sdk-go"
)

// Tool defines the interface that all tools must implement
type Tool interface {
	Name() string
	Description() string
	InputSchema() anthropic.ToolInputSchemaParam
	Execute(ctx context.Context, agent ToolContext, input json.RawMessage) (string, error)
}

// ToolContext provides the interface for tools to interact with the agent
// This eliminates the need for global variables and enables proper dependency injection
type ToolContext interface {
	WorkingDir() string
	ResolveFilePath(relativePath string) (string, error)
}

// ToolDefinition represents a complete tool definition for registration
type ToolDefinition struct {
	Name        string
	Description string
	InputSchema anthropic.ToolInputSchemaParam
	Function    func(json.RawMessage) (string, error)
}

// Registry manages all available tools
type Registry struct {
	tools map[string]Tool
}

// NewRegistry creates a new tool registry
func NewRegistry() *Registry {
	return &Registry{
		tools: make(map[string]Tool),
	}
}

// Register adds a tool to the registry
func (r *Registry) Register(tool Tool) {
	r.tools[tool.Name()] = tool
}

// Get retrieves a tool by name
func (r *Registry) Get(name string) (Tool, bool) {
	tool, exists := r.tools[name]
	return tool, exists
}

// All returns all registered tools as ToolDefinitions for the Anthropic SDK
func (r *Registry) All() []ToolDefinition {
	var definitions []ToolDefinition
	for _, tool := range r.tools {
		definitions = append(definitions, ToolDefinition{
			Name:        tool.Name(),
			Description: tool.Description(),
			InputSchema: tool.InputSchema(),
			Function:    nil, // Will be populated during execution
		})
	}
	return definitions
}

// Execute runs a tool with the given input
func (r *Registry) Execute(ctx context.Context, agent ToolContext, toolName string, input json.RawMessage) (string, error) {
	tool, exists := r.tools[toolName]
	if !exists {
		return "", &ToolNotFoundError{Name: toolName}
	}

	return tool.Execute(ctx, agent, input)
}

// ToolNotFoundError is returned when a requested tool doesn't exist
type ToolNotFoundError struct {
	Name string
}

func (e *ToolNotFoundError) Error() string {
	return "tool " + e.Name + " not found"
}
