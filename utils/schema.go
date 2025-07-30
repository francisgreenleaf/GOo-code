package utils

import (
	"reflect"
	"sync"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/invopop/jsonschema"
)

// SchemaGenerator handles JSON schema generation from Go types with caching
type SchemaGenerator struct {
	reflector *jsonschema.Reflector
	cache     map[reflect.Type]anthropic.ToolInputSchemaParam
	mu        sync.RWMutex
}

// NewSchemaGenerator creates a new schema generator with optimized settings
func NewSchemaGenerator() *SchemaGenerator {
	reflector := &jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}

	return &SchemaGenerator{
		reflector: reflector,
		cache:     make(map[reflect.Type]anthropic.ToolInputSchemaParam),
	}
}

// GenerateSchemaForType generates a JSON schema for the given value
func (sg *SchemaGenerator) GenerateSchemaForType(v interface{}) anthropic.ToolInputSchemaParam {
	t := reflect.TypeOf(v)

	// Check cache first
	sg.mu.RLock()
	if schema, exists := sg.cache[t]; exists {
		sg.mu.RUnlock()
		return schema
	}
	sg.mu.RUnlock()

	// Generate schema
	schema := sg.reflector.Reflect(v)

	// Convert to Anthropic format
	toolSchema := anthropic.ToolInputSchemaParam{
		Properties: schema.Properties,
	}

	// Cache the result
	sg.mu.Lock()
	sg.cache[t] = toolSchema
	sg.mu.Unlock()

	return toolSchema
}

// Default global schema generator instance
var defaultGenerator = NewSchemaGenerator()

// GenerateSchema is a convenience function that uses the default generator
func GenerateSchema[T any]() anthropic.ToolInputSchemaParam {
	var v T
	return defaultGenerator.GenerateSchemaForType(v)
}
