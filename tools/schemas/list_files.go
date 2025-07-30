package schemas

import (
	"anthropic-chat/utils"
)

// ListFilesInput represents the input schema for the list_files tool
type ListFilesInput struct {
	Path string `json:"path,omitempty" jsonschema_description:"Optional relative path (defaults to current directory)."`
}

// ListFilesInputSchema is the cached schema for ListFilesInput
var ListFilesInputSchema = utils.GenerateSchema[ListFilesInput]()
