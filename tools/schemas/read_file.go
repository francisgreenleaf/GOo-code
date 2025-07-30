package schemas

import (
	"anthropic-chat/utils"
)

// ReadFileInput represents the input schema for the read_file tool
type ReadFileInput struct {
	Path string `json:"path" jsonschema_description:"Relative file path in working directory."`
}

// ReadFileInputSchema is the cached schema for ReadFileInput
var ReadFileInputSchema = utils.GenerateSchema[ReadFileInput]()
