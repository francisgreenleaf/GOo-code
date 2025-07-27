## Brief overview
Guidelines for ensuring SDK and library method compatibility when working with Go projects in this workspace. These rules emphasize using the context7 MCP server to reference current documentation for Anthropic SDK and jsonschema libraries to prevent compatibility issues.

## SDK Documentation Reference
- Always use the context7 MCP server to fetch current documentation for Anthropic SDK and jsonschema libraries before implementing or modifying related code
- Verify method signatures, parameter requirements, and return types against the latest documentation
- Check for deprecated methods or breaking changes in the library versions being used

## Development workflow
- Before making changes to Anthropic API calls or jsonschema operations, use `resolve-library-id` to get the correct library ID
- Use `get-library-docs` with the resolved library ID to fetch current documentation for the specific methods being implemented
- Cross-reference the documentation with existing code to identify potential compatibility issues
- Apply the same verification process for both Anthropic SDK and jsonschema library usage

## Coding best practices
- Ensure all Anthropic SDK and jsonschema method calls match the current API specification
- Validate that parameter names, types, and requirements align with the documented interface
- Handle SDK and library responses according to the current documented structure and error patterns
- Verify jsonschema validation patterns and methods against current library documentation

## Project context
- This workspace contains a Go project that integrates with the Anthropic SDK and jsonschema library
- Compatibility issues can arise from SDK/library updates or incorrect method usage
- Using context7 MCP server provides access to up-to-date documentation to prevent these issues for both libraries
