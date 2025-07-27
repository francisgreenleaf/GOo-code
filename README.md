# Anthropic Chat Agent

A Go-based chat agent using the Anthropic Claude API with advanced tool calling capabilities and conversation management.

## Setup

### 1. Install Dependencies

```bash
go mod tidy
```

### 2. Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and replace `your_anthropic_api_key_here` with your actual Anthropic API key:

```bash
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
```

### 3. Run the Application

```bash
go run main.go
```

## Features

- Interactive chat with Claude 3.5 Sonnet
- Multiple tool capabilities:
  - **read_file**: Read contents of files within the working directory
  - **list_files**: List files and directories within the working directory
  - **edit_file**: Create new files or append content to existing files
- Working directory selection and management
- Advanced conversation management:
  - Token counting and monitoring
  - Automatic conversation summarization when approaching token limits
  - Conversation length management to stay within API limits
- Slash commands for enhanced interaction
- Security features with path traversal protection
- Environment-based configuration (API key not hardcoded)
- Proper .gitignore to prevent API key exposure

## Security

- The `.env` file containing your API key is gitignored and will not be committed to version control
- The application will fail gracefully if no API key is provided
- API keys can be set via environment variables or the `.env` file
- File operations are restricted to the selected working directory
- Path traversal attacks are prevented (no `..` paths allowed)
- All file paths are validated and sanitized

## Environment Variables

- `ANTHROPIC_API_KEY`: Your Anthropic API key (required)

## Usage

### Starting the Application

When you run the application, you'll be prompted to select a working directory:
- Press Enter to use the current directory
- Enter a path (supports `~/` for home directory) to use a different directory

### Interactive Chat

Once running, you can:
- Chat with Claude naturally - it has access to file tools within your working directory
- Use slash commands for additional functionality
- Type your messages and press Enter
- Use Ctrl+C to quit

### Slash Commands

- `/cd` - Change the working directory during the session
- `/tokens` - View current conversation token count and usage statistics

### Tool Capabilities

The agent can:
- **Read files**: View contents of any file in the working directory
- **List directories**: Browse the file structure within the working directory  
- **Edit files**: Create new files or append content to existing files
- All file operations are sandboxed to the selected working directory for security

### Conversation Management

The application automatically manages long conversations:
- Monitors token usage (190K token limit with buffer)
- Creates summaries of older messages when approaching limits
- Preserves recent context while maintaining conversation flow
- Shows token usage statistics with the `/tokens` command

## Technical Details

- Uses Claude 3.5 Sonnet Latest model
- Implements automatic conversation summarization
- Token counting and management
- Secure file system operations with path validation
- JSON schema validation for tool inputs
