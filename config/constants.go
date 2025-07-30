package config

import "regexp"

// Constants for conversation management
const (
	MaxOutputTokens    = 10000  // Claude response generation limit
	MaxInputTokens     = 200000 // Full context window for conversation
	WarningThreshold   = 190000 // Show warning at this input token count
	RecentMessagesKeep = 6      // Keep last 3 exchanges (6 messages)
	SummaryTokenTarget = 2000   // Target token count for summary
)

// Safety constants for command execution
var DangerousCommands = []string{
	"rm", "rmdir", "del", "erase",
	"format", "fdisk", "mkfs",
	"dd", "shred", "wipe",
	"chmod 000", "chown root",
	"kill -9", "killall",
	"shutdown", "reboot", "halt",
}

var DangerousPatterns = []*regexp.Regexp{
	regexp.MustCompile(`rm\s+.*-r.*f`),         // rm -rf
	regexp.MustCompile(`rm\s+.*-f.*r`),         // rm -fr
	regexp.MustCompile(`>\s*/dev/(null|zero)`), // redirect to /dev/null
	regexp.MustCompile(`rm\s+.*\*`),            // rm with wildcards
	regexp.MustCompile(`find.*-delete`),        // find with delete
	regexp.MustCompile(`git\s+reset\s+--hard`), // destructive git operations
}
