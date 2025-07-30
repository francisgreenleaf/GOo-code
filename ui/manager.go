package ui

import (
	"fmt"
	"sync"
	"time"
)

// Manager handles UI-related functionality
type Manager struct{}

// NewManager creates a new UI manager
func NewManager() *Manager {
	return &Manager{}
}

// ShowWelcome displays the welcome message
func (m *Manager) ShowWelcome() {
	fmt.Println(`  ▄████  ▒█████   ▒█████   ▄████▄   ▒█████  ▓█████▄ ▓█████ 
 ██▒ ▀█▒▒██▒  ██▒▒██▒  ██▒▒██▀ ▀█  ▒██▒  ██▒▒██▀ ██▌▓█   ▀ 
▒██░▄▄▄░▒██░  ██▒▒██░  ██▒▒▓█    ▄ ▒██░  ██▒░██   █▌▒███   
░▓█  ██▓▒██   ██░▒██   ██░▒▓▓▄ ▄██▒▒██   ██░░▓█▄   ▌▒▓█  ▄ 
░▒▓███▀▒░ ████▓▒░░ ████▓▒░▒ ▓███▀ ░░ ████▓▒░░▒████▓ ░▒████▒
 ░▒   ▒ ░ ▒░▒░▒░ ░ ▒░▒░▒░ ░ ░▒ ▒  ░░ ▒░▒░▒░  ▒▒▓  ▒ ░░ ▒░ ░
  ░   ░   ░ ▒ ▒░   ░ ▒ ▒░   ░  ▒     ░ ▒ ▒░  ░ ▒  ▒  ░ ░  ░
░ ░   ░ ░ ░ ░ ▒  ░ ░ ░ ▒  ░        ░ ░ ░ ▒   ░ ░  ░    ░   
      ░     ░ ░      ░ ░  ░ ░          ░ ░     ░       ░  ░
                          ░                  ░             
GOOCODE is an agent built by Francis Greenleaf 
(gh: francisgreenleaf X: @inferencetoken) and Cline (cline.bot)
Francis built this while working at Cline. It's a side project. 
It can perform basic agentic tasks in your directory. Use at your own risk.`)
}

// ShowCommands displays available commands
func (m *Manager) ShowCommands() {
	fmt.Println("BASIC COMMANDS:")
	fmt.Println("Chat with GooCode (use 'ctrl-c' to quit)")
	fmt.Printf("Type '/cd' to change working directory\n")
	fmt.Printf("Type '/tokens' to see current token count\n\n")
}

// ThinkingAnimation handles the "thinking..." animation
type ThinkingAnimation struct {
	stopChan chan bool
	wg       sync.WaitGroup
	running  bool
	mu       sync.Mutex
}

// NewThinkingAnimation creates a new thinking animation
func (m *Manager) NewThinkingAnimation() *ThinkingAnimation {
	return &ThinkingAnimation{
		stopChan: make(chan bool),
	}
}

// Start begins the thinking animation
func (ta *ThinkingAnimation) Start() {
	ta.mu.Lock()
	defer ta.mu.Unlock()

	if ta.running {
		return
	}

	ta.running = true
	ta.wg.Add(1)

	go func() {
		defer ta.wg.Done()

		dots := 1
		fmt.Print("\u001b[93mthinking\u001b[0m.")

		ticker := time.NewTicker(500 * time.Millisecond)
		defer ticker.Stop()

		for {
			select {
			case <-ta.stopChan:
				return
			case <-ticker.C:
				if dots < 3 {
					dots++
					fmt.Print(".")
				} else {
					fmt.Print("\r\u001b[93mthinking\u001b[0m.")
					dots = 1
				}
			}
		}
	}()
}

// Stop ends the thinking animation
func (ta *ThinkingAnimation) Stop() {
	ta.mu.Lock()
	defer ta.mu.Unlock()

	if !ta.running {
		return
	}

	ta.running = false
	close(ta.stopChan)
	ta.wg.Wait()

	fmt.Print("\r\033[K")
}
