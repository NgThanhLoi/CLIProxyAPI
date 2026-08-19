package helps

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// ============================================================
// Request transform: OpenAI → CommandCode /alpha/generate
// ============================================================

// GenerateCommandCodeUUID generates a random RFC 4122 v4 UUID using crypto/rand.
func GenerateCommandCodeUUID() string {
	var b [16]byte
	_, _ = rand.Read(b[:])
	b[6] = (b[6] & 0x0f) | 0x40 // RFC 4122 version 4
	b[8] = (b[8] & 0x3f) | 0x80 // RFC 4122 variant
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}

// TransformToCommandCode transforms an OpenAI Chat Completions payload into CommandCode's /alpha/generate format.
func TransformToCommandCode(model string, openaiPayload []byte) []byte {
	var body struct {
		Messages []struct {
			Role       string      `json:"role"`
			Content    interface{} `json:"content"`
			ToolCalls  interface{} `json:"tool_calls,omitempty"`
			ToolCallID string      `json:"tool_call_id,omitempty"`
			Name       string      `json:"name,omitempty"`
		} `json:"messages"`
		MaxTokens           *int        `json:"max_tokens,omitempty"`
		MaxCompletionTokens *int        `json:"max_completion_tokens,omitempty"`
		Temperature         *float64    `json:"temperature,omitempty"`
		TopP                *float64    `json:"top_p,omitempty"`
		Tools               interface{} `json:"tools,omitempty"`
		ToolChoice          interface{} `json:"tool_choice,omitempty"`
	}
	if err := json.Unmarshal(openaiPayload, &body); err != nil {
		return openaiPayload
	}

	var systemTexts []string
	var messages []map[string]interface{}
	toolNamesByCallID := make(map[string]string)

	for _, msg := range body.Messages {
		if msg.Role == "system" {
			text := flattenContent(msg.Content)
			if text != "" {
				systemTexts = append(systemTexts, text)
			}
			continue
		}

		if msg.Role == "tool" {
			callID := msg.ToolCallID
			toolName := msg.Name
			if toolName == "" {
				toolName = toolNamesByCallID[callID]
			}
			if toolName == "" {
				toolName = "tool"
			}
			contentStr := flattenContent(msg.Content)
			messages = append(messages, map[string]interface{}{
				"role":    "user",
				"content": fmt.Sprintf("[Tool Result from %s (%s)]:\n%s", toolName, callID, contentStr),
			})
			continue
		}

		var textParts []string
		if text := flattenContent(msg.Content); text != "" {
			textParts = append(textParts, text)
		}
		if msg.Role == "assistant" && msg.ToolCalls != nil {
			if tcSlice, ok := msg.ToolCalls.([]interface{}); ok {
				for _, tc := range tcSlice {
					if tcMap, ok := tc.(map[string]interface{}); ok {
						id, _ := tcMap["id"].(string)
						var name string
						var argsStr string
						if fn, ok := tcMap["function"].(map[string]interface{}); ok {
							name, _ = fn["name"].(string)
							if s, ok := fn["arguments"].(string); ok {
								argsStr = s
							} else if b, err := json.Marshal(fn["arguments"]); err == nil {
								argsStr = string(b)
							}
						}
						if id != "" && name != "" {
							toolNamesByCallID[id] = name
							textParts = append(textParts, fmt.Sprintf("[Assistant Tool Call: %s(%s), id: %s]", name, argsStr, id))
						}
					}
				}
			}
		}

		content := strings.Join(textParts, "\n\n")
		if content == "" {
			content = " "
		}

		messages = append(messages, map[string]interface{}{
			"role":    msg.Role,
			"content": content,
		})
	}

	temperature := 0.3
	if body.Temperature != nil {
		temperature = *body.Temperature
	}

	params := map[string]interface{}{
		"model":       model,
		"messages":    messages,
		"stream":      true,
		"temperature": temperature,
	}
	if body.MaxCompletionTokens != nil {
		params["max_tokens"] = *body.MaxCompletionTokens
	} else if body.MaxTokens != nil {
		params["max_tokens"] = *body.MaxTokens
	}
	if body.TopP != nil {
		params["top_p"] = *body.TopP
	}
	if ccTools := toCommandCodeTools(body.Tools); ccTools != nil {
		params["tools"] = ccTools
	}
	if ccChoice := toCommandCodeToolChoice(body.ToolChoice); ccChoice != nil {
		params["tool_choice"] = ccChoice
	}
	if system := strings.Join(systemTexts, "\n\n"); system != "" {
		params["system"] = system
	}

	ccReq := map[string]interface{}{
		"threadId": GenerateCommandCodeUUID(),
		"memory":   "",
		"config": map[string]interface{}{
			"workingDir":    "/tmp",
			"date":          time.Now().Format("2006-01-02"),
			"environment":   "linux",
			"structure":     []string{},
			"isGitRepo":     false,
			"currentBranch": "",
			"mainBranch":    "",
			"gitStatus":     "",
			"recentCommits": []string{},
		},
		"params": params,
	}
	result, err := json.Marshal(ccReq)
	if err != nil {
		return openaiPayload
	}
	return result
}

func flattenContent(content interface{}) string {
	if content == nil {
		return ""
	}
	switch v := content.(type) {
	case string:
		return v
	case []interface{}:
		var parts []string
		for _, p := range v {
			if s, ok := p.(string); ok {
				parts = append(parts, s)
			}
			if obj, ok := p.(map[string]interface{}); ok {
				if t, ok := obj["text"].(string); ok {
					parts = append(parts, t)
				}
			}
		}
		return strings.Join(parts, "\n")
	default:
		return fmt.Sprintf("%v", content)
	}
}

func toContentBlocks(content interface{}) []map[string]interface{} {
	if content == nil {
		return []map[string]interface{}{{"type": "text", "text": ""}}
	}
	switch v := content.(type) {
	case string:
		return []map[string]interface{}{{"type": "text", "text": v}}
	case []interface{}:
		var blocks []map[string]interface{}
		for _, part := range v {
			if s, ok := part.(string); ok {
				blocks = append(blocks, map[string]interface{}{"type": "text", "text": s})
			} else if obj, ok := part.(map[string]interface{}); ok {
				if t, ok := obj["type"].(string); ok && t != "text" {
					blocks = append(blocks, obj)
				} else if text, ok := obj["text"].(string); ok {
					blocks = append(blocks, map[string]interface{}{"type": "text", "text": text})
				}
			}
		}
		if len(blocks) == 0 {
			return []map[string]interface{}{{"type": "text", "text": ""}}
		}
		return blocks
	default:
		return []map[string]interface{}{{"type": "text", "text": fmt.Sprintf("%v", content)}}
	}
}

func toCommandCodeTools(tools interface{}) interface{} {
	if tools == nil {
		return nil
	}
	slice, ok := tools.([]interface{})
	if !ok {
		return tools
	}
	var out []map[string]interface{}
	for _, item := range slice {
		m, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		if _, hasName := m["name"]; hasName {
			if _, hasSchema := m["input_schema"]; hasSchema {
				out = append(out, m)
				continue
			}
		}
		fn, ok := m["function"].(map[string]interface{})
		if !ok {
			out = append(out, m)
			continue
		}
		tool := map[string]interface{}{
			"name": fn["name"],
		}
		if desc, ok := fn["description"].(string); ok && desc != "" {
			tool["description"] = desc
		}
		if params, ok := fn["parameters"].(map[string]interface{}); ok {
			tool["input_schema"] = params
		} else {
			tool["input_schema"] = map[string]interface{}{
				"type": "object",
			}
		}
		out = append(out, tool)
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func toCommandCodeToolChoice(toolChoice interface{}) interface{} {
	if toolChoice == nil {
		return nil
	}
	switch v := toolChoice.(type) {
	case string:
		switch v {
		case "auto":
			return map[string]interface{}{"type": "auto"}
		case "required":
			return map[string]interface{}{"type": "any"}
		case "none":
			return nil
		default:
			return map[string]interface{}{"type": "auto"}
		}
	case map[string]interface{}:
		if t, ok := v["type"].(string); ok {
			if t == "function" {
				if fn, ok := v["function"].(map[string]interface{}); ok {
					if name, ok := fn["name"].(string); ok {
						return map[string]interface{}{
							"type": "tool",
							"name": name,
						}
					}
				}
			}
		}
		return v
	default:
		return nil
	}
}

// ============================================================
// Response transform: NDJSON → OpenAI (non-streaming)
// ============================================================

// TransformCommandCodeResponseToOpenAI transforms a complete CommandCode NDJSON response body into an OpenAI Chat Completion response.
func TransformCommandCodeResponseToOpenAI(model string, ndjsonBody []byte) []byte {
	var content strings.Builder
	var reasoning strings.Builder

	type toolCallBuilder struct {
		id        string
		name      string
		arguments strings.Builder
	}
	var toolOrder []string
	toolMap := make(map[string]*toolCallBuilder)

	lines := strings.Split(string(ndjsonBody), "\n")
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || trimmed == "[DONE]" {
			continue
		}
		var event map[string]interface{}
		if err := json.Unmarshal([]byte(trimmed), &event); err != nil {
			continue
		}
		eventType, _ := event["type"].(string)
		switch eventType {
		case "text-delta":
			if t := ccGetStr(event, "text", "delta"); t != "" {
				content.WriteString(t)
			}
		case "reasoning-delta":
			if t := ccGetStr(event, "text", "delta"); t != "" {
				reasoning.WriteString(t)
			}
		case "tool-input-start":
			id := ccGetStr(event, "id", "toolCallId")
			name := ccGetStr(event, "toolName", "name")
			if id != "" {
				if _, exists := toolMap[id]; !exists {
					toolOrder = append(toolOrder, id)
					toolMap[id] = &toolCallBuilder{id: id, name: name}
				}
			}
		case "tool-input-delta":
			id := ccGetStr(event, "id", "toolCallId")
			delta := ccGetStr(event, "delta", "inputTextDelta")
			if b, exists := toolMap[id]; exists {
				b.arguments.WriteString(delta)
			}
		case "tool-call":
			id := ccGetStr(event, "toolCallId", "id")
			name := ccGetStr(event, "toolName", "name")
			argsStr := "{}"
			if input, ok := event["input"]; ok {
				if s, ok := input.(string); ok {
					argsStr = s
				} else if b, err := json.Marshal(input); err == nil {
					argsStr = string(b)
				}
			}
			if b, exists := toolMap[id]; exists {
				if b.arguments.Len() == 0 {
					b.arguments.WriteString(argsStr)
				}
			} else if id != "" {
				toolOrder = append(toolOrder, id)
				tb := &toolCallBuilder{id: id, name: name}
				tb.arguments.WriteString(argsStr)
				toolMap[id] = tb
			}
		case "error":
			errMsg := ccExtractErrorMessage(event)
			if content.Len() > 0 {
				content.WriteString("\n\n")
			}
			content.WriteString(fmt.Sprintf("[CommandCode error: %s]", errMsg))
		}
	}

	message := map[string]interface{}{
		"role":    "assistant",
		"content": content.String(),
	}
	if reasoning.Len() > 0 {
		message["reasoning_content"] = reasoning.String()
	}

	finishReason := "stop"
	if len(toolOrder) > 0 {
		var toolCalls []map[string]interface{}
		for i, id := range toolOrder {
			tb := toolMap[id]
			args := tb.arguments.String()
			if args == "" {
				args = "{}"
			}
			toolCalls = append(toolCalls, map[string]interface{}{
				"index": i,
				"id":    tb.id,
				"type":  "function",
				"function": map[string]interface{}{
					"name":      tb.name,
					"arguments": args,
				},
			})
		}
		message["tool_calls"] = toolCalls
		finishReason = "tool_calls"
	}

	if model == "" {
		model = "commandcode"
	}

	totalTokens := (content.Len() + reasoning.Len()) / 4
	if totalTokens == 0 && (content.Len() > 0 || reasoning.Len() > 0) {
		totalTokens = 1
	}

	result := map[string]interface{}{
		"id":      fmt.Sprintf("chatcmpl-cc-%d", time.Now().UnixMilli()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"message":       message,
				"finish_reason": finishReason,
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     0,
			"completion_tokens": totalTokens,
			"total_tokens":      totalTokens,
		},
	}
	out, _ := json.Marshal(result)
	return out
}

// ============================================================
// Streaming: NDJSON event → OpenAI chunk
// ============================================================

// CCStreamState tracks streaming state across multiple NDJSON events.
type CCStreamState struct {
	Model         string
	ResponseID    string
	Created       int64
	ChunkIndex    int
	ToolIndex     int
	ToolIndexByID map[string]int
}

// NewCCStreamState creates a new CCStreamState.
func NewCCStreamState(model string) *CCStreamState {
	if model == "" {
		model = "commandcode"
	}
	return &CCStreamState{
		Model:         model,
		ResponseID:    fmt.Sprintf("chatcmpl-cc-%d", time.Now().UnixMilli()),
		Created:       time.Now().Unix(),
		ToolIndexByID: make(map[string]int),
	}
}

func (s *CCStreamState) ensureInitialized() {
	if s.ResponseID == "" {
		s.ResponseID = fmt.Sprintf("chatcmpl-cc-%d", time.Now().UnixMilli())
		s.Created = time.Now().Unix()
	}
	if s.ToolIndexByID == nil {
		s.ToolIndexByID = make(map[string]int)
	}
}

// CCEventToOpenAIChunks converts a single NDJSON line to OpenAI SSE chunk payloads.
func CCEventToOpenAIChunks(line string, state *CCStreamState) [][]byte {
	if state == nil {
		state = NewCCStreamState("")
	}
	state.ensureInitialized()

	line = strings.TrimSpace(line)
	if line == "" || line == "[DONE]" {
		return nil
	}

	var event map[string]interface{}
	if err := json.Unmarshal([]byte(line), &event); err != nil {
		return nil
	}
	eventType, _ := event["type"].(string)

	// Already an OpenAI chunk? pass through
	if obj, ok := event["object"].(string); ok && obj == "chat.completion.chunk" {
		return [][]byte{[]byte(line)}
	}

	var out [][]byte
	switch eventType {
	case "text-delta":
		text := ccGetStr(event, "text", "delta")
		if text == "" {
			break
		}
		delta := map[string]interface{}{"content": text}
		if state.ChunkIndex == 0 {
			delta["role"] = "assistant"
		}
		state.ChunkIndex++
		out = append(out, ccMakeChunk(state, delta, nil))

	case "reasoning-delta":
		text := ccGetStr(event, "text", "delta")
		if text == "" {
			break
		}
		delta := map[string]interface{}{"reasoning_content": text}
		if state.ChunkIndex == 0 {
			delta["role"] = "assistant"
		}
		state.ChunkIndex++
		out = append(out, ccMakeChunk(state, delta, nil))

	case "tool-input-start":
		id := ccGetStr(event, "id", "toolCallId")
		idx := state.ToolIndex
		state.ToolIndexByID[id] = idx
		state.ToolIndex++
		name := ccGetStr(event, "toolName", "name")
		delta := map[string]interface{}{
			"tool_calls": []map[string]interface{}{
				{
					"index": idx,
					"id":    id,
					"type":  "function",
					"function": map[string]interface{}{
						"name":      name,
						"arguments": "",
					},
				},
			},
		}
		if state.ChunkIndex == 0 {
			delta["role"] = "assistant"
		}
		state.ChunkIndex++
		out = append(out, ccMakeChunk(state, delta, nil))

	case "tool-input-delta":
		id := ccGetStr(event, "id", "toolCallId")
		idx, ok := state.ToolIndexByID[id]
		if !ok {
			break
		}
		delta := map[string]interface{}{
			"tool_calls": []map[string]interface{}{
				{
					"index": idx,
					"function": map[string]interface{}{
						"arguments": ccGetStr(event, "delta", "inputTextDelta"),
					},
				},
			},
		}
		out = append(out, ccMakeChunk(state, delta, nil))

	case "tool-call":
		id := ccGetStr(event, "toolCallId", "id")
		idx, exists := state.ToolIndexByID[id]
		if !exists {
			idx = state.ToolIndex
			state.ToolIndexByID[id] = idx
			state.ToolIndex++
		}
		input := event["input"]
		argsStr := "{}"
		if s, ok := input.(string); ok {
			argsStr = s
		} else if b, err := json.Marshal(input); err == nil {
			argsStr = string(b)
		}
		name := ccGetStr(event, "toolName", "name")
		delta := map[string]interface{}{
			"tool_calls": []map[string]interface{}{
				{
					"index": idx,
					"id":    id,
					"type":  "function",
					"function": map[string]interface{}{
						"name":      name,
						"arguments": argsStr,
					},
				},
			},
		}
		if state.ChunkIndex == 0 {
			delta["role"] = "assistant"
		}
		state.ChunkIndex++
		out = append(out, ccMakeChunk(state, delta, nil))

	case "finish-step":
		fr := ccFinishReason(ccGetStr(event, "finishReason"))
		out = append(out, ccMakeChunk(state, map[string]interface{}{}, &fr))

	case "finish":
		fr := ccFinishReason(ccGetStr(event, "finishReason"))
		if fr == "" {
			fr = "stop"
		}
		out = append(out, ccMakeChunk(state, map[string]interface{}{}, &fr))

	case "error":
		errMsg := ccExtractErrorMessage(event)
		fr := "stop"
		out = append(out, ccMakeChunk(state, map[string]interface{}{
			"content": fmt.Sprintf("\n\n[CommandCode error: %s]", errMsg),
		}, &fr))
	}
	return out
}

func ccMakeChunk(state *CCStreamState, delta map[string]interface{}, finishReason *string) []byte {
	choice := map[string]interface{}{
		"index": 0,
		"delta": delta,
	}
	if finishReason != nil {
		choice["finish_reason"] = *finishReason
	}
	chunk := map[string]interface{}{
		"id":      state.ResponseID,
		"object":  "chat.completion.chunk",
		"created": state.Created,
		"model":   state.Model,
		"choices": []interface{}{choice},
	}
	result, _ := json.Marshal(chunk)
	return result
}

func ccGetStr(event map[string]interface{}, keys ...string) string {
	for _, k := range keys {
		if v, ok := event[k]; ok {
			if s, ok := v.(string); ok && s != "" {
				return s
			}
		}
	}
	return ""
}

func ccExtractErrorMessage(event map[string]interface{}) string {
	if errObj, ok := event["error"]; ok {
		if s, ok := errObj.(string); ok && s != "" {
			return s
		}
		if m, ok := errObj.(map[string]interface{}); ok {
			if s, ok := m["message"].(string); ok && s != "" {
				return s
			}
		}
	}
	if s, ok := event["message"].(string); ok && s != "" {
		return s
	}
	return "unknown error"
}

func ccFinishReason(reason string) string {
	switch reason {
	case "tool_use":
		return "tool_calls"
	case "end_turn", "stop":
		return "stop"
	case "max_tokens":
		return "length"
	default:
		return "stop"
	}
}
