package executor

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"strings"
	"time"
)

// ============================================================
// Request transform: OpenAI → CommandCode /alpha/generate
// ============================================================

func generateCommandCodeUUID() string {
	b := make([]byte, 16)
	for i := range b { b[i] = byte(rand.Intn(256)) }
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:])
}

func transformToCommandCode(model string, openaiPayload []byte) []byte {
	var body struct {
		Messages []struct {
			Role    string      `json:"role"`
			Content interface{} `json:"content"`
		} `json:"messages"`
		MaxTokens   *int     `json:"max_tokens,omitempty"`
		Temperature *float64 `json:"temperature,omitempty"`
	}
	if err := json.Unmarshal(openaiPayload, &body); err != nil {
		return openaiPayload
	}

	var systemTexts []string
	var messages []map[string]interface{}

	for _, msg := range body.Messages {
		if msg.Role == "system" {
			text := flattenContent(msg.Content)
			if text != "" { systemTexts = append(systemTexts, text) }
			continue
		}
		messages = append(messages, map[string]interface{}{
			"role":    msg.Role,
			"content": toContentBlocks(msg.Content),
		})
	}

	maxTokens := 4096
	if body.MaxTokens != nil { maxTokens = *body.MaxTokens }
	temperature := 0.3
	if body.Temperature != nil { temperature = *body.Temperature }

	params := map[string]interface{}{
		"model": model, "messages": messages, "stream": true,
		"max_tokens": maxTokens, "temperature": temperature,
	}
	if system := strings.Join(systemTexts, "\n\n"); system != "" {
		params["system"] = system
	}

	ccReq := map[string]interface{}{
		"threadId": generateCommandCodeUUID(), "memory": "",
		"config": map[string]interface{}{
			"workingDir": "/tmp", "date": time.Now().Format("2006-01-02"),
			"environment": "linux", "structure": []string{},
			"isGitRepo": false, "currentBranch": "", "mainBranch": "",
			"gitStatus": "", "recentCommits": []string{},
		},
		"params": params,
	}
	result, _ := json.Marshal(ccReq)
	return result
}

func flattenContent(content interface{}) string {
	if content == nil { return "" }
	switch v := content.(type) {
	case string: return v
	case []interface{}:
		var parts []string
		for _, p := range v {
			if s, ok := p.(string); ok { parts = append(parts, s) }
			if obj, ok := p.(map[string]interface{}); ok {
				if t, ok := obj["text"].(string); ok { parts = append(parts, t) }
			}
		}
		return strings.Join(parts, "\n")
	default: return fmt.Sprintf("%v", content)
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
				if text, ok := obj["text"].(string); ok {
					blocks = append(blocks, map[string]interface{}{"type": "text", "text": text})
				}
			}
		}
		if len(blocks) == 0 { return []map[string]interface{}{{"type": "text", "text": ""}} }
		return blocks
	default:
		return []map[string]interface{}{{"type": "text", "text": fmt.Sprintf("%v", content)}}
	}
}

// ============================================================
// Response transform: NDJSON → OpenAI (non-streaming)
// ============================================================

func transformCommandCodeResponseToOpenAI(ndjsonBody []byte) []byte {
	var content strings.Builder
	var reasoning strings.Builder
	lines := strings.Split(string(ndjsonBody), "\n")
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" { continue }
		var event map[string]interface{}
		if err := json.Unmarshal([]byte(trimmed), &event); err != nil {
			continue
		}
		eventType, _ := event["type"].(string)
		switch eventType {
		case "text-delta":
			if t, ok := event["text"].(string); ok { content.WriteString(t) }
		case "reasoning-delta":
			if t, ok := event["text"].(string); ok { reasoning.WriteString(t) }
		}
	}
	message := map[string]interface{}{"role": "assistant", "content": content.String()}
	if reasoning.Len() > 0 { message["reasoning_content"] = reasoning.String() }
	result := map[string]interface{}{
		"id": fmt.Sprintf("chatcmpl-cc-%d", time.Now().UnixMilli()),
		"object": "chat.completion", "created": time.Now().Unix(),
		"model": "commandcode",
		"choices": []map[string]interface{}{{"index": 0, "message": message, "finish_reason": "stop"}},
		"usage": map[string]interface{}{"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
	}
	out, _ := json.Marshal(result)
	return out
}

// ============================================================
// Streaming: NDJSON event → OpenAI chunk
// ============================================================

type ccStreamState struct {
	model          string
	responseID     string
	created        int64
	chunkIndex     int
	toolIndex      int
	toolIndexByID  map[string]int
}

func ccEnsureState(state *ccStreamState) {
	if state.responseID == "" {
		state.responseID = fmt.Sprintf("chatcmpl-cc-%d", time.Now().UnixMilli())
		state.created = time.Now().Unix()
		state.toolIndexByID = make(map[string]int)
	}
}

// ccEventToOpenAIChunks converts one NDJSON line to OpenAI SSE chunks.
// Follows 9router's commandcode-to-openai.js exactly.
func ccEventToOpenAIChunks(line string, state *ccStreamState) [][]byte {
	ccEnsureState(state)
	line = strings.TrimSpace(line)
	if line == "" || line == "[DONE]" { return nil }

	var event map[string]interface{}
	if err := json.Unmarshal([]byte(line), &event); err != nil { return nil }
	eventType, _ := event["type"].(string)

	// Already OpenAI chunk? pass through
	if obj, ok := event["object"].(string); ok && obj == "chat.completion.chunk" {
		return [][]byte{[]byte(line)}
	}

	var out [][]byte
	switch eventType {
	case "text-delta":
		text := ccGetStr(event, "text", "delta")
		if text == "" { break }
		delta := map[string]interface{}{"content": text}
		if state.chunkIndex == 0 { delta["role"] = "assistant" }
		state.chunkIndex++
		out = append(out, ccChunk(state, delta, nil))

	case "reasoning-delta":
		text := ccGetStr(event, "text")
		if text == "" { break }
		delta := map[string]interface{}{"reasoning_content": text}
		if state.chunkIndex == 0 { delta["role"] = "assistant" }
		state.chunkIndex++
		out = append(out, ccChunk(state, delta, nil))

	case "tool-input-start":
		id := ccGetStr(event, "id", "toolCallId")
		idx := state.toolIndex
		state.toolIndexByID[id] = idx
		state.toolIndex++
		delta := map[string]interface{}{
			"tool_calls": []map[string]interface{}{{
				"index": idx, "id": id, "type": "function",
				"function": map[string]interface{}{"name": ccGetStr(event, "toolName"), "arguments": ""},
			}},
		}
		if state.chunkIndex == 0 { delta["role"] = "assistant" }
		state.chunkIndex++
		out = append(out, ccChunk(state, delta, nil))

	case "tool-input-delta":
		id := ccGetStr(event, "id", "toolCallId")
		idx, ok := state.toolIndexByID[id]
		if !ok { break }
		delta := map[string]interface{}{
			"tool_calls": []map[string]interface{}{{
				"index": idx,
				"function": map[string]interface{}{"arguments": ccGetStr(event, "delta", "inputTextDelta")},
			}},
		}
		out = append(out, ccChunk(state, delta, nil))

	case "tool-call":
		id := ccGetStr(event, "toolCallId")
		if _, exists := state.toolIndexByID[id]; exists { break }
		idx := state.toolIndex
		state.toolIndexByID[id] = idx
		state.toolIndex++
		input := event["input"]
		argsStr := "{}"
		if s, ok := input.(string); ok { argsStr = s } else { b, _ := json.Marshal(input); argsStr = string(b) }
		delta := map[string]interface{}{
			"tool_calls": []map[string]interface{}{{
				"index": idx, "id": id, "type": "function",
				"function": map[string]interface{}{"name": ccGetStr(event, "toolName"), "arguments": argsStr},
			}},
		}
		if state.chunkIndex == 0 { delta["role"] = "assistant" }
		state.chunkIndex++
		out = append(out, ccChunk(state, delta, nil))

	case "finish-step":
		fr := ccFinishReason(ccGetStr(event, "finishReason"))
		out = append(out, ccChunk(state, map[string]interface{}{}, &fr))

	case "finish":
		fr := ccFinishReason(ccGetStr(event, "finishReason"))
		if fr == "" { fr = "stop" }
		out = append(out, ccChunk(state, map[string]interface{}{}, &fr))

	case "error":
		errMsg := ccGetStr(event, "error", "message")
		fr := "stop"
		out = append(out, ccChunk(state, map[string]interface{}{"content": fmt.Sprintf("\n\n[CommandCode error: %s]", errMsg)}, &fr))
	}
	return out
}

func ccChunk(state *ccStreamState, delta map[string]interface{}, finishReason *string) []byte {
	choice := map[string]interface{}{"index": 0, "delta": delta}
	if finishReason != nil { choice["finish_reason"] = *finishReason }
	chunk := map[string]interface{}{
		"object": "chat.completion.chunk", "id": state.responseID,
		"created": state.created, "model": state.model,
		"choices": []interface{}{choice},
	}
	result, _ := json.Marshal(chunk)
	return result
}

func ccGetStr(event map[string]interface{}, keys ...string) string {
	for _, k := range keys {
		if v, ok := event[k]; ok {
			if s, ok := v.(string); ok { return s }
		}
	}
	return ""
}

func ccFinishReason(reason string) string {
	switch reason {
	case "tool_use": return "tool_calls"
	case "end_turn", "stop": return "stop"
	case "max_tokens": return "length"
	default: return "stop"
	}
}
