package helps

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/tidwall/gjson"
)

type vercelOpenAIMsg struct {
	Role       string      `json:"role"`
	Content    interface{} `json:"content"`
	ToolCalls  interface{} `json:"tool_calls,omitempty"`
	ToolCallID string      `json:"tool_call_id,omitempty"`
	Name       string      `json:"name,omitempty"`
}

type vercelToolCallAccumulator struct {
	id   string
	name string
	args strings.Builder
}

// GenerateVercelUUID generates a random UUID v4 string.
func GenerateVercelUUID() string {
	return uuid.New().String()
}

// TransformToVercel translates an OpenAI Chat Completions payload into a Vercel AI Gateway v3/v4 payload.
func TransformToVercel(model string, openaiPayload []byte) []byte {
	var body struct {
		Messages            []vercelOpenAIMsg `json:"messages"`
		Temperature         *float64          `json:"temperature"`
		TopP                *float64          `json:"top_p"`
		MaxTokens           *int              `json:"max_tokens"`
		MaxCompletionTokens *int              `json:"max_completion_tokens"`
		Tools               json.RawMessage   `json:"tools"`
		ToolChoice          json.RawMessage   `json:"tool_choice"`
	}
	if err := json.Unmarshal(openaiPayload, &body); err != nil {
		return openaiPayload
	}

	var prompt []map[string]interface{}
	toolNamesByCallID := make(map[string]string)

	for _, msg := range body.Messages {
		if msg.Role == "system" {
			text := flattenContent(msg.Content)
			if text != "" {
				prompt = append(prompt, map[string]interface{}{
					"role": "system",
					"content": []map[string]interface{}{
						{
							"type": "text",
							"text": text,
						},
					},
				})
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

			var resultObj interface{}
			if err := json.Unmarshal([]byte(contentStr), &resultObj); err != nil {
				resultObj = contentStr
			}

			prompt = append(prompt, map[string]interface{}{
				"role": "tool",
				"content": []map[string]interface{}{
					{
						"type":       "tool-result",
						"toolCallId": callID,
						"toolName":   toolName,
						"result":     resultObj,
					},
				},
			})
			continue
		}

		var contentParts []map[string]interface{}
		if text := flattenContent(msg.Content); text != "" {
			contentParts = append(contentParts, map[string]interface{}{
				"type": "text",
				"text": text,
			})
		}

		if msg.Role == "assistant" && msg.ToolCalls != nil {
			if tcSlice, ok := msg.ToolCalls.([]interface{}); ok {
				for _, tc := range tcSlice {
					if tcMap, ok := tc.(map[string]interface{}); ok {
						id, _ := tcMap["id"].(string)
						var name string
						var argsObj interface{}
						if fnMap, ok := tcMap["function"].(map[string]interface{}); ok {
							name, _ = fnMap["name"].(string)
							if argsStr, ok := fnMap["arguments"].(string); ok && argsStr != "" {
								if err := json.Unmarshal([]byte(argsStr), &argsObj); err != nil {
									argsObj = map[string]interface{}{}
								}
							} else if argsM, ok := fnMap["arguments"].(map[string]interface{}); ok {
								argsObj = argsM
							}
						}
						if id != "" && name != "" {
							toolNamesByCallID[id] = name
							if argsObj == nil {
								argsObj = map[string]interface{}{}
							}
							contentParts = append(contentParts, map[string]interface{}{
								"type":       "tool-call",
								"toolCallId": id,
								"toolName":   name,
								"args":       argsObj,
							})
						}
					}
				}
			}
		}

		if len(contentParts) == 0 {
			contentParts = append(contentParts, map[string]interface{}{
				"type": "text",
				"text": " ",
			})
		}

		prompt = append(prompt, map[string]interface{}{
			"role":    msg.Role,
			"content": contentParts,
		})
	}

	temperature := 0.3
	if body.Temperature != nil {
		temperature = *body.Temperature
	}

	reqPayload := map[string]interface{}{
		"prompt":      prompt,
		"temperature": temperature,
	}

	if body.MaxCompletionTokens != nil {
		reqPayload["max_tokens"] = *body.MaxCompletionTokens
	} else if body.MaxTokens != nil {
		reqPayload["max_tokens"] = *body.MaxTokens
	}
	if body.TopP != nil {
		reqPayload["top_p"] = *body.TopP
	}
	if vTools := toVercelTools(body.Tools); vTools != nil {
		reqPayload["tools"] = vTools
	}
	if vChoice := toVercelToolChoice(body.ToolChoice); vChoice != nil {
		reqPayload["tool_choice"] = vChoice
	}

	res, err := json.Marshal(reqPayload)
	if err != nil {
		return openaiPayload
	}
	return res
}

func toVercelTools(toolsRaw json.RawMessage) []map[string]interface{} {
	if len(toolsRaw) == 0 {
		return nil
	}
	var tools []struct {
		Type     string `json:"type"`
		Function struct {
			Name        string                 `json:"name"`
			Description string                 `json:"description"`
			Parameters  map[string]interface{} `json:"parameters"`
		} `json:"function"`
	}
	if err := json.Unmarshal(toolsRaw, &tools); err != nil || len(tools) == 0 {
		return nil
	}

	var result []map[string]interface{}
	for _, t := range tools {
		if t.Function.Name == "" {
			continue
		}
		params := t.Function.Parameters
		if params == nil {
			params = map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			}
		}
		result = append(result, map[string]interface{}{
			"type":        "function",
			"name":        t.Function.Name,
			"description": t.Function.Description,
			"parameters":  params,
		})
	}
	return result
}

func toVercelToolChoice(choiceRaw json.RawMessage) interface{} {
	if len(choiceRaw) == 0 {
		return nil
	}
	str := strings.TrimSpace(string(choiceRaw))
	if str == `"auto"` || str == `"none"` || str == `"required"` {
		return strings.Trim(str, `"`)
	}
	var obj map[string]interface{}
	if err := json.Unmarshal(choiceRaw, &obj); err == nil {
		return obj
	}
	return nil
}

// TransformVercelResponseToOpenAI converts an entire Vercel AI Gateway response to OpenAI non-streaming JSON.
func TransformVercelResponseToOpenAI(model string, ssePayload []byte) ([]byte, error) {
	var fullText strings.Builder
	var reasoningText strings.Builder
	var toolCalls []map[string]interface{}
	var finishReason string
	var usageInputTokens int64
	var usageOutputTokens int64

	toolMap := make(map[string]*vercelToolCallAccumulator)
	lines := strings.Split(string(ssePayload), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "data:") {
			line = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		}
		if line == "" || line == "[DONE]" {
			continue
		}

		eventType := gjson.Get(line, "type").String()
		switch eventType {
		case "text-delta":
			if delta := gjson.Get(line, "delta").String(); delta != "" {
				fullText.WriteString(delta)
			} else if text := gjson.Get(line, "text").String(); text != "" {
				fullText.WriteString(text)
			}
		case "reasoning-delta":
			if delta := gjson.Get(line, "delta").String(); delta != "" {
				reasoningText.WriteString(delta)
			} else if text := gjson.Get(line, "text").String(); text != "" {
				reasoningText.WriteString(text)
			}
		case "tool-call":
			id := gjson.Get(line, "toolCallId").String()
			name := gjson.Get(line, "toolName").String()
			args := gjson.Get(line, "args").Raw
			if id != "" && name != "" {
				toolCalls = append(toolCalls, map[string]interface{}{
					"id":   id,
					"type": "function",
					"function": map[string]interface{}{
						"name":      name,
						"arguments": args,
					},
				})
			}
		case "tool-input-start":
			id := gjson.Get(line, "id").String()
			name := gjson.Get(line, "toolName").String()
			if id != "" {
				toolMap[id] = &vercelToolCallAccumulator{id: id, name: name}
			}
		case "tool-input-delta":
			id := gjson.Get(line, "id").String()
			delta := gjson.Get(line, "delta").String()
			if acc, ok := toolMap[id]; ok {
				acc.args.WriteString(delta)
			}
		case "finish", "finish-step":
			if r := gjson.Get(line, "finishReason").String(); r != "" {
				finishReason = r
			}
			if in := gjson.Get(line, "usage.inputTokens").Int(); in > 0 {
				usageInputTokens = in
			}
			if out := gjson.Get(line, "usage.outputTokens").Int(); out > 0 {
				usageOutputTokens = out
			}
		}
	}

	for _, acc := range toolMap {
		toolCalls = append(toolCalls, map[string]interface{}{
			"id":   acc.id,
			"type": "function",
			"function": map[string]interface{}{
				"name":      acc.name,
				"arguments": acc.args.String(),
			},
		})
	}

	if finishReason == "" {
		if len(toolCalls) > 0 {
			finishReason = "tool_calls"
		} else {
			finishReason = "stop"
		}
	}

	msgObj := map[string]interface{}{
		"role":    "assistant",
		"content": fullText.String(),
	}
	if reasoningText.Len() > 0 {
		msgObj["reasoning_content"] = reasoningText.String()
	}
	if len(toolCalls) > 0 {
		msgObj["tool_calls"] = toolCalls
		if msgObj["content"] == "" {
			msgObj["content"] = nil
		}
	}

	openaiResp := map[string]interface{}{
		"id":      "chatcmpl-" + GenerateVercelUUID(),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"message":       msgObj,
				"finish_reason": finishReason,
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     usageInputTokens,
			"completion_tokens": usageOutputTokens,
			"total_tokens":      usageInputTokens + usageOutputTokens,
		},
	}

	return json.Marshal(openaiResp)
}

// TransformVercelStreamChunk translates a single Vercel SSE event line into an OpenAI streaming SSE chunk.
func TransformVercelStreamChunk(model string, line string, chatID string) []byte {
	line = strings.TrimSpace(line)
	if strings.HasPrefix(line, "data:") {
		line = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
	}
	if line == "" || line == "[DONE]" {
		return nil
	}

	eventType := gjson.Get(line, "type").String()
	var chunk map[string]interface{}

	switch eventType {
	case "text-delta":
		text := gjson.Get(line, "delta").String()
		if text == "" {
			text = gjson.Get(line, "text").String()
		}
		if text == "" {
			return nil
		}
		chunk = map[string]interface{}{
			"id":      chatID,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"content": text,
					},
					"finish_reason": nil,
				},
			},
		}

	case "reasoning-delta":
		text := gjson.Get(line, "delta").String()
		if text == "" {
			text = gjson.Get(line, "text").String()
		}
		if text == "" {
			return nil
		}
		chunk = map[string]interface{}{
			"id":      chatID,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"reasoning_content": text,
					},
					"finish_reason": nil,
				},
			},
		}

	case "tool-call":
		id := gjson.Get(line, "toolCallId").String()
		name := gjson.Get(line, "toolName").String()
		args := gjson.Get(line, "args").Raw
		chunk = map[string]interface{}{
			"id":      chatID,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"tool_calls": []map[string]interface{}{
							{
								"index": 0,
								"id":    id,
								"type":  "function",
								"function": map[string]interface{}{
									"name":      name,
									"arguments": args,
								},
							},
						},
					},
					"finish_reason": nil,
				},
			},
		}

	case "tool-input-start":
		id := gjson.Get(line, "id").String()
		name := gjson.Get(line, "toolName").String()
		chunk = map[string]interface{}{
			"id":      chatID,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"tool_calls": []map[string]interface{}{
							{
								"index": 0,
								"id":    id,
								"type":  "function",
								"function": map[string]interface{}{
									"name":      name,
									"arguments": "",
								},
							},
						},
					},
					"finish_reason": nil,
				},
			},
		}

	case "tool-input-delta":
		delta := gjson.Get(line, "delta").String()
		chunk = map[string]interface{}{
			"id":      chatID,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"delta": map[string]interface{}{
						"tool_calls": []map[string]interface{}{
							{
								"index": 0,
								"function": map[string]interface{}{
									"arguments": delta,
								},
							},
						},
					},
					"finish_reason": nil,
				},
			},
		}

	case "finish", "finish-step":
		reason := gjson.Get(line, "finishReason").String()
		if reason == "" {
			reason = "stop"
		}
		chunk = map[string]interface{}{
			"id":      chatID,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []map[string]interface{}{
				{
					"index":         0,
					"delta":         map[string]interface{}{},
					"finish_reason": reason,
				},
			},
		}
	}

	if chunk == nil {
		return nil
	}

	b, err := json.Marshal(chunk)
	if err != nil {
		return nil
	}
	return []byte("data: " + string(b) + "\n\n")
}
