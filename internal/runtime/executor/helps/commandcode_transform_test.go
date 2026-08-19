package helps

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestGenerateCommandCodeUUID(t *testing.T) {
	uuid1 := GenerateCommandCodeUUID()
	uuid2 := GenerateCommandCodeUUID()
	if uuid1 == "" || uuid2 == "" {
		t.Fatalf("expected non-empty UUIDs")
	}
	if uuid1 == uuid2 {
		t.Fatalf("expected different UUIDs, got identical: %s", uuid1)
	}
	if len(uuid1) != 36 {
		t.Fatalf("expected 36-character UUID, got length %d: %s", len(uuid1), uuid1)
	}
}

func TestTransformToCommandCode(t *testing.T) {
	openaiPayload := []byte(`{
		"model": "deepseek-v4-pro",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "Hello world"},
			{"role": "assistant", "content": [{"type": "text", "text": "Hi there"}]}
		],
		"max_tokens": 2048,
		"temperature": 0.7
	}`)

	transformed := TransformToCommandCode("deepseek-v4-pro", openaiPayload)
	var ccReq map[string]interface{}
	if err := json.Unmarshal(transformed, &ccReq); err != nil {
		t.Fatalf("failed to unmarshal transformed payload: %v", err)
	}

	if ccReq["threadId"] == nil || ccReq["threadId"] == "" {
		t.Errorf("expected threadId to be populated")
	}

	params, ok := ccReq["params"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected params object in ccReq")
	}

	if params["model"] != "deepseek-v4-pro" {
		t.Errorf("expected model deepseek-v4-pro, got %v", params["model"])
	}
	if params["system"] != "You are a helpful assistant." {
		t.Errorf("expected system prompt, got %v", params["system"])
	}
	if params["max_tokens"] != float64(2048) {
		t.Errorf("expected max_tokens 2048, got %v", params["max_tokens"])
	}
	if params["temperature"] != float64(0.7) {
		t.Errorf("expected temperature 0.7, got %v", params["temperature"])
	}
}

func TestTransformToCommandCodeWithToolHistory(t *testing.T) {
	openaiPayload := []byte(`{
		"model": "deepseek-v4-pro",
		"messages": [
			{"role": "user", "content": "What is the weather?"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [
					{
						"id": "call_123",
						"type": "function",
						"function": {"name": "get_weather", "arguments": "{\"city\":\"Hanoi\"}"}
					}
				]
			},
			{
				"role": "tool",
				"tool_call_id": "call_123",
				"content": "Sunny, 30C"
			}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "get_weather",
					"description": "Get weather for a city",
					"parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
				}
			}
		]
	}`)

	transformed := TransformToCommandCode("deepseek-v4-pro", openaiPayload)
	var ccReq map[string]interface{}
	if err := json.Unmarshal(transformed, &ccReq); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	params := ccReq["params"].(map[string]interface{})
	if params["tools"] == nil {
		t.Errorf("expected tools in params")
	}

	msgs := params["messages"].([]interface{})
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(msgs))
	}

	asstMsg := msgs[1].(map[string]interface{})
	if asstMsg["tool_calls"] == nil {
		t.Errorf("expected tool_calls on assistant message")
	}

	toolMsg := msgs[2].(map[string]interface{})
	if toolMsg["tool_call_id"] != "call_123" {
		t.Errorf("expected tool_call_id on tool message, got %v", toolMsg["tool_call_id"])
	}
}

func TestTransformCommandCodeResponseToOpenAI(t *testing.T) {
	ndjson := strings.Join([]string{
		`{"type":"reasoning-delta","text":"Let me think about this."}`,
		`{"type":"text-delta","text":"Hello! "}`,
		`{"type":"text-delta","text":"How can I assist you today?"}`,
		`{"type":"finish","finishReason":"stop"}`,
	}, "\n")

	respBytes := TransformCommandCodeResponseToOpenAI("deepseek-v4-pro", []byte(ndjson))
	var resp map[string]interface{}
	if err := json.Unmarshal(respBytes, &resp); err != nil {
		t.Fatalf("failed to unmarshal transformed response: %v", err)
	}

	if resp["model"] != "deepseek-v4-pro" {
		t.Errorf("expected model deepseek-v4-pro, got %v", resp["model"])
	}

	choices, ok := resp["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		t.Fatalf("expected non-empty choices")
	}

	choice0 := choices[0].(map[string]interface{})
	msg := choice0["message"].(map[string]interface{})

	if msg["content"] != "Hello! How can I assist you today?" {
		t.Errorf("unexpected content: %v", msg["content"])
	}
	if msg["reasoning_content"] != "Let me think about this." {
		t.Errorf("unexpected reasoning_content: %v", msg["reasoning_content"])
	}
	if choice0["finish_reason"] != "stop" {
		t.Errorf("unexpected finish_reason: %v", choice0["finish_reason"])
	}
}

func TestTransformCommandCodeResponseWithToolCalls(t *testing.T) {
	ndjson := strings.Join([]string{
		`{"type":"tool-input-start","id":"call_123","toolName":"get_weather"}`,
		`{"type":"tool-input-delta","id":"call_123","delta":"{\"location\":\"Hanoi\"}"}`,
		`{"type":"finish","finishReason":"tool_use"}`,
	}, "\n")

	respBytes := TransformCommandCodeResponseToOpenAI("deepseek-v4-pro", []byte(ndjson))
	var resp map[string]interface{}
	if err := json.Unmarshal(respBytes, &resp); err != nil {
		t.Fatalf("failed to unmarshal transformed response: %v", err)
	}

	choices := resp["choices"].([]interface{})
	choice0 := choices[0].(map[string]interface{})
	msg := choice0["message"].(map[string]interface{})

	if choice0["finish_reason"] != "tool_calls" {
		t.Errorf("expected finish_reason tool_calls, got %v", choice0["finish_reason"])
	}

	toolCalls, ok := msg["tool_calls"].([]interface{})
	if !ok || len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %v", msg["tool_calls"])
	}

	tc0 := toolCalls[0].(map[string]interface{})
	if tc0["id"] != "call_123" {
		t.Errorf("expected id call_123, got %v", tc0["id"])
	}
	fn := tc0["function"].(map[string]interface{})
	if fn["name"] != "get_weather" {
		t.Errorf("expected function name get_weather, got %v", fn["name"])
	}
	if fn["arguments"] != "{\"location\":\"Hanoi\"}" {
		t.Errorf("expected function arguments, got %v", fn["arguments"])
	}
}

func TestCCEventToOpenAIChunks(t *testing.T) {
	state := NewCCStreamState("deepseek-v4-pro")

	// Text delta
	chunks := CCEventToOpenAIChunks(`{"type":"text-delta","text":"chunk 1"}`, state)
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	var chunkMap map[string]interface{}
	if err := json.Unmarshal(chunks[0], &chunkMap); err != nil {
		t.Fatalf("failed to parse chunk: %v", err)
	}
	choices := chunkMap["choices"].([]interface{})
	delta := choices[0].(map[string]interface{})["delta"].(map[string]interface{})
	if delta["content"] != "chunk 1" {
		t.Errorf("expected content chunk 1, got %v", delta["content"])
	}
	if delta["role"] != "assistant" {
		t.Errorf("expected role assistant on first chunk, got %v", delta["role"])
	}

	// Reasoning delta
	chunks2 := CCEventToOpenAIChunks(`{"type":"reasoning-delta","text":"thinking..."}`, state)
	if len(chunks2) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks2))
	}
	var chunkMap2 map[string]interface{}
	if err := json.Unmarshal(chunks2[0], &chunkMap2); err != nil {
		t.Fatalf("failed to parse chunk: %v", err)
	}
	delta2 := chunkMap2["choices"].([]interface{})[0].(map[string]interface{})["delta"].(map[string]interface{})
	if delta2["reasoning_content"] != "thinking..." {
		t.Errorf("expected reasoning_content thinking..., got %v", delta2["reasoning_content"])
	}

	// Finish
	chunks3 := CCEventToOpenAIChunks(`{"type":"finish","finishReason":"stop"}`, state)
	if len(chunks3) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks3))
	}
	var chunkMap3 map[string]interface{}
	if err := json.Unmarshal(chunks3[0], &chunkMap3); err != nil {
		t.Fatalf("failed to parse chunk: %v", err)
	}
	choice3 := chunkMap3["choices"].([]interface{})[0].(map[string]interface{})
	if choice3["finish_reason"] != "stop" {
		t.Errorf("expected finish_reason stop, got %v", choice3["finish_reason"])
	}
}
