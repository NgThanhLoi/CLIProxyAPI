package helps

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestGenerateVercelUUID(t *testing.T) {
	uuid1 := GenerateVercelUUID()
	uuid2 := GenerateVercelUUID()
	if uuid1 == "" || uuid2 == "" {
		t.Errorf("expected non-empty UUIDs")
	}
	if uuid1 == uuid2 {
		t.Errorf("expected different UUIDs, got identical %s", uuid1)
	}
}

func TestTransformToVercel(t *testing.T) {
	openaiPayload := []byte(`{
		"model": "zai/glm-5.2",
		"messages": [
			{"role": "system", "content": "You are an assistant."},
			{"role": "user", "content": "Hello!"}
		],
		"temperature": 0.5,
		"max_tokens": 1024
	}`)

	transformed := TransformToVercel("zai/glm-5.2", openaiPayload)
	var vReq map[string]interface{}
	if err := json.Unmarshal(transformed, &vReq); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	prompt, ok := vReq["prompt"].([]interface{})
	if !ok || len(prompt) != 2 {
		t.Fatalf("expected prompt with 2 items, got %v", vReq["prompt"])
	}

	sysMsg := prompt[0].(map[string]interface{})
	if sysMsg["role"] != "system" {
		t.Errorf("expected system role, got %v", sysMsg["role"])
	}

	userMsg := prompt[1].(map[string]interface{})
	if userMsg["role"] != "user" {
		t.Errorf("expected user role, got %v", userMsg["role"])
	}

	if vReq["max_tokens"] != float64(1024) {
		t.Errorf("expected max_tokens 1024, got %v", vReq["max_tokens"])
	}
	if vReq["temperature"] != float64(0.5) {
		t.Errorf("expected temperature 0.5, got %v", vReq["temperature"])
	}
}

func TestTransformToVercelWithToolHistory(t *testing.T) {
	openaiPayload := []byte(`{
		"model": "zai/glm-5.2",
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
				"content": "{\"temp\": 30}"
			}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "get_weather",
					"description": "Get weather",
					"parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
				}
			}
		]
	}`)

	transformed := TransformToVercel("zai/glm-5.2", openaiPayload)
	var vReq map[string]interface{}
	if err := json.Unmarshal(transformed, &vReq); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if vReq["tools"] == nil {
		t.Errorf("expected tools in vReq")
	}

	prompt := vReq["prompt"].([]interface{})
	if len(prompt) != 3 {
		t.Fatalf("expected 3 items in prompt, got %d", len(prompt))
	}

	asstMsg := prompt[1].(map[string]interface{})
	asstContent := asstMsg["content"].([]interface{})
	if len(asstContent) == 0 {
		t.Fatalf("expected assistant content parts")
	}
	tcPart := asstContent[0].(map[string]interface{})
	if tcPart["type"] != "tool-call" || tcPart["toolName"] != "get_weather" {
		t.Errorf("expected tool-call for get_weather, got %v", tcPart)
	}

	toolMsg := prompt[2].(map[string]interface{})
	toolContent := toolMsg["content"].([]interface{})
	trPart := toolContent[0].(map[string]interface{})
	if trPart["type"] != "tool-result" || trPart["toolCallId"] != "call_123" {
		t.Errorf("expected tool-result for call_123, got %v", trPart)
	}
}

func TestTransformVercelResponseToOpenAI(t *testing.T) {
	sseData := strings.Join([]string{
		`data: {"type":"reasoning-delta","text":"Analyzing request."}`,
		`data: {"type":"text-delta","delta":"Hello there!"}`,
		`data: {"type":"finish-step","finishReason":"stop","usage":{"inputTokens":10,"outputTokens":5}}`,
	}, "\n")

	openaiBytes, err := TransformVercelResponseToOpenAI("zai/glm-5.2", []byte(sseData))
	if err != nil {
		t.Fatalf("failed to transform: %v", err)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(openaiBytes, &resp); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	choices := resp["choices"].([]interface{})
	if len(choices) == 0 {
		t.Fatalf("expected choices")
	}
	choice := choices[0].(map[string]interface{})
	msg := choice["message"].(map[string]interface{})
	if msg["content"] != "Hello there!" {
		t.Errorf("expected Hello there!, got %v", msg["content"])
	}
	if msg["reasoning_content"] != "Analyzing request." {
		t.Errorf("expected reasoning text, got %v", msg["reasoning_content"])
	}
	if choice["finish_reason"] != "stop" {
		t.Errorf("expected stop finish_reason, got %v", choice["finish_reason"])
	}
}

func TestTransformVercelStreamChunk(t *testing.T) {
	chunk1 := TransformVercelStreamChunk("zai/glm-5.2", `data: {"type":"text-delta","delta":"Hi"}`, "chat-1")
	if !strings.Contains(string(chunk1), "data: {") || !strings.Contains(string(chunk1), `"content":"Hi"`) {
		t.Errorf("unexpected text chunk: %s", string(chunk1))
	}

	chunk2 := TransformVercelStreamChunk("zai/glm-5.2", `data: {"type":"reasoning-delta","delta":"Thinking"}`, "chat-1")
	if !strings.Contains(string(chunk2), `"reasoning_content":"Thinking"`) {
		t.Errorf("unexpected reasoning chunk: %s", string(chunk2))
	}

	chunk3 := TransformVercelStreamChunk("zai/glm-5.2", `data: {"type":"finish","finishReason":"stop"}`, "chat-1")
	if !strings.Contains(string(chunk3), `"finish_reason":"stop"`) {
		t.Errorf("unexpected finish chunk: %s", string(chunk3))
	}
}
