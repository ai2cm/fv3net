package main

import (
	"testing"
)

func TestExtractJson(t *testing.T) {
	data := []byte(`{"body": "{\"foo\": \"bar\"}"}`)

	inner, err := extractBody(data)
	if err != nil {
		t.Error(err)
	}
	if string(inner) != `{"foo": "bar"}` {
		t.Fatal("Not as expected")
	}
}
