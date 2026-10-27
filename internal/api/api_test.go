package api

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"os"
	"strings"
	"testing"
)

func TestSenderOneLinePerEvent(t *testing.T) {
	var buf bytes.Buffer
	s := NewSender(&buf)
	events := []Event{
		Start{Seed: 1, Optimizer: "ga"},
		Generation{Cycle: 0, Best: 1},
		Generation{Cycle: 1, Best: 2},
		Done{Reason: "target"},
	}
	for _, e := range events {
		if err := s.Send(e); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	lines := strings.Split(strings.TrimRight(buf.String(), "\n"), "\n")
	if len(lines) != len(events) {
		t.Fatalf("expected one line per event: expected %d, got %d", len(events), len(lines))
	}
	for i, l := range lines {
		if !json.Valid([]byte(l)) {
			t.Errorf("line %d isn't valid json: %s", i, l)
		}
	}
}

func TestSenderTypes(t *testing.T) {
	tests := []struct {
		e    Event
		kind string
	}{
		{Start{}, "start"},
		{Generation{}, "generation"},
		{Done{}, "done"},
		{Error{}, "error"},
	}
	for _, tt := range tests {
		var buf bytes.Buffer
		if err := NewSender(&buf).Send(tt.e); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		var got struct {
			Type string          `json:"type"`
			Data json.RawMessage `json:"data"`
		}
		if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
			t.Fatalf("not valid json: %v", err)
		}
		if got.Type != tt.kind {
			t.Errorf("incorrect type: expected %q, got %q", tt.kind, got.Type)
		}
		if len(got.Data) == 0 || string(got.Data) == "null" {
			t.Errorf("%s event has no data", tt.kind)
		}
	}
}

func TestSenderRoundTrip(t *testing.T) {
	tests := []Event{
		Start{Seed: 7, Optimizer: "cmaes"},
		Generation{Cycle: 3, Best: 1.5, Mean: 1, Min: 0.5, Spread: 0.2, BestGenome: []float64{1, -2, 3.25}, BatchTime: 12.5},
		Done{Reason: "stalled", Best: 1.5, BestGenome: []float64{1, 2}, Cycles: 40, Total: 900, Stall: 10},
		Error{Message: "sim died"},
	}
	for _, want := range tests {
		var buf bytes.Buffer
		if err := NewSender(&buf).Send(want); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		var raw struct {
			Data json.RawMessage `json:"data"`
		}
		if err := json.Unmarshal(buf.Bytes(), &raw); err != nil {
			t.Fatalf("not valid json: %v", err)
		}

		var got Event
		var err error
		switch want.(type) {
		case Start:
			var v Start
			err = json.Unmarshal(raw.Data, &v)
			got = v
		case Generation:
			var v Generation
			err = json.Unmarshal(raw.Data, &v)
			got = v
		case Done:
			var v Done
			err = json.Unmarshal(raw.Data, &v)
			got = v
		case Error:
			var v Error
			err = json.Unmarshal(raw.Data, &v)
			got = v
		}
		if err != nil {
			t.Fatalf("could not decode %s: %v", want.kind(), err)
		}
		gj, _ := json.Marshal(got)
		wj, _ := json.Marshal(want)
		if !bytes.Equal(gj, wj) {
			t.Errorf("%s didn't round trip: sent %s, got %s", want.kind(), wj, gj)
		}
	}
}

func TestSenderFieldNames(t *testing.T) {
	var buf bytes.Buffer
	_ = NewSender(&buf).Send(Generation{BestGenome: []float64{}})
	for _, k := range []string{"cycle", "best", "mean", "min", "spread", "best_genome", "batch_time_ms"} {
		if !strings.Contains(buf.String(), `"`+k+`"`) {
			t.Errorf("generation is missing %q: %s", k, buf.String())
		}
	}
}

type failWriter struct{}

func (failWriter) Write(p []byte) (int, error) {
	return 0, io.ErrClosedPipe
}

func TestSenderWriteFails(t *testing.T) {
	if err := NewSender(failWriter{}).Send(Start{}); err == nil {
		t.Errorf("expected error when the writer fails")
	}
}

func capture(t *testing.T, f func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("could not make a pipe: %v", err)
	}
	old := os.Stdout
	os.Stdout = w
	f()
	os.Stdout = old
	_ = w.Close()
	b, _ := io.ReadAll(r)
	return string(b)
}

func TestPrinterOneLinePerEvent(t *testing.T) {
	tests := []struct {
		name string
		e    Event
	}{
		{"start", Start{Seed: 1, Optimizer: "ga"}},
		{"generation", Generation{Cycle: 1, Best: 2}},
		{"target", Done{Reason: "target", Best: 1, Cycles: 3}},
		{"max cycles", Done{Reason: "max_cycles", Best: 1, Cycles: 3}},
		{"stalled", Done{Reason: "stalled", Stall: 4}},
		{"error", Error{Message: "sim died"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := capture(t, func() { _ = NewPrinter().Send(tt.e) })
			if !strings.HasSuffix(out, "\n") || strings.Count(out, "\n") != 1 {
				t.Errorf("expected exactly one line, got %q", out)
			}
		})
	}
}

func TestPrinterError(t *testing.T) {
	out := capture(t, func() { _ = NewPrinter().Send(Error{Message: "sim died"}) })
	if !strings.Contains(out, "sim died") {
		t.Errorf("error message didn't print, got %q", out)
	}
}

func TestPrinterGeneration(t *testing.T) {
	out := capture(t, func() { _ = NewPrinter().Send(Generation{Cycle: 12, Best: 0.5}) })
	sc := bufio.NewScanner(strings.NewReader(out))
	sc.Scan()
	if !strings.Contains(sc.Text(), "12") || !strings.Contains(sc.Text(), "0.5") {
		t.Errorf("generation line is missing the cycle or best: %q", sc.Text())
	}
}
