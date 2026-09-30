package api

import (
	"encoding/json"
	"fmt"
	"io"
)

type Event interface {
	kind() string
}

type Start struct {
	Seed      int64  `json:"seed"`
	Optimizer string `json:"optimizer"`
}
type Generation struct {
	Cycle      int       `json:"cycle"`
	Best       float64   `json:"best"`
	Mean       float64   `json:"mean"`
	Min        float64   `json:"min"`
	Spread     float64   `json:"spread"`
	BestGenome []float64 `json:"best_genome"`
	BatchTime  float64   `json:"batch_time_ms"`
}
type Done struct {
	Reason     string    `json:"reason"`
	Best       float64   `json:"best"`
	BestGenome []float64 `json:"best_genome"`
	Cycles     int       `json:"cycles"`
	Total      float64   `json:"total_ms"`
	Stall      int       `json:"stall"`
}
type Error struct {
	Message string `json:"message"`
}

func (Start) kind() string      { return "start" }
func (Generation) kind() string { return "generation" }
func (Done) kind() string       { return "done" }
func (Error) kind() string      { return "error" }

type Emitter interface {
	Send(e Event) error
}

type Sender struct {
	enc *json.Encoder
}

func NewSender(w io.Writer) *Sender {
	return &Sender{json.NewEncoder(w)}
}
func (s *Sender) Send(e Event) error {
	return s.enc.Encode(struct {
		Type string `json:"type"`
		Data Event  `json:"data"`
	}{e.kind(), e})
}

type Printer struct{}

func NewPrinter() *Printer {
	return &Printer{}
}
func (p *Printer) Send(e Event) error {
	switch v := e.(type) {
	case Start:
		fmt.Printf("Optimizer: %s with random seed %d", v.Optimizer, v.Seed)
	case Generation:
		fmt.Printf("Cycle: %d Best: %f Mean: %f Spread: %f BatchTimeMS: %f", v.Cycle, v.Best, v.Mean, v.Spread, v.BatchTime)
	case Done:
		switch v.Reason {
		case "stalled":
			fmt.Printf("Simulation excited, no improvement in %d cycle ")
		}
	}
	return nil
}
