package main

// sphere in go. mostly here to show the engine doesnt care what language the sim
// is in, and to have an eval so cheap the pool overhead is all thats left
// args: n dims, span (half width of each range)

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

func main() {
	n := int(arg(1, 10))
	span := arg(2, 5)

	in := bufio.NewScanner(os.Stdin)
	in.Buffer(make([]byte, 1024*1024), 1024*1024)
	for in.Scan() {
		line := strings.TrimSpace(in.Text())
		if line == "" {
			continue
		}
		var weights []float64
		if err := json.Unmarshal([]byte(line), &weights); err != nil || len(weights) != n {
			fmt.Println(0.0)
			continue
		}
		fmt.Println(compute(weights, span))
	}
}

func compute(weights []float64, span float64) float64 {
	total := 0.0
	for _, w := range weights {
		total += w * w
	}
	dist := math.Sqrt(total / (float64(len(weights)) * span * span))
	return math.Trunc(math.Max(0, 1-dist)*10000) / 10000
}

func arg(i int, def float64) float64 {
	if len(os.Args) > i {
		if v, err := strconv.ParseFloat(os.Args[i], 64); err == nil {
			return v
		}
	}
	return def
}
