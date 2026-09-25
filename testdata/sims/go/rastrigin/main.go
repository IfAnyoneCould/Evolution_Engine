package main

// rastrigin in go, same scoring as examples/rastrigin_sim.py without the busy
// work. cheap enough to throw huge populations at
// args: n dims, span (half width of each range, 5.12 is the classic)

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

const A = 10.0

func main() {
	n := int(arg(1, 10))
	span := arg(2, 5.12)

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

func rastrigin(x []float64) float64 {
	total := A * float64(len(x))
	for _, v := range x {
		total += v*v - A*math.Cos(2*math.Pi*v)
	}
	return total
}

func compute(weights []float64, span float64) float64 {
	n := float64(len(weights))
	worst := A*n + n*(span*span+A) // rough upper bound, same as the python one
	return math.Trunc(math.Max(0, 1-rastrigin(weights)/worst)*10000) / 10000
}

func arg(i int, def float64) float64 {
	if len(os.Args) > i {
		if v, err := strconv.ParseFloat(os.Args[i], 64); err == nil {
			return v
		}
	}
	return def
}
