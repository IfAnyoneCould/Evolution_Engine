package main

// rastrigin that burns real cpu per eval instead of sleeping, so the cost can be
// dialed from nothing up to a second or so and the pool actually has to split
// cores. the burn is filler like busy_work in examples/rastrigin_sim.py
// args: n dims, work iters (about 50M is a second, depends on the machine), span

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

var sink float64 // keeps the burn from getting optimized out

func main() {
	n := int(arg(1, 10))
	iters := int(arg(2, 5000000))
	span := arg(3, 5.12)

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
		sink += burn(iters, weights[0])
		fmt.Println(compute(weights, span))
	}
}

func burn(iters int, seed float64) float64 {
	acc := 0.0
	v := seed
	for range iters {
		v = v*1.000001 + 0.5
		acc += math.Cos(v)*math.Cos(v) + (v - math.Floor(v))
	}
	return acc
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
	worst := A*n + n*(span*span+A)
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
