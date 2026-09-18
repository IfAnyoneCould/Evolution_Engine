package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

func main() {
	mode := "sum"
	if len(os.Args) > 1 {
		mode = os.Args[1]
	}
	if mode == "startfail" {
		os.Exit(1)
	}

	in := bufio.NewScanner(os.Stdin)
	count := 0
	for in.Scan() {
		line := strings.TrimSpace(in.Text())
		if line == "" {
			continue
		}
		var weights []float64
		if err := json.Unmarshal([]byte(line), &weights); err != nil {
			fmt.Fprintf(os.Stderr, "bad input %q\n", line)
			os.Exit(1)
		}
		count++

		switch mode {
		case "sum":
			fmt.Println(sum(weights))
		case "negsq":
			fmt.Println(negSquares(weights))
		case "len":
			fmt.Println(float64(len(weights)))
		case "count":
			fmt.Println(float64(count))
		case "garbage":
			fmt.Println("not a number")
		case "die":
			os.Exit(1)
		default:
			fmt.Fprintf(os.Stderr, "unknown mode %q\n", mode)
			os.Exit(1)
		}
	}
}

func sum(weights []float64) float64 {
	total := 0.0
	for _, w := range weights {
		total += w
	}
	return total
}

func negSquares(weights []float64) float64 {
	total := 0.0
	for _, w := range weights {
		total -= w * w
	}
	return total
}
