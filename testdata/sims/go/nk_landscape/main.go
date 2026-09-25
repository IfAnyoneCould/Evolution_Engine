package main

// go port of nk_landscape.py. same model (adjacent neighbors around a ring,
// exact max and min by dp, score rescaled to [0, 1]) but go's rng, so the same
// seed gives a different landscape than the python one
// args: n bits (genome is n, weight > 0.5 is a 1), k, seed

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strconv"
	"strings"
)

func main() {
	n := int(arg(1, 20))
	k := int(arg(2, 3))
	seed := uint64(arg(3, 1))
	k = max(0, min(k, n/2)) // the dp needs n >= 2k

	rng := rand.New(rand.NewPCG(seed, 0))
	tables := make([][]float64, n)
	for i := range tables {
		tables[i] = make([]float64, 1<<(k+1))
		for j := range tables[i] {
			tables[i][j] = rng.Float64()
		}
	}
	lo := extreme(tables, n, k, math.Min)
	hi := extreme(tables, n, k, math.Max)

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
		bits := make([]int, n)
		for i, w := range weights {
			if w > 0.5 {
				bits[i] = 1
			}
		}
		score := (raw(bits, tables, k) - lo) / (hi - lo)
		fmt.Println(math.Trunc(math.Max(0, math.Min(1, score))*10000) / 10000)
	}
}

func raw(bits []int, tables [][]float64, k int) float64 {
	n := len(bits)
	total := 0.0
	for i := range n {
		idx := 0
		for j := range k + 1 {
			idx = idx*2 + bits[(i+j)%n]
		}
		total += tables[i][idx]
	}
	return total / float64(n)
}

// fix the first k bits, walk the ring with the last k bits as the state (packed
// in an int), then close the loop with the wrapped windows
func extreme(tables [][]float64, n, k int, pick func(a, b float64) float64) float64 {
	mask := 1<<k - 1
	states := 1 << k
	best := math.NaN()
	for prefix := range states {
		cur := make([]float64, states)
		for s := range cur {
			cur[s] = math.NaN()
		}
		cur[prefix] = 0
		for j := k; j < n; j++ {
			next := make([]float64, states)
			for s := range next {
				next[s] = math.NaN()
			}
			for s, v := range cur {
				if math.IsNaN(v) {
					continue
				}
				for bit := range 2 {
					w := s<<1 | bit
					val := v + tables[j-k][w]
					key := w & mask
					if math.IsNaN(next[key]) {
						next[key] = val
					} else {
						next[key] = pick(next[key], val)
					}
				}
			}
			cur = next
		}
		for s, v := range cur {
			if math.IsNaN(v) {
				continue
			}
			seq := s<<k | prefix // last k bits then the first k bits
			for m := range k {
				w := (seq >> (k - 1 - m)) & (1<<(k+1) - 1)
				v += tables[n-k+m][w]
			}
			if math.IsNaN(best) {
				best = v
			} else {
				best = pick(best, v)
			}
		}
	}
	return best / float64(n)
}

func arg(i int, def float64) float64 {
	if len(os.Args) > i {
		if v, err := strconv.ParseFloat(os.Args[i], 64); err == nil {
			return v
		}
	}
	return def
}
