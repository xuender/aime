package prob_test

import (
	"fmt"

	"github.com/xuender/aime/prob"
)

func ExampleProb_Proto() {
	pro := prob.NewProb[int]()
	pro.Add([]int{1, 2, 3, 4})
	pro.Add([]int{5, 6, 7, 8})

	newPro := prob.Load[int](pro.Proto())

	fmt.Println(newPro.Total())

	// Output:
	// 2
}
